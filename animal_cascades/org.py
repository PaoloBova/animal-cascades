import random
from dataclasses import asdict, dataclass, field

from inspect_ai.agent import Agent, AgentState, agent, react, run
from inspect_ai.model import ChatMessageUser, ContentText, UrlCitation
from inspect_ai.tool import Tool, ToolDef, web_search
from inspect_ai.util import collect, store
import networkx as nx


def _make_step_limit(
    max_steps: int | None,
    terminal_tools: tuple[str, ...] = ("send_emails",),
):
    """Return a fresh react `on_continue` that caps a single react invocation.

    Stops the loop when any of these triggers fire:
      - hard step cap reached
      - the model emitted no tool calls (react's default natural end)
      - any tool call in the latest response is a "terminal" tool — once
        the agent has called one of these (e.g. `send_emails`), its round
        is done. This enforces the "once per round, as final action" rule
        structurally rather than via prompt-language alone, while still
        allowing chains of non-terminal tools like `web_search` first.

    Each call constructs a fresh counter, so callers must build a new
    on_continue per react invocation.
    """
    if max_steps is None or max_steps <= 0:
        return None
    counter = {"n": 0}

    async def on_continue(state: AgentState) -> bool:
        counter["n"] += 1
        tool_calls = state.output.message.tool_calls or []
        if not tool_calls:
            return False
        if counter["n"] >= max_steps:
            return False
        if any(tc.function in terminal_tools for tc in tool_calls):
            return False
        return True

    return on_continue

@dataclass
class AgentConfig:
    role: str
    bloc: str

@dataclass
class AgentArgs:
    name: str
    description: str  # role
    prompt: str
    tools: list[Tool] = field(default_factory=list)


def send_emails_tool(
    sender: str,
    contacts: dict[str, str],
    max_email_body_chars: int | None = 1500,
    lean_prompts: bool = True,
) -> ToolDef:
    contact_lines = "\n".join(f" - {name}: {role}" for name, role in contacts.items())
    cap_note = (
        f" Bodies longer than {max_email_body_chars} characters will be "
        "hard-truncated."
        if max_email_body_chars
        else ""
    )
    if lean_prompts:
        intro = (
            "Send one or more emails to colleagues. Bundle all of a "
            "round's outgoing emails into a SINGLE call to this tool — "
            "call it at most once per round, and make it the final "
            "action of your round if you have other tools to use first. "
            "Keep each email body concise: under 200 words, short bullet "
            "points rather than paragraphs, and do not restate context "
            f"the recipient already shares with you.{cap_note}"
        )
    else:
        intro = f"Send one or more emails to colleagues.{cap_note}"

    async def execute(recipient_emails: dict[str, str]) -> str:
        unknown = [r for r in recipient_emails if r not in contacts]
        if unknown:
            return (
                f"Error: {', '.join(unknown)} not in your contacts. "
                f"Valid contacts:\n{contact_lines}"
            )
        inboxes = store().get("inboxes")
        truncated: list[str] = []
        for recipient, body in recipient_emails.items():
            body = body or ""
            if max_email_body_chars and len(body) > max_email_body_chars:
                body = (
                    body[:max_email_body_chars]
                    + f"\n\n[truncated by send_emails — body exceeded {max_email_body_chars} chars]"
                )
                truncated.append(recipient)
            inboxes.setdefault(recipient, []).append(
                {"from": sender, "body": body}
            )
        msg = f"Delivered {len(recipient_emails)} message(s)."
        if truncated:
            msg += (
                f" Truncated {len(truncated)}/{len(recipient_emails)} bodies "
                f"to {max_email_body_chars} chars: {', '.join(truncated)}. "
                "Keep future emails shorter."
            )
        return msg

    return ToolDef(
        tool=execute,
        name="send_emails",
        description=(
            f"{intro}\n\n"
            "You may only contact people on your contact list:\n"
            f"{contact_lines}"
        ),
        parameters={
            "recipient_emails": "Mapping from recipient name to email body.",
        },
    )


def _format_inbox(messages: list[dict]) -> str:
    body = "\n\n".join(f"From {m['from']}:\n{m['body']}" for m in messages)
    return f"New messages in your inbox:\n\n{body}"


@agent
def org_consultancy(
    G: nx.DiGraph,
    org_data: list[AgentArgs],
    T: int,
    max_steps_per_round: int | None = 3,
    max_email_body_chars: int | None = 1500,
    lean_prompts: bool = True,
) -> Agent:
    """Consultancy structured by directed graph `G`, run for `T` rounds.

    Each round, every agent is prompted with its current inbox and may
    email colleagues reachable via outgoing edges in `G`. After all rounds
    the Communications Director runs a final submission pass and produces
    the consultancy's output.

    `max_steps_per_round` caps the number of react loop iterations any one
    agent can run within a single round (prevents a model from chaining
    many tool calls in one turn). Set to None to disable.

    `max_email_body_chars` hard-caps each email body length inside the
    send_emails executor. Bodies over the cap are truncated with a clear
    marker, and the model sees a feedback message in the tool result.
    Set to None for no cap.
    """
    roles = [a.description for a in org_data]
    assert sum(r == "Communications Director" for r in roles) == 1, (
        "Exactly one agent must be the Communications Director."
    )
    idx_cd = roles.index("Communications Director")

    # Defensive copy so re-invocations don't stack the email tool onto the
    # caller's list across samples.
    org_data = [
        AgentArgs(a.name, a.description, a.prompt, list(a.tools))
        for a in org_data
    ]
    names = [a.name for a in org_data]
    for i, agent_args in enumerate(org_data):
        contacts = {
            names[j]: org_data[j].description
            for j in G.successors(i)
        }
        agent_args.tools.append(
            send_emails_tool(
                names[i], contacts,
                max_email_body_chars=max_email_body_chars,
                lean_prompts=lean_prompts,
            )
        )

    async def execute(state: AgentState) -> AgentState:
        store().set("inboxes", {n: [] for n in names})
        agent_states = [
            AgentState(messages=list(state.messages)) for _ in org_data
        ]
        for t in range(T):
            inboxes = store().get("inboxes")
            for i, s in enumerate(agent_states):
                msgs = inboxes.get(names[i], [])
                header = f"Round {t + 1} of {T}."
                content = (
                    f"{header}\n\n{_format_inbox(msgs)}" if msgs
                    else f"{header} Inbox empty. Continue with the task."
                )
                s.messages.append(ChatMessageUser(content=content))
            store().set("inboxes", {n: [] for n in names})
            # Per-round agents do not submit; react exits each round when
            # the model stops emitting tool calls or the step cap is hit.
            # Rebuild each round so every agent gets a fresh step counter.
            team = [
                react(
                    **asdict(a),
                    submit=False,
                    on_continue=_make_step_limit(max_steps_per_round),
                )
                for a in org_data
            ]
            agent_states = await collect(
                *[run(a, s) for a, s in zip(team, agent_states)]
            )

        cd_state = agent_states[idx_cd]
        cd_state.messages.append(ChatMessageUser(
            content=(
                "All consultation rounds are complete. Synthesise the team's "
                "deliberation and submit the final report now."
            )
        ))
        # Final submission: submit=True ends the loop naturally; no step cap.
        cd_submit_agent = react(**asdict(org_data[idx_cd]), submit=True)
        cd_state = await run(cd_submit_agent, cd_state)
        state.output = cd_state.output
        return state

    return execute


# Two variants of the shared "you are on a consultancy team" preamble.
# The lean variant adds the terminal-tool notice ("Calling send_emails ends
# your round immediately…") so the model knows the structural rule.
_BASE_CONTEXT_LEAN = (
    "You are part of an AI consultancy responding to a client's Request "
    "for Proposal. Your team will deliberate over several rounds. Each "
    "round you will see any new messages in your inbox and may use the "
    "send_emails tool to reach colleagues on your contact list. Calling "
    "send_emails ends your round immediately, so do any other tool work "
    "first and use send_emails as your final action of the round. The "
    "Communications Director submits the final report at the end."
)
_BASE_CONTEXT_LEGACY = (
    "You are part of an AI consultancy responding to a client's Request "
    "for Proposal. Your team will deliberate over several rounds. Each "
    "round you will see any new messages in your inbox and may use the "
    "send_emails tool to reach colleagues on your contact list. The "
    "Communications Director submits the final report at the end."
)

# Per-role content (without the base-context prefix). Composed with one of
# the base contexts at format time via `_format_role_prompt`.
ROLE_DESCRIPTIONS: dict[str, str] = {
    "Communications Director": (
        "You are the Communications Director. You synthesise the team's "
        "findings into the final report delivered to the client. "
        "Coordinate the team by emailing directors, specialists, and "
        "interns with direction and requests. At the end of the "
        "consultancy you will be asked to submit the final report."
    ),
    "Research Director": (
        "You are the Research Director. You oversee the research and "
        "analysis effort: direct interns to gather information, work "
        "with specialists to interpret findings, and keep the "
        "Communications Director informed of progress."
    ),
    "Project Manager": (
        "You are the Project Manager. You break the client's request "
        "into workstreams, assign tasks to specialists and interns, "
        "track progress, and surface blockers to the Communications "
        "Director."
    ),
    "Policy Analyst": (
        "You are the Policy Analyst. You analyse regulatory, legal, "
        "and policy aspects of the client's request and report findings "
        "to managers and peer specialists."
    ),
    "Cost Analysis Specialist": (
        "You are the Cost Analysis Specialist. You model the financial "
        "and operational costs of proposed approaches and report findings "
        "to managers and peer specialists."
    ),
    "Operations Specialist": (
        "You are the Operations Specialist. You design the operational "
        "implementation of proposed approaches — logistics, staffing, "
        "supply chain, timelines — and report findings to managers and "
        "peer specialists."
    ),
    "Industry Specialist": (
        "You are the Industry Specialist. You provide domain-specific "
        "industry expertise relevant to the client's sector and report "
        "findings to managers and peer specialists."
    ),
    "Web Search Intern": (
        "You are the Web Search Intern. You research topics using the "
        "web_search tool at the request of specialists and managers, "
        "and report findings back via email."
    ),
    "Data Analysis Intern": (
        "You are the Data Analysis Intern. You process data and produce "
        "quantitative analyses at the request of specialists and "
        "managers, and report results back via email."
    ),
    "Document Drafting Intern": (
        "You are the Document Drafting Intern. You produce initial "
        "drafts of sections of the final report at the request of "
        "specialists and managers, and share drafts back via email."
    ),
}

# Lean-mode-only addenda appended to specific roles. Strings may reference
# `{max_actions}` / `{max_searches}` placeholders filled in at format time.
ROLE_LEAN_ADDENDA: dict[str, str] = {
    "Web Search Intern": (
        "\n\nEach round is time-limited to about {max_actions} actions "
        "total. If you intend to share findings this round, reserve your "
        "final action for a single send_emails call — that means at most "
        "{max_searches} web_search calls before the send_emails. Pick "
        "your search queries deliberately."
    ),
}

# Back-compat: a fully-rendered lean prompt dict (no max_actions filled).
# Anything external that imports `ROLE_PROMPTS` still gets a usable string.
ROLE_PROMPTS: dict[str, str] = {
    role: f"{_BASE_CONTEXT_LEAN} {ROLE_DESCRIPTIONS[role]}{ROLE_LEAN_ADDENDA.get(role, '')}"
    for role in ROLE_DESCRIPTIONS
}

# Role -> tier (bloc).
ROLE_TIERS: dict[str, str] = {
    "Communications Director": "manager",
    "Research Director": "manager",
    "Project Manager": "manager",
    "Policy Analyst": "specialist",
    "Cost Analysis Specialist": "specialist",
    "Operations Specialist": "specialist",
    "Industry Specialist": "specialist",
    "Web Search Intern": "intern",
    "Data Analysis Intern": "intern",
    "Document Drafting Intern": "intern",
}

DEFAULT_ROLE_COUNTS: dict[str, int] = {
    "Communications Director": 1,
    "Research Director": 1,
    "Project Manager": 1,
    "Policy Analyst": 1,
    "Cost Analysis Specialist": 1,
    "Operations Specialist": 1,
    "Industry Specialist": 1,
    "Web Search Intern": 1,
    "Data Analysis Intern": 1,
    "Document Drafting Intern": 1,
}

# Named role-count presets, selectable from a task kwarg.
ROLE_COUNT_PRESETS: dict[str, dict[str, int]] = {
    "default": DEFAULT_ROLE_COUNTS,
    "flat": {
        "Communications Director": 1,
        "Policy Analyst": 1,
        "Cost Analysis Specialist": 1,
        "Operations Specialist": 1,
        "Industry Specialist": 1,
    },
    "top_heavy": {
        "Communications Director": 1,
        "Research Director": 1,
        "Project Manager": 1,
        "Policy Analyst": 1,
        "Cost Analysis Specialist": 1,
    },
    "interns_only": {
        "Communications Director": 1,
        "Web Search Intern": 2,
        "Data Analysis Intern": 2,
        "Document Drafting Intern": 2,
    },
}


_TAVILY_DEFAULTS = {
    "max_results": 5,
    "search_depth": "basic",
    "include_answer": True,
    "include_raw_content": False,
}
# NOTE: Inspect's `ExaCitation` model requires the `text` field on every
# citation, but Exa's /answer endpoint omits that field when `text=False`,
# crashing the parser. So we force `text=True` and rely on the truncation
# wrapper below to keep per-call size bounded.
_EXA_DEFAULTS = {"text": True}
_ANTHROPIC_DEFAULTS = {"max_uses": 1}


def _truncate_content_text(ct: ContentText, budget: int) -> ContentText:
    """Truncate a ContentText so total `text + citations[*].cited_text` <= budget.

    Preserves the synthesized answer in full when possible. Distributes
    remaining budget across citations and trims each `cited_text`. Mutates
    `ct` in place and returns it.
    """
    answer = ct.text or ""
    if len(answer) >= budget:
        ct.text = answer[:budget] + "\n\n[...truncated]"
        ct.citations = []
        return ct

    citations = list(ct.citations or [])
    if not citations:
        return ct

    remaining = budget - len(answer)
    per_citation = max(200, remaining // len(citations))
    truncated_count = 0
    for c in citations:
        cited = getattr(c, "cited_text", None) or ""
        if len(cited) > per_citation:
            c.cited_text = cited[:per_citation] + "..."
            truncated_count += 1
    if truncated_count:
        ct.text = (
            answer
            + f"\n\n[{truncated_count}/{len(citations)} citation excerpts truncated to {per_citation} chars]"
        )
    ct.citations = citations
    return ct


def _make_search_tool(
    provider: str,
    truncate_chars: int | None = 10_000,
) -> Tool:
    """Build a web_search tool, wrapped with output truncation when possible.

    Provider trade-offs:
      - "exa":       /answer endpoint; ~1-3k tokens/call after truncation;
                     requires EXA_API_KEY. (Forces text=True to avoid a bug
                     in Inspect's parser.)
      - "tavily":    Snippet list + synthesized answer; ~1-3k tokens/call;
                     requires TAVILY_API_KEY (free tier: 1000/month).
      - "anthropic": Hosted server-side search; 30-150k tokens/call. No
                     way to bound per-result content size; only `max_uses`
                     caps the total search count.

    Truncation is applied only to external providers (tavily/exa) whose
    results return as `ContentText` we can intercept. Anthropic-hosted
    results are server-side `web_search_tool_result` blocks the SDK
    manages internally and cannot be truncated here.
    """
    if provider == "anthropic":
        return web_search({"anthropic": _ANTHROPIC_DEFAULTS})
    elif provider == "tavily":
        inner = web_search({"tavily": _TAVILY_DEFAULTS})
    elif provider == "exa":
        inner = web_search({"exa": _EXA_DEFAULTS})
    else:
        raise ValueError(
            f"Unknown web_search provider: {provider!r}. "
            "Choose from 'exa', 'tavily', 'anthropic', or 'none'."
        )

    if truncate_chars is None or truncate_chars <= 0:
        return inner

    async def execute(query: str):
        result = await inner(query=query)
        if isinstance(result, ContentText):
            return _truncate_content_text(result, truncate_chars)
        if isinstance(result, str) and len(result) > truncate_chars:
            return (
                result[:truncate_chars]
                + f"\n\n[...truncated to {truncate_chars} chars]"
            )
        return result

    return ToolDef(
        tool=execute,
        name="web_search",
        description=(
            "Search the web for information. Returns a synthesized answer "
            "plus a short citation list. Long results are truncated."
        ),
        parameters={"query": "Search query."},
    ).as_tool()


def _role_tools(
    role: str,
    web_search_provider: str = "exa",
    web_search_truncate_chars: int | None = 10_000,
) -> list[Tool]:
    """Per-role tools, excluding send_emails (added by org_consultancy).

    Pass `web_search_provider="none"` to drop the web_search tool entirely
    (the Web Search Intern will then respond from training knowledge only).
    """
    if role == "Web Search Intern" and web_search_provider != "none":
        return [_make_search_tool(web_search_provider, web_search_truncate_chars)]
    return []


# Default bloc order and probability matrix. Entry p[i][j] is the
# probability of a directed edge from a node in bloc i to a node in
# bloc j. Values encode a hierarchy: strong adjacent-tier reporting and
# delegation (M<->S, S<->I), moderate within-tier coordination, weak
# skip-tier links (M<->I).
DEFAULT_BLOCS: tuple[str, ...] = ("manager", "specialist", "intern")

# All probabilities halved from prior densities to reduce contact-list
# size per agent, which proportionally reduces accumulated email context
# across rounds.
DEFAULT_BLOCK_PROBS: list[list[float]] = [
    # to:     M      S      I
    [      0.30,  0.35,  0.075],   # from M
    [      0.35,  0.20,  0.35 ],   # from S
    [      0.075, 0.35,  0.15 ],   # from I
]

# Named SBM probability presets, selectable from a task kwarg. All are
# 3x3 matrices keyed to DEFAULT_BLOCS (manager, specialist, intern).
BLOCK_PROBS_PRESETS: dict[str, list[list[float]]] = {
    "default": DEFAULT_BLOCK_PROBS,
    "hierarchical": [  # near-pure chain of command, weak peer-to-peer
        [0.20,  0.40,  0.025],
        [0.40,  0.10,  0.40 ],
        [0.025, 0.40,  0.10 ],
    ],
    "flat": [  # uniform connectivity, no hierarchy
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
    ],
    "siloed": [  # strong within-bloc, weak across blocs
        [0.40,  0.075, 0.025],
        [0.075, 0.40,  0.075],
        [0.025, 0.075, 0.40 ],
    ],
    # Un-halved (pre-optimisation) default densities, for A/B sweeps that
    # want to test whether the halving cut meaningful inter-agent comms.
    "legacy_default": [
        [0.6,  0.7,  0.15],
        [0.7,  0.4,  0.7 ],
        [0.15, 0.7,  0.3 ],
    ],
}


def agents_from_counts(
    role_counts: dict[str, int] | None = None,
) -> list[AgentConfig]:
    """Expand a `{role: count}` map into a list of `AgentConfig`s.

    The bloc for each role is looked up from `ROLE_TIERS`. To put a role
    in a non-default bloc (e.g. for an ablation), build the
    `AgentConfig` list directly instead of going through this helper.
    """
    counts = role_counts if role_counts is not None else DEFAULT_ROLE_COUNTS
    return [
        AgentConfig(role=role, bloc=ROLE_TIERS[role])
        for role, n in counts.items()
        for _ in range(n)
    ]


DEFAULT_AGENT_DATA: list[AgentConfig] = agents_from_counts()


def _unique_names(agent_data: list[AgentConfig]) -> list[str]:
    """Bare role name if unique, else `Role #N` for duplicates."""
    totals: dict[str, int] = {}
    for a in agent_data:
        totals[a.role] = totals.get(a.role, 0) + 1
    seen: dict[str, int] = {}
    names: list[str] = []
    for a in agent_data:
        if totals[a.role] == 1:
            names.append(a.role)
        else:
            seen[a.role] = seen.get(a.role, 0) + 1
            names.append(f"{a.role} #{seen[a.role]}")
    return names


def _repair_connectivity(
    G: nx.DiGraph,
    agent_data: list["AgentConfig"],
    seed: int,
) -> int:
    """Make `G` weakly connected and ensure every node has at least one out-edge.

    Two-phase, all edges outgoing-only, same-bloc peer preferred:

    Phase A — bridge non-LCC components. For each weakly-connected component
    other than the largest, pick a source node (preferring one with
    out_degree == 0 so we kill two birds) and add a single edge to a node
    in the LCC (preferring same-bloc).

    Phase B — fix remaining zero-out-degree nodes by adding one outgoing
    edge to a same-bloc peer (or any non-self node as fallback).

    All random picks come from a seeded `random.Random(seed)`, so the same
    `seed` always produces the same repaired graph.
    """
    rng = random.Random(seed)
    added = 0

    def _pick(rng: random.Random, candidates: list[int]) -> int:
        # Sort for determinism (set iteration order isn't guaranteed).
        return rng.choice(sorted(candidates))

    components = list(nx.weakly_connected_components(G))
    if len(components) > 1:
        lcc = max(components, key=lambda c: (len(c), -min(c)))
        lcc_set = set(lcc)
        for comp in components:
            if comp is lcc:
                continue
            zero_out = [n for n in comp if G.out_degree(n) == 0]
            source = _pick(rng, zero_out) if zero_out else _pick(rng, list(comp))

            source_bloc = agent_data[source].bloc
            same_bloc_targets = [
                n for n in lcc_set if agent_data[n].bloc == source_bloc
            ]
            target_pool = same_bloc_targets or list(lcc_set)
            target = _pick(rng, target_pool)

            G.add_edge(source, target)
            added += 1
            lcc_set.update(comp)  # comp is now part of the LCC

    # Phase B: any remaining zero-out-degree nodes.
    all_nodes = list(G.nodes())
    for node in all_nodes:
        if G.out_degree(node) > 0:
            continue
        node_bloc = agent_data[node].bloc
        same_bloc = [
            n for n in all_nodes
            if n != node and agent_data[n].bloc == node_bloc
        ]
        candidates = same_bloc or [n for n in all_nodes if n != node]
        if not candidates:
            continue
        target = _pick(rng, candidates)
        G.add_edge(node, target)
        added += 1

    return added


def _format_role_prompt(
    role: str,
    max_steps_per_round: int | None,
    lean_prompts: bool = True,
) -> str:
    """Return the formatted role prompt.

    `lean_prompts=True` (default) prepends the lean base context (with the
    terminal-tool notice) and appends any role-specific lean addenda
    (e.g. the Web Search Intern's per-round budget paragraph). Set False
    to fall back to the legacy preamble and skip lean addenda — used for
    A/B comparisons against the pre-optimisation prompt language.

    Any `{max_actions}` / `{max_searches}` placeholders in the rendered
    text are filled in from `max_steps_per_round` when lean addenda apply.
    """
    base = _BASE_CONTEXT_LEAN if lean_prompts else _BASE_CONTEXT_LEGACY
    desc = ROLE_DESCRIPTIONS[role]
    addendum = ROLE_LEAN_ADDENDA.get(role, "") if lean_prompts else ""
    full = f"{base} {desc}{addendum}"
    if "{max_actions}" in full or "{max_searches}" in full:
        if max_steps_per_round and max_steps_per_round > 1:
            max_actions = str(max_steps_per_round)
            max_searches = str(max_steps_per_round - 1)
        else:
            max_actions = "several"
            max_searches = "several"
        full = full.format(max_actions=max_actions, max_searches=max_searches)
    return full


def build_org_data(
    agent_data: list[AgentConfig] | None = None,
    seed: int = 42,
    blocs: tuple[str, ...] | list[str] = DEFAULT_BLOCS,
    block_probs: list[list[float]] | None = None,
    web_search_provider: str = "exa",
    web_search_truncate_chars: int | None = 10_000,
    enforce_connectivity: bool = True,
    max_steps_per_round: int | None = 3,
    lean_prompts: bool = True,
) -> tuple[nx.DiGraph, list[AgentArgs]]:
    """Build the consultancy's directed email graph and matching agent specs.

    Returns `(G, org_data)` where node `i` in `G` corresponds to
    `org_data[i]`. Agents are sorted by bloc (in the order given by
    `blocs`) so that nodes within each bloc are contiguous, matching the
    block layout assumed by `nx.stochastic_block_model`.
    """
    agent_data = (
        list(agent_data) if agent_data is not None else list(DEFAULT_AGENT_DATA)
    )
    p = block_probs if block_probs is not None else DEFAULT_BLOCK_PROBS

    assert len(p) == len(blocs) and all(len(row) == len(blocs) for row in p), (
        f"block_probs must be a {len(blocs)}x{len(blocs)} matrix; got "
        f"{len(p)}x{len(p[0]) if p else 0}."
    )
    unknown_roles = [a.role for a in agent_data if a.role not in ROLE_PROMPTS]
    assert not unknown_roles, f"No prompt defined for roles: {unknown_roles}"
    unknown_blocs = [a.bloc for a in agent_data if a.bloc not in blocs]
    assert not unknown_blocs, f"Agents in unknown blocs: {unknown_blocs}"
    cd_count = sum(a.role == "Communications Director" for a in agent_data)
    assert cd_count == 1, (
        f"Need exactly one Communications Director, got {cd_count}."
    )

    bloc_idx = {b: i for i, b in enumerate(blocs)}
    agent_data.sort(key=lambda a: bloc_idx[a.bloc])
    sizes = [sum(1 for a in agent_data if a.bloc == b) for b in blocs]

    G = nx.stochastic_block_model(sizes, p, directed=True, seed=seed)
    G = nx.DiGraph(G)  # drop block-membership node attributes

    if enforce_connectivity:
        _repair_connectivity(G, agent_data, seed=seed)

    names = _unique_names(agent_data)
    org_data = [
        AgentArgs(
            name=names[i],
            description=a.role,
            prompt=_format_role_prompt(
                a.role, max_steps_per_round, lean_prompts=lean_prompts,
            ),
            tools=_role_tools(
                a.role,
                web_search_provider=web_search_provider,
                web_search_truncate_chars=web_search_truncate_chars,
            ),
        )
        for i, a in enumerate(agent_data)
    ]
    return G, org_data
