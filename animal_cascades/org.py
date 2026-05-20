from dataclasses import asdict, dataclass, field

from inspect_ai.agent import Agent, AgentState, agent, react, run
from inspect_ai.model import ChatMessageUser
from inspect_ai.tool import Tool, ToolDef, web_search
from inspect_ai.util import collect, store
import networkx as nx

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


def send_emails_tool(sender: str, contacts: dict[str, str]) -> ToolDef:
    contact_lines = "\n".join(f" - {name}: {role}" for name, role in contacts.items())

    async def execute(recipient_emails: dict[str, str]) -> str:
        unknown = [r for r in recipient_emails if r not in contacts]
        if unknown:
            return (
                f"Error: {', '.join(unknown)} not in your contacts. "
                f"Valid contacts:\n{contact_lines}"
            )
        inboxes = store().get("inboxes")
        for recipient, body in recipient_emails.items():
            inboxes.setdefault(recipient, []).append(
                {"from": sender, "body": body}
            )
        return f"Delivered {len(recipient_emails)} message(s)."

    return ToolDef(
        tool=execute,
        name="send_emails",
        description=(
            "Send one or more emails to colleagues. You may only contact "
            f"people on your contact list:\n{contact_lines}"
        ),
        parameters={
            "recipient_emails": "Mapping from recipient name to email body.",
        },
    )


def _format_inbox(messages: list[dict]) -> str:
    body = "\n\n".join(f"From {m['from']}:\n{m['body']}" for m in messages)
    return f"New messages in your inbox:\n\n{body}"


@agent
def org_consultancy(G: nx.DiGraph, org_data: list[AgentArgs], T: int,) -> Agent:
    """Consultancy structured by directed graph `G`, run for `T` rounds.

    Each round, every agent is prompted with its current inbox and may
    email colleagues reachable via outgoing edges in `G`. After all rounds
    the Communications Director runs a final submission pass and produces
    the consultancy's output.
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
        agent_args.tools.append(send_emails_tool(names[i], contacts))

    # Per-round agents do not submit: react exits each round when the model
    # stops emitting tool calls. A second submit-enabled agent runs once
    # at the very end so the consultancy ends with an explicit final report.
    team = [react(**asdict(a), submit=False) for a in org_data]
    cd_submit_agent = react(**asdict(org_data[idx_cd]), submit=True)

    async def execute(state: AgentState) -> AgentState:
        store().set("inboxes", {n: [] for n in names})
        agent_states = [
            AgentState(messages=list(state.messages)) for _ in team
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
        cd_state = await run(cd_submit_agent, cd_state)
        state.output = cd_state.output
        return state

    return execute


_BASE_CONTEXT = (
    "You are part of an AI consultancy responding to a client's Request "
    "for Proposal. Your team will deliberate over several rounds. Each "
    "round you will see any new messages in your inbox and may use the "
    "send_emails tool to reach colleagues on your contact list. The "
    "Communications Director submits the final report at the end."
)

ROLE_PROMPTS: dict[str, str] = {
    "Communications Director": (
        f"{_BASE_CONTEXT} You are the Communications Director. You "
        "synthesise the team's findings into the final report delivered "
        "to the client. Coordinate the team by emailing directors, "
        "specialists, and interns with direction and requests. At the "
        "end of the consultancy you will be asked to submit the final "
        "report."
    ),
    "Research Director": (
        f"{_BASE_CONTEXT} You are the Research Director. You oversee the "
        "research and analysis effort: direct interns to gather "
        "information, work with specialists to interpret findings, and "
        "keep the Communications Director informed of progress."
    ),
    "Project Manager": (
        f"{_BASE_CONTEXT} You are the Project Manager. You break the "
        "client's request into workstreams, assign tasks to specialists "
        "and interns, track progress, and surface blockers to the "
        "Communications Director."
    ),
    "Policy Analyst": (
        f"{_BASE_CONTEXT} You are the Policy Analyst. You analyse "
        "regulatory, legal, and policy aspects of the client's request "
        "and report findings to managers and peer specialists."
    ),
    "Cost Analysis Specialist": (
        f"{_BASE_CONTEXT} You are the Cost Analysis Specialist. You "
        "model the financial and operational costs of proposed "
        "approaches and report findings to managers and peer specialists."
    ),
    "Operations Specialist": (
        f"{_BASE_CONTEXT} You are the Operations Specialist. You design "
        "the operational implementation of proposed approaches — "
        "logistics, staffing, supply chain, timelines — and report "
        "findings to managers and peer specialists."
    ),
    "Industry Specialist": (
        f"{_BASE_CONTEXT} You are the Industry Specialist. You provide "
        "domain-specific industry expertise relevant to the client's "
        "sector and report findings to managers and peer specialists."
    ),
    "Web Search Intern": (
        f"{_BASE_CONTEXT} You are the Web Search Intern. You research "
        "topics using the web_search tool at the request of specialists "
        "and managers, and report findings back via email."
    ),
    "Data Analysis Intern": (
        f"{_BASE_CONTEXT} You are the Data Analysis Intern. You process "
        "data and produce quantitative analyses at the request of "
        "specialists and managers, and report results back via email."
    ),
    "Document Drafting Intern": (
        f"{_BASE_CONTEXT} You are the Document Drafting Intern. You "
        "produce initial drafts of sections of the final report at the "
        "request of specialists and managers, and share drafts back via "
        "email."
    ),
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


def _role_tools(role: str) -> list[Tool]:
    """Per-role tools, excluding send_emails (added by org_consultancy)."""
    if role == "Web Search Intern":
        return [web_search()]
    return []


# Default bloc order and probability matrix. Entry p[i][j] is the
# probability of a directed edge from a node in bloc i to a node in
# bloc j. Values encode a hierarchy: strong adjacent-tier reporting and
# delegation (M<->S, S<->I), moderate within-tier coordination, weak
# skip-tier links (M<->I).
DEFAULT_BLOCS: tuple[str, ...] = ("manager", "specialist", "intern")

DEFAULT_BLOCK_PROBS: list[list[float]] = [
    # to:    M     S     I
    [      0.6,  0.7,  0.15],   # from M
    [      0.7,  0.4,  0.7 ],   # from S
    [      0.15, 0.7,  0.3 ],   # from I
]

# Named SBM probability presets, selectable from a task kwarg. All are
# 3x3 matrices keyed to DEFAULT_BLOCS (manager, specialist, intern).
BLOCK_PROBS_PRESETS: dict[str, list[list[float]]] = {
    "default": DEFAULT_BLOCK_PROBS,
    "hierarchical": [  # near-pure chain of command, weak peer-to-peer
        [0.4,  0.8,  0.05],
        [0.8,  0.2,  0.8 ],
        [0.05, 0.8,  0.2 ],
    ],
    "flat": [  # uniform connectivity, no hierarchy
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ],
    "siloed": [  # strong within-bloc, weak across blocs
        [0.8,  0.15, 0.05],
        [0.15, 0.8,  0.15],
        [0.05, 0.15, 0.8 ],
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


def build_org_data(
    agent_data: list[AgentConfig] | None = None,
    seed: int = 42,
    blocs: tuple[str, ...] | list[str] = DEFAULT_BLOCS,
    block_probs: list[list[float]] | None = None,
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

    names = _unique_names(agent_data)
    org_data = [
        AgentArgs(
            name=names[i],
            description=a.role,
            prompt=ROLE_PROMPTS[a.role],
            tools=_role_tools(a.role),
        )
        for i, a in enumerate(agent_data)
    ]
    return G, org_data
