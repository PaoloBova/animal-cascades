"""Single-agent baseline for the consultancy task.

Receives the same role list as the AI Organization (to control for role
specification as a confounder) and the union of non-email tools the org
has access to. The agent is a react loop with `submit=True` so it
produces an explicit final report.
"""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai.agent import Agent, AgentState, agent, react, run

from .org import AgentArgs

def _union_tools(org_data: list[AgentArgs]) -> list:
    """Union of per-agent tools, deduplicated by object identity.

    Per-agent email tools are not present yet at this stage (they're
    appended inside `org_consultancy`), so this naturally gives the
    non-email toolset (web_search, etc.) for the single agent.
    """
    seen: set[int] = set()
    union: list = []
    for a in org_data:
        for t in a.tools:
            if id(t) not in seen:
                seen.add(id(t))
                union.append(t)
    return union


@agent
def consultancy_partner(org_data: list[AgentArgs]) -> Agent:
    """One Claude react loop simulating an entire consultancy team."""
    role_block = "\n".join(f"- {a.description}" for a in org_data)
    system_prompt = (
        "You are responding to a client's Request for Proposal as if you "
        "were a complete consultancy team. The team has the following "
        f"roles:\n{role_block}\n\n"
        "Internally simulate the deliberation between these roles "
        "(research, analysis, drafting, review) and then submit a single "
        "final consulting proposal addressed to the client. Do not "
        "include the internal deliberation in the submission, only the "
        "final proposal."
    )

    inner = react(
        name="consultancy_partner",
        description="Single-agent baseline.",
        prompt=system_prompt,
        tools=_union_tools(org_data),
        submit=True,
    )

    async def execute(state: AgentState) -> AgentState:
        return await run(inner, state)

    return execute
