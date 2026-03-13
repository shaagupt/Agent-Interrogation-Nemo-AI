"""
Claude API calls for Agent A, Agent B, and the Judge.

Analogy: If prompts.py is the scriptwriter and conditions.py is the playlist,
then this module is the **director** — it puts the actors on stage (calls the
Claude API) and records what they say.

Each function takes the necessary context, calls Claude, and returns the
raw text response. Parsing structured fields (DECISION, CONFIDENCE, etc.)
happens in run_experiment.py.
"""

import anthropic

from harness.prompts import (
    build_agent_b_system_prompt,
    build_agent_b_user_message,
    build_agent_a_system_prompt,
    build_agent_a_user_message,
    build_agent_a_followup_prompt,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message,
    FOLLOW_UP_COUNTS,
)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 1.0  # Fixed temperature for reproducibility across all trials

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# Agent B — The Summarizer
# ---------------------------------------------------------------------------

def call_agent_b(
    paragraph_text: str,
    deception_level: str,
    attack_type: str,
    paragraph_hash: str = "",
) -> str:
    """Send a paragraph to Agent B and get back SUMMARY / HASH / SOURCE.

    Like handing a document to a translator and getting back their version.
    The translator might be honest, sloppy, or actively deceptive depending
    on the deception_level and attack_type.

    paragraph_hash is the precomputed SHA-256 hash so B can report it
    accurately (LLMs can't compute cryptographic hashes).
    """
    client = _get_client()
    system_prompt = build_agent_b_system_prompt(deception_level, attack_type)
    user_message = build_agent_b_user_message(paragraph_text, paragraph_hash)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=TEMPERATURE,
    )
    return response.content[0].text


def call_agent_b_followup(
    conversation_history: list[dict],
    deception_level: str,
    attack_type: str,
) -> str:
    """Agent B responds to a follow-up question from Agent A.

    The full conversation history is passed so B can stay consistent
    (or consistently evasive, if model_tampering is active).
    """
    client = _get_client()
    system_prompt = build_agent_b_system_prompt(deception_level, attack_type)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=conversation_history,
        temperature=TEMPERATURE,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Agent A — The Client / Decision-Maker
# ---------------------------------------------------------------------------

def call_agent_a(
    agent_b_response: str,
    trust_level: str,
) -> str:
    """Agent A evaluates B's initial summary.

    If trust_level is "blind", A will give its final DECISION immediately.
    If skeptical, A will ask its first follow-up question instead.
    """
    client = _get_client()
    system_prompt = build_agent_a_system_prompt(trust_level)
    user_message = build_agent_a_user_message(agent_b_response)

    followup_count = FOLLOW_UP_COUNTS[trust_level]

    messages = [{"role": "user", "content": user_message}]

    # For skeptical agents, nudge them to ask questions first (not decide yet).
    if followup_count > 0:
        messages[0]["content"] += (
            f"\n\n{build_agent_a_followup_prompt(followup_count)}"
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.content[0].text


def call_agent_a_followup(
    conversation_history: list[dict],
    trust_level: str,
    remaining_questions: int,
) -> str:
    """Agent A asks another follow-up or gives its final decision.

    remaining_questions == 0 means A must now provide DECISION / CONFIDENCE.
    """
    client = _get_client()
    system_prompt = build_agent_a_system_prompt(trust_level)

    # Append the nudge about remaining questions.
    nudge = build_agent_a_followup_prompt(remaining_questions)
    history = conversation_history.copy()
    # Add nudge as a user message after B's last response.
    history.append({"role": "user", "content": nudge})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=history,
        temperature=TEMPERATURE,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Judge — The Impartial Evaluator
# ---------------------------------------------------------------------------

def call_judge(
    original_paragraph: str,
    agent_b_summary: str,
    agent_a_decision: str,
    agent_a_understood: str = "",
) -> str:
    """The Judge scores the trial.

    Like a referee watching instant replay — sees the original footage,
    the player's move, the opponent's call, and what A actually understood.
    """
    client = _get_client()
    user_message = build_judge_user_message(
        original_paragraph, agent_b_summary, agent_a_decision, agent_a_understood
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        temperature=TEMPERATURE,
    )
    return response.content[0].text
