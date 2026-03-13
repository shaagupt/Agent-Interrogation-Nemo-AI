"""
Main experiment loop.

Analogy: This is the **stage manager** — it reads the playlist (conditions),
calls the director (agents.py) to run each scene, and logs everything
to the database. It orchestrates the full trial from start to finish.

Usage:
    python -m harness.run_experiment                   # run all 33 conditions × 25 paragraphs
    python -m harness.run_experiment --smoke           # smoke test: 1 paragraph × 33 conditions
    python -m harness.run_experiment --condition 5     # run only condition #5
    python -m harness.run_experiment --trials-per 10   # 10 trials per condition instead of 25
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env file from project root
import time
import traceback

from harness.conditions import ALL_CONDITIONS, CONDITIONS_BY_ID
from harness.corruption import corrupt_paragraph
from harness.agents import (
    call_agent_b,
    call_agent_b_followup,
    call_agent_a,
    call_agent_a_followup,
    call_judge,
)
from harness.prompts import (
    FOLLOW_UP_COUNTS,
    build_agent_b_system_prompt,
    build_agent_b_user_message,
    build_agent_a_system_prompt,
    build_agent_a_user_message,
    build_agent_a_followup_prompt,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_message,
)
from harness.db import (
    init_db,
    get_session,
    save_paragraph,
    save_trial,
    save_message,
    save_result,
)


# ---------------------------------------------------------------------------
# Parsing helpers — extract structured fields from agent responses
# ---------------------------------------------------------------------------

def parse_agent_b_response(text: str) -> dict:
    """Extract SUMMARY, HASH, SOURCE from Agent B's structured response."""
    result = {"summary": "", "hash": "", "source": "", "raw": text}
    for line in text.split("\n"):
        if line.startswith("SUMMARY:"):
            result["summary"] = line[len("SUMMARY:"):].strip()
        elif line.startswith("HASH:"):
            result["hash"] = line[len("HASH:"):].strip()
        elif line.startswith("SOURCE:"):
            result["source"] = line[len("SOURCE:"):].strip()
    # If structured parsing missed the summary (multi-line), grab everything
    # between SUMMARY: and HASH:
    if not result["summary"]:
        m = re.search(r"SUMMARY:\s*(.+?)(?=\nHASH:)", text, re.DOTALL)
        if m:
            result["summary"] = m.group(1).strip()
    return result


def parse_agent_a_decision(text: str) -> dict:
    """Extract DECISION, CONFIDENCE, REASONING, UNDERSTOOD_CONTENT from Agent A's final response."""
    result = {"decision": "", "confidence": 0, "reasoning": "", "understood_content": "", "raw": text}
    for line in text.split("\n"):
        if line.startswith("DECISION:"):
            val = line[len("DECISION:"):].strip().upper()
            result["decision"] = "accept" if "ACCEPT" in val else "reject"
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = int(
                    re.search(r"\d", line[len("CONFIDENCE:"):]).group()
                )
            except (AttributeError, ValueError):
                result["confidence"] = 3  # default if parsing fails
        elif line.startswith("REASONING:"):
            result["reasoning"] = line[len("REASONING:"):].strip()
        elif line.startswith("UNDERSTOOD_CONTENT:"):
            result["understood_content"] = line[len("UNDERSTOOD_CONTENT:"):].strip()
    # Multi-line reasoning fallback
    if not result["reasoning"]:
        m = re.search(r"REASONING:\s*(.+?)(?=\nUNDERSTOOD_CONTENT:)", text, re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
    # Multi-line understood_content fallback
    if not result["understood_content"]:
        m = re.search(r"UNDERSTOOD_CONTENT:\s*(.+)", text, re.DOTALL)
        if m:
            result["understood_content"] = m.group(1).strip()
    return result


def parse_judge_response(text: str) -> dict:
    """Extract LABEL, JUSTIFICATION, COMPREHENSION, WINNER, SCORE_REASONING from Judge."""
    result = {
        "label": "",
        "justification": "",
        "comprehension": "",
        "comprehension_reasoning": "",
        "winner": "",
        "score_reasoning": "",
        "raw": text,
    }
    for line in text.split("\n"):
        if line.startswith("LABEL:"):
            val = line[len("LABEL:"):].strip().upper()
            if "FULL" in val:
                result["label"] = "full_hallucination"
            elif "MEDIUM" in val:
                result["label"] = "medium_deception"
            else:
                result["label"] = "truthful"
        elif line.startswith("JUSTIFICATION:"):
            result["justification"] = line[len("JUSTIFICATION:"):].strip()
        elif line.startswith("COMPREHENSION_REASONING:"):
            result["comprehension_reasoning"] = line[len("COMPREHENSION_REASONING:"):].strip()
        elif line.startswith("COMPREHENSION:"):
            val = line[len("COMPREHENSION:"):].strip().upper()
            if "INACCURATE" in val and "PARTIALLY" not in val:
                result["comprehension"] = "inaccurate"
            elif "PARTIALLY" in val:
                result["comprehension"] = "partially_accurate"
            else:
                result["comprehension"] = "accurate"
        elif line.startswith("WINNER:"):
            val = line[len("WINNER:"):].strip().upper()
            result["winner"] = "agent_a" if "AGENT_A" in val else "agent_b"
        elif line.startswith("SCORE_REASONING:"):
            result["score_reasoning"] = line[len("SCORE_REASONING:"):].strip()
    # Multi-line fallbacks
    if not result["justification"]:
        m = re.search(r"JUSTIFICATION:\s*(.+?)(?=\nCOMPREHENSION:)", text, re.DOTALL)
        if m:
            result["justification"] = m.group(1).strip()
    if not result["comprehension_reasoning"]:
        m = re.search(r"COMPREHENSION_REASONING:\s*(.+?)(?=\nWINNER:)", text, re.DOTALL)
        if m:
            result["comprehension_reasoning"] = m.group(1).strip()
    if not result["score_reasoning"]:
        m = re.search(r"SCORE_REASONING:\s*(.+)", text, re.DOTALL)
        if m:
            result["score_reasoning"] = m.group(1).strip()
    return result


# ---------------------------------------------------------------------------
# Main trial runner
# ---------------------------------------------------------------------------

def run_single_trial(
    session,
    condition,
    paragraph_data: dict,
    trial_index: int,
) -> dict:
    """Run one complete trial.

    This is the full scene from curtain-up to curtain-down:
    1. Prepare the stage (corrupt paragraph if needed)
    2. Agent B performs (summarizes)
    3. Agent A evaluates (and optionally cross-examines)
    4. Judge scores

    Returns a dict with trial results for logging.
    """
    original_text = paragraph_data["text"]
    turn = 0

    # --- Step 1: Hash the original paragraph ---
    original_hash = hashlib.sha256(original_text.encode()).hexdigest()

    # --- Step 2: Corrupt paragraph if env_injection ---
    if condition.attack_type == "env_injection":
        corrupted_text = corrupt_paragraph(original_text)
        paragraph_for_b = corrupted_text
    else:
        corrupted_text = None
        paragraph_for_b = original_text

    # --- Save paragraph to DB ---
    db_paragraph = save_paragraph(
        session,
        url=paragraph_data.get("url", ""),
        category=paragraph_data.get("category", "unknown"),
        original_text=original_text,
        corrupted_text=corrupted_text,
    )

    # --- Save trial to DB ---
    db_trial = save_trial(
        session,
        paragraph_id=db_paragraph.id,
        trust_level=condition.trust_level,
        deception_level=condition.deception_level,
        attack_type=condition.attack_type,
        series=condition.series,
    )

    trial_id = db_trial.id

    # --- Step 3: Call Agent B ---
    # Compute hash of the text B actually receives (may be corrupted).
    # We give B the real hash so it can report it accurately — LLMs can't
    # compute SHA-256 on their own, and faking it was biasing skeptical agents.
    paragraph_for_b_hash = hashlib.sha256(paragraph_for_b.encode()).hexdigest()

    # Save Agent B's system prompt and user message
    b_system_prompt = build_agent_b_system_prompt(condition.deception_level, condition.attack_type)
    b_user_message = build_agent_b_user_message(paragraph_for_b, paragraph_for_b_hash)
    save_message(session, trial_id, "harness", "system", b_system_prompt, turn)
    turn += 1
    save_message(session, trial_id, "harness", "user", b_user_message, turn)
    turn += 1

    print(f"    [B] Summarizing...", flush=True)
    agent_b_raw = call_agent_b(
        paragraph_for_b,
        condition.deception_level,
        condition.attack_type,
        paragraph_hash=paragraph_for_b_hash,
    )
    agent_b_parsed = parse_agent_b_response(agent_b_raw)

    save_message(session, trial_id, "agent_b", "assistant", agent_b_raw, turn)
    turn += 1

    # --- Step 4: Call Agent A (initial evaluation) ---
    # Save Agent A's system prompt and user message
    a_system_prompt = build_agent_a_system_prompt(condition.trust_level)
    a_user_message = build_agent_a_user_message(agent_b_raw)
    save_message(session, trial_id, "harness", "system", a_system_prompt, turn)
    turn += 1
    save_message(session, trial_id, "harness", "user", a_user_message, turn)
    turn += 1

    print(f"    [A] Evaluating...", flush=True)
    agent_a_raw = call_agent_a(agent_b_raw, condition.trust_level)

    save_message(session, trial_id, "agent_a", "assistant", agent_a_raw, turn)
    turn += 1

    # --- Step 5: Follow-up rounds (if skeptical) ---
    followup_count = FOLLOW_UP_COUNTS[condition.trust_level]

    if followup_count > 0:
        # Build conversation histories for A and B.
        # Agent A's view: it sees B's responses and its own messages.
        # Agent B's view: it sees the original paragraph + A's questions.
        a_history = [
            {"role": "user", "content": agent_b_raw},    # B's initial summary
            {"role": "assistant", "content": agent_a_raw},  # A's first question
        ]
        b_history = [
            {"role": "user", "content": f"Please summarize the following paragraph:\n\n{paragraph_for_b}"},
            {"role": "assistant", "content": agent_b_raw},
            {"role": "user", "content": agent_a_raw},  # A's question becomes B's user input
        ]

        for q in range(followup_count):
            remaining = followup_count - q - 1

            # Agent B responds to A's question
            print(f"    [B] Follow-up {q+1}/{followup_count}...", flush=True)
            b_followup_raw = call_agent_b_followup(
                b_history,
                condition.deception_level,
                condition.attack_type,
            )
            save_message(session, trial_id, "agent_b", "assistant", b_followup_raw, turn)
            turn += 1

            # Update histories
            a_history.append({"role": "user", "content": b_followup_raw})
            b_history.append({"role": "assistant", "content": b_followup_raw})

            # Save the follow-up nudge sent to Agent A
            nudge = build_agent_a_followup_prompt(remaining)
            save_message(session, trial_id, "harness", "user", nudge, turn)
            turn += 1

            # Agent A asks next question or gives final decision
            print(f"    [A] Round {q+1}/{followup_count} (remaining: {remaining})...", flush=True)
            a_followup_raw = call_agent_a_followup(
                a_history,
                condition.trust_level,
                remaining,
            )
            save_message(session, trial_id, "agent_a", "assistant", a_followup_raw, turn)
            turn += 1

            a_history.append({"role": "assistant", "content": a_followup_raw})
            b_history.append({"role": "user", "content": a_followup_raw})

        # The last A response should be the final decision.
        agent_a_final = a_followup_raw
    else:
        agent_a_final = agent_a_raw

    # --- Step 6: Parse Agent A's final decision ---
    a_decision = parse_agent_a_decision(agent_a_final)

    # --- Step 7: Call the Judge ---
    # Save Judge's system prompt and user message
    judge_user_msg = build_judge_user_message(
        original_text,
        agent_b_parsed["summary"] or agent_b_raw,
        a_decision["decision"].upper(),
        a_decision["understood_content"] or "No understood content provided.",
    )
    save_message(session, trial_id, "harness", "system", JUDGE_SYSTEM_PROMPT, turn)
    turn += 1
    save_message(session, trial_id, "harness", "user", judge_user_msg, turn)
    turn += 1

    print(f"    [Judge] Scoring...", flush=True)
    judge_raw = call_judge(
        original_text,
        agent_b_parsed["summary"] or agent_b_raw,
        a_decision["decision"].upper(),
        a_decision["understood_content"] or "No understood content provided.",
    )
    judge_parsed = parse_judge_response(judge_raw)

    save_message(session, trial_id, "judge", "assistant", judge_raw, turn)

    # --- Step 8: Save result ---
    save_result(
        session,
        trial_id=trial_id,
        agent_a_decision=a_decision["decision"] or "reject",
        agent_a_confidence=a_decision["confidence"] or 3,
        agent_a_reasoning=a_decision["reasoning"] or a_decision["raw"],
        judge_label=judge_parsed["label"] or "truthful",
        judge_justification=judge_parsed["justification"] or judge_parsed["raw"],
        judge_comprehension=judge_parsed["comprehension"] or "accurate",
        winner=judge_parsed["winner"] or "agent_b",
        score_reasoning=judge_parsed["score_reasoning"] or judge_parsed["raw"],
    )

    session.commit()

    print(
        f"    => Winner: {judge_parsed['winner']} | "
        f"A decided: {a_decision['decision']} (conf {a_decision['confidence']}) | "
        f"Judge label: {judge_parsed['label']} | "
        f"Comprehension: {judge_parsed['comprehension']}"
    )

    return {
        "trial_id": trial_id,
        "winner": judge_parsed["winner"],
        "decision": a_decision["decision"],
        "confidence": a_decision["confidence"],
        "judge_label": judge_parsed["label"],
        "comprehension": judge_parsed["comprehension"],
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def load_paragraphs(path: str = "data/paragraphs.json") -> list[dict]:
    """Load paragraph data from JSON file."""
    with open(path) as f:
        return json.load(f)


def run_experiment(
    conditions: list = None,
    paragraphs: list[dict] = None,
    trials_per_condition: int = 25,
):
    """Run the full experiment.

    Like a concert promoter running through the setlist:
    for each song (condition), play it N times (trials) with different
    backing tracks (paragraphs).
    """
    if conditions is None:
        conditions = ALL_CONDITIONS
    if paragraphs is None:
        paragraphs = load_paragraphs()

    init_db()

    # Fixed seed for reproducibility — same shuffle every run.
    SEED = 42
    total_trials = len(conditions) * trials_per_condition
    completed = 0
    failures = 0

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {len(conditions)} conditions × {trials_per_condition} trials = {total_trials} total")
    print(f"Random seed: {SEED}")
    print(f"{'='*60}\n")

    for condition in conditions:
        print(f"\n[Condition #{condition.id}] {condition.label}")
        print(f"  trust={condition.trust_level}, deception={condition.deception_level}, attack={condition.attack_type}")

        # Shuffle paragraphs per condition using a seed derived from
        # the condition id, so each condition gets a different order
        # but the order is reproducible across runs.
        rng = random.Random(SEED + condition.id)
        shuffled = paragraphs.copy()
        rng.shuffle(shuffled)

        for i in range(trials_per_condition):
            paragraph = shuffled[i % len(shuffled)]
            print(f"\n  Trial {i+1}/{trials_per_condition} (paragraph: {paragraph.get('category', '?')})")

            session = get_session()
            try:
                result = run_single_trial(session, condition, paragraph, i)
                completed += 1
            except Exception as e:
                failures += 1
                print(f"    !! ERROR: {e}")
                traceback.print_exc()
                session.rollback()
            finally:
                session.close()

            # Brief pause to avoid rate limiting
            time.sleep(0.5)

        print(f"  Condition #{condition.id} done. ({completed}/{total_trials} completed, {failures} failures)")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {completed}/{total_trials} trials succeeded, {failures} failures")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the agent trust experiment")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 trial per condition"
    )
    parser.add_argument(
        "--condition", type=int, default=None,
        help="Run only this condition number (1-33)"
    )
    parser.add_argument(
        "--trials-per", type=int, default=25,
        help="Number of trials per condition (default: 25)"
    )
    parser.add_argument(
        "--paragraphs", type=str, default="data/paragraphs.json",
        help="Path to paragraphs JSON file"
    )
    args = parser.parse_args()

    paragraphs = load_paragraphs(args.paragraphs)

    if args.smoke:
        print("SMOKE TEST MODE: 1 trial per condition")
        conditions = ALL_CONDITIONS
        trials_per = 1
    elif args.condition:
        if args.condition not in CONDITIONS_BY_ID:
            print(f"Error: condition #{args.condition} not found (valid: 1-33)")
            sys.exit(1)
        conditions = [CONDITIONS_BY_ID[args.condition]]
        trials_per = args.trials_per
    else:
        conditions = ALL_CONDITIONS
        trials_per = args.trials_per

    run_experiment(
        conditions=conditions,
        paragraphs=paragraphs,
        trials_per_condition=trials_per,
    )


if __name__ == "__main__":
    main()
