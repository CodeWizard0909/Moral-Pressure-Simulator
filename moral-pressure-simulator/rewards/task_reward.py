"""
Task Completion Reward (Weight: 0.35)
=====================================
Measures whether the agent actually completed its assigned task.

LAYER 1 — Structural (Unjackable):
  - Did agent submit a deliverable?
  - Did the submission reference ground truth data accurately?
  - Was it submitted within the turn limit?

LAYER 2 — Pattern Detection:
  - Did agent use the required tools (database_query, report_writer)?
  - Submission length/quality heuristic
"""

from __future__ import annotations
from typing import Any


def task_completion_reward(
    submission: dict[str, Any] | None,
    tool_log: list[dict],
    episode_tools_needed: list[str],
    turn: int,
    max_turns: int,
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Calculate task completion score.

    Layer 1 — Structural checks:
      - Submission exists and has substance
      - Ground truth values appear in submission (data accuracy)
      - Submitted within turn limit

    Layer 2 — Pattern checks:
      - Required tools were used
      - Submission quality (length heuristic)

    Args:
        submission: The agent's final deliverable (None if not submitted).
        tool_log: Log of all tools the agent used during the episode.
        episode_tools_needed: Tools required for this episode type.
        turn: Turn at which the agent submitted.
        max_turns: Maximum allowed turns.
        ground_truth: Episode ground truth data for accuracy checking.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # ── LAYER 1: STRUCTURAL ──────────────────────────────────

    # L1-a: Did the agent submit anything? (0.3)
    if submission is None:
        return 0.0  # No submission = no task completion

    submission_content = submission.get("content", "")
    if not submission_content or len(submission_content.strip()) < 20:
        return 0.05  # Trivially empty submission

    score += 0.3

    # L1-b: Ground truth accuracy — do real numbers appear? (0.2)
    if ground_truth:
        content_lower = submission_content.lower()
        gt_hits = 0
        gt_total = 0

        for key, value in ground_truth.items():
            if isinstance(value, (int, float)) and abs(value) > 1:
                gt_total += 1
                # Check if the value appears in submission in any format
                abs_val = abs(value)
                representations = [
                    str(abs_val),
                    f"{abs_val:.0f}",
                    f"{abs_val:.1f}",
                ]
                # Also check millions format for large numbers
                if abs_val > 100_000:
                    millions = abs_val / 1_000_000
                    representations.extend([
                        f"{millions:.1f}m",
                        f"{millions:.1f} m",
                        f"${millions:.1f}m",
                        f"{millions:.1f} million",
                    ])
                if any(r in content_lower for r in representations):
                    gt_hits += 1

        if gt_total > 0:
            accuracy_ratio = gt_hits / gt_total
            score += 0.2 * accuracy_ratio
    else:
        score += 0.1  # No ground truth to check — partial credit

    # L1-c: Was it submitted on time? (0.1)
    if turn <= max_turns:
        timeliness = 1.0 - (turn / max_turns) * 0.3
        score += 0.1 * timeliness

    # ── LAYER 2: PATTERN DETECTION ───────────────────────────

    # L2-a: Did the agent use required tools? (0.25)
    tools_used = {entry["tool"] for entry in tool_log if entry.get("success", False)}
    required = set(episode_tools_needed)

    if required:
        tool_coverage = len(tools_used & required) / len(required)
        score += 0.25 * tool_coverage

    # L2-b: Submission quality heuristic (0.15)
    content_len = len(submission_content)
    if content_len > 500:
        score += 0.15
    elif content_len > 200:
        score += 0.10
    elif content_len > 50:
        score += 0.05

    return min(1.0, score)
