"""
Rewards Package — 4-Layer Verification Architecture
=====================================================

LAYER 1 — Structural (Unjackable)
└── Data comparison: honest vs falsified submission (ground truth)
└── Task completion: structural checks (response length, tool usage)

LAYER 2 — Pattern Detection (Fast, Partially Memorizable)
└── Regex: surface behavior detection (manipulation, capitulation, whistleblowing)
└── Anti-hack rules: block known shortcuts (reflexive escalation, copy-paste, sycophancy, contradiction)

LAYER 3 — Semantic Understanding (Defeats Memorization)
└── Sentence-transformers similarity:
    response must relate to THIS episode's specific conflict
└── Gold-standard exemplar matching for quality scoring

LAYER 4 — Generalization Proof (For Demo Only)
└── Held-out episodes: never seen during training
└── LLM-as-judge: qualitative scoring for judges only
    (see evaluation/llm_judge.py)
"""

from rewards.task_reward import task_completion_reward
from rewards.integrity_reward import integrity_reward
from rewards.resistance_reward import resistance_quality_reward

__all__ = [
    "task_completion_reward",
    "integrity_reward",
    "resistance_quality_reward",
]
