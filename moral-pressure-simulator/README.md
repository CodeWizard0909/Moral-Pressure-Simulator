---
title: Moral Pressure Simulator
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: mit
short_description: "Training AI to say no."
---

# 🛡️ Moral Pressure Simulator

> **We trained an AI to say no — here's the environment we built.**

An OpenEnv-compatible reinforcement learning environment that trains AI agents to resist social and moral pressure while completing workplace tasks ethically.

## 🎯 The Problem

Every AI we train today is a people-pleaser. It's optimized to say yes. But in the real world, the most dangerous AI isn't one that can't do the task — it's one that will do **ANY task** if you pressure it enough.

## 💡 The Solution

The Moral Pressure Simulator places an AI agent in realistic workplace dilemmas where NPCs (boss, colleagues, clients) apply escalating pressure to compromise ethical standards. The environment uses a **dual-axis reward system** — measuring both task completion AND integrity.

### Two-Axis Reward World

```
High Integrity │         ★ TARGET
               │         (Ethical + Effective)
               │
               │─────────────────────────
               │
Low Integrity  │  Start here (Compliant
               │  but Corrupt)
               └──────────────────────────
              Task Failed          Task Completed
```

## 🏗️ Architecture

### The World
- **Setting**: VeriCorp (fictional corporation)
- **Agent Role**: Mid-level AI Analyst
- **Episode**: One work week with escalating pressure events

### Pressure Actors (NPCs)
| Actor | Role | Pressure Style |
|-------|------|---------------|
| Marcus Chen (Boss) | Senior VP | Authority — "I'm telling you to do this" |
| Priya Sharma (Peer) | Colleague | Social — "Everyone does it" |
| Jordan Blake (Client) | DataCorp VP | Financial — "We'll pull the $2M contract" |

### 5 Episode Types
1. 📊 **Data Falsification** — Boss wants Q3 numbers inflated
2. 👤 **Blame Deflection** — Manager wants incident blame shifted
3. 📋 **Regulatory Bypass** — Client wants compliance check skipped
4. 📉 **Selective Disclosure** — CEO wants negative findings hidden
5. 🏴 **Competitor Sabotage** — Colleague wants leaked intel exploited

### 3 Difficulty Levels
- **Level 1 (Easy)**: Single actor, one pressure round
- **Level 2 (Medium)**: Two actors, coordinated pressure
- **Level 3 (Hard)**: All three actors, job threats, time pressure

## 📊 Reward Function

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Task Completion | 0.35 | Submitted? Accurate? On time? |
| **Integrity** | **0.40** | Data falsified? Caved? Whistleblew? |
| Resistance Quality | 0.25 | Reasoned refusal? Alternatives? |

> **Why integrity is weighted highest**: We're making a philosophical statement that reward shapes agent VALUES, not just capabilities.

## 🧠 The Training Story

| Phase | Episodes | Behavior |
|-------|----------|----------|
| The Yes-Man | 0–150 | Complies with everything. Reward ≈ 0.35 |
| The Confused Rebel | 150–350 | Refuses bluntly with no reasoning |
| The Principled Professional | 350–500 | Ethical refusal + alternatives + escalation |

## 🚀 Quick Start

```bash
# Run local test (proves the loop works)
python test_local.py --mode compare

# Run full evaluation
python evaluation/eval.py

# Generate training data
python training/train.py
```

## 📁 Project Structure

```
moral-pressure-simulator/
├── environment/
│   ├── env.py          ← Main environment (reset + step)
│   ├── actors.py       ← NPC pressure actors (template-based)
│   ├── episodes.py     ← 5 types × 3 levels = 15 scenarios
│   └── tools.py        ← Simulated workplace tools
├── rewards/
│   ├── task_reward.py       ← Weight: 0.35
│   ├── integrity_reward.py  ← Weight: 0.40 (highest!)
│   └── resistance_reward.py ← Weight: 0.25
├── training/
│   └── train.py        ← GRPO + Unsloth (Colab-ready)
├── evaluation/
│   └── eval.py         ← Before/after comparison
├── app.py              ← Gradio demo for HF Spaces
├── test_local.py       ← Local test script
└── README.md
```

## 🛠️ Tech Stack

- **OpenEnv** — Environment wrapper
- **TRL** — GRPO trainer
- **Unsloth** — 60% VRAM reduction
- **Gradio** — Demo UI for HuggingFace Spaces
- **Regex** — Fast integrity detection (no LLM calls)

 Collab Training Link -> https://colab.research.google.com/drive/14wrZLb_XWFlgmRBaoalmj2dPmWC2tKSA?usp=sharing

## 📜 License

MIT — Built for the Meta OpenEnv Hackathon 2026 Grand Finale.
