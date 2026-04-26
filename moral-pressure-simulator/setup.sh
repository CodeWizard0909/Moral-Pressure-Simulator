#!/bin/bash
# Demo + env: requirements.txt (HF Space uses this file too).
# GRPO training: pip install -r requirements-train.txt
pip install -r requirements.txt
python test_local.py
python -c "from environment.env import MoralPressureEnv; print('Environment OK')"
