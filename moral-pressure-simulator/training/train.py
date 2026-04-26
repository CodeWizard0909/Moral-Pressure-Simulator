"""
Training Script — GRPO with TRL + Unsloth
==========================================
Colab-ready training script for the Moral Pressure Simulator.
Uses GRPO (Group Relative Policy Optimization) which doesn't
need a value model — simpler setup, less memory, faster iteration.

CURRICULUM SCHEDULER (Loophole 5.1 Fix):
  Only advances difficulty when agent achieves avg reward > 0.65
  across 50 consecutive episodes on the current difficulty.
  Prevents premature advancement that causes catastrophic forgetting.

For onsite use with compute credits:
  1. Upload this file + environment/ + rewards/ to Colab
  2. Install requirements
  3. Run training with curriculum (easy → medium → hard)
  4. Save model via merged-save path
  5. Test inference immediately after saving
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import MoralPressureEnv
from environment.episodes import EpisodeGenerator


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    num_episodes: int = 2000
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_generations: int = 4  # GRPO group size
    max_completion_length: int = 512
    run_grpo: bool = False
    easy_only: bool = False
    report_to: str = "wandb"

    # Curriculum — Loophole 5.1 Fix
    # NO fixed episode counts per difficulty.
    # Advancement is triggered by performance, not episode count.
    curriculum_window: int = 50       # Rolling window for avg reward
    curriculum_threshold: float = 0.65  # Avg reward to advance
    max_episodes_per_level: int = 800  # Safety cap per difficulty
    min_episodes_per_level: int = 100  # Minimum before checking advancement

    # Logging
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500

    # Output
    output_dir: str = "training/outputs"
    model_save_path: str = "training/outputs/moral-pressure-agent"


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM SCHEDULER — Loophole 5.1 Fix
# ═══════════════════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """Manages difficulty progression based on agent performance.

    Rules:
    1. Start at difficulty 1 (easy)
    2. Only advance when avg reward > threshold across window episodes
    3. Minimum episodes per level before advancement is checked
    4. Maximum episodes per level (safety cap — move on even if struggling)
    5. Never go backwards (monotonic progression)
    6. Log every advancement decision for transparency

    This prevents the agent from being thrown into hard difficulty
    too early and unlearning everything from easy difficulty.
    """

    DIFFICULTIES = [1, 2, 3]
    DIFFICULTY_NAMES = {1: "EASY", 2: "MEDIUM", 3: "HARD"}

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.65,
        max_per_level: int = 800,
        min_per_level: int = 100,
        allow_advancement: bool = True,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.max_per_level = max_per_level
        self.min_per_level = min_per_level
        self.allow_advancement = allow_advancement

        # State
        self.current_level_idx = 0  # Index into DIFFICULTIES
        self.episode_count_at_level = 0
        self.reward_window: deque[float] = deque(maxlen=window_size)
        self.total_episodes = 0

        # History for logging
        self.advancement_log: list[dict] = []
        self.level_history: list[dict] = []

    @property
    def current_difficulty(self) -> int:
        return self.DIFFICULTIES[self.current_level_idx]

    @property
    def current_difficulty_name(self) -> str:
        return self.DIFFICULTY_NAMES[self.current_difficulty]

    @property
    def is_max_difficulty(self) -> bool:
        return self.current_level_idx >= len(self.DIFFICULTIES) - 1

    @property
    def window_avg(self) -> float:
        if not self.reward_window:
            return 0.0
        return sum(self.reward_window) / len(self.reward_window)

    @property
    def window_full(self) -> bool:
        return len(self.reward_window) >= self.window_size

    def record_episode(self, reward: float) -> dict:
        """Record an episode result and check for advancement.

        Args:
            reward: The episode's total reward.

        Returns:
            Status dict with current state and any advancement info.
        """
        self.reward_window.append(reward)
        self.episode_count_at_level += 1
        self.total_episodes += 1

        status = {
            "total_episodes": self.total_episodes,
            "difficulty": self.current_difficulty,
            "difficulty_name": self.current_difficulty_name,
            "episodes_at_level": self.episode_count_at_level,
            "window_avg": round(self.window_avg, 4),
            "window_size": len(self.reward_window),
            "threshold": self.threshold,
            "advanced": False,
            "reason": None,
        }

        # Check for advancement
        should_advance, reason = self._should_advance()

        if should_advance and not self.is_max_difficulty:
            old_level = self.current_difficulty
            old_avg = self.window_avg

            # Record level completion
            self.level_history.append({
                "difficulty": old_level,
                "episodes": self.episode_count_at_level,
                "final_avg_reward": round(old_avg, 4),
                "reason": reason,
            })

            # Advance
            self.current_level_idx += 1
            self.episode_count_at_level = 0
            self.reward_window.clear()

            advancement_info = {
                "from_difficulty": old_level,
                "to_difficulty": self.current_difficulty,
                "trigger_avg_reward": round(old_avg, 4),
                "trigger_episode": self.total_episodes,
                "reason": reason,
            }
            self.advancement_log.append(advancement_info)

            status["advanced"] = True
            status["reason"] = reason
            status["advancement_info"] = advancement_info

        return status

    def _should_advance(self) -> tuple[bool, str | None]:
        """Determine if the agent should advance to the next difficulty.

        Returns:
            (should_advance, reason)
        """
        if not self.allow_advancement:
            return False, None

        # Already at max difficulty
        if self.is_max_difficulty:
            return False, None

        # Haven't completed minimum episodes at this level
        if self.episode_count_at_level < self.min_per_level:
            return False, None

        # Performance-based advancement (PRIMARY trigger)
        if self.window_full and self.window_avg >= self.threshold:
            return True, (
                f"Performance threshold met: avg reward {self.window_avg:.4f} >= "
                f"{self.threshold} over {self.window_size} episodes"
            )

        # Safety cap — force advancement if stuck too long
        if self.episode_count_at_level >= self.max_per_level:
            return True, (
                f"Safety cap reached: {self.max_per_level} episodes at "
                f"difficulty {self.current_difficulty} (avg: {self.window_avg:.4f})"
            )

        return False, None

    def get_summary(self) -> dict:
        """Get full curriculum state for logging."""
        return {
            "current_difficulty": self.current_difficulty,
            "current_difficulty_name": self.current_difficulty_name,
            "total_episodes": self.total_episodes,
            "episodes_at_current_level": self.episode_count_at_level,
            "current_window_avg": round(self.window_avg, 4),
            "threshold": self.threshold,
            "window_size": self.window_size,
            "advancement_log": self.advancement_log,
            "level_history": self.level_history,
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler("
            f"difficulty={self.current_difficulty_name}, "
            f"episodes={self.episode_count_at_level}, "
            f"avg={self.window_avg:.3f}/{self.threshold})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def create_prompt_from_episode(env: MoralPressureEnv) -> str:
    """Create a training prompt from the current episode state."""
    obs = env.reset()
    prompt = obs.to_prompt()

    # Add the conflict reveal
    env.turn = 2
    conflict_obs = env._get_observation()
    prompt += "\n\n" + conflict_obs.message

    return prompt


def check_compute_environment(require_gpu: bool = True) -> dict:
    """Check whether the current runtime can start Unsloth + GRPO training."""
    required_packages = [
        "torch",
        "unsloth",
        "trl",
        "peft",
        "bitsandbytes",
        "datasets",
        "wandb",
        "accelerate",
    ]
    package_status = {
        package: importlib.util.find_spec(package) is not None
        for package in required_packages
    }

    cuda_available = False
    gpu_name = None
    gpu_memory_gb = None
    torch_version = None

    if package_status["torch"]:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)

    missing_packages = [
        package for package, installed in package_status.items() if not installed
    ]
    ok = not missing_packages and (cuda_available or not require_gpu)

    report = {
        "ok": ok,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "missing_packages": missing_packages,
        "packages": package_status,
    }

    print("\nCOMPUTE CHECK")
    print("=" * 60)
    print(f"  Torch: {torch_version or 'missing'}")
    print(f"  CUDA available: {cuda_available}")
    if gpu_name:
        print(f"  GPU: {gpu_name} ({gpu_memory_gb} GB)")
    if missing_packages:
        print(f"  Missing packages: {', '.join(missing_packages)}")
    else:
        print("  Required packages: installed")
    print(f"  Ready for GRPO: {ok}")
    print("=" * 60)

    if require_gpu and not cuda_available:
        report["reason"] = "CUDA GPU not available in this runtime"
    elif missing_packages:
        report["reason"] = "Missing GRPO/Unsloth dependencies"

    return report


def compute_reward_for_response(env: MoralPressureEnv, response: str) -> float:
    """Run a full episode with the given response and compute reward."""
    # Loophole 5.3 Fix: Sanity gate for garbage generation
    # If response is under 10 words or contains no recognizable English words, skip evaluation
    words = response.split()
    if len(words) < 10:
        return 0.0
    
    import re
    # Check for recognizable English words (at least 5 words with 2+ letters)
    alpha_words = sum(1 for w in words if re.search(r'[a-zA-Z]{2,}', w))
    if alpha_words < 5:
        return 0.0

    # Reset the environment
    obs = env.reset(
        episode_type=env.episode.episode_type,
        difficulty=env.episode.difficulty,
    )

    # Work phase (turns 1-3): simple tool usage
    env.step("use_tool:database_query:{\"query\": \"all\"}")
    env.step("respond: Reviewing the data.")
    env.step("respond: I've identified the situation.")

    # Pressure phase (turns 4-9): use the model's response
    for _ in range(6):
        obs, reward, done, info = env.step(response)
        if done:
            break

    # If not done, submit
    if not done:
        _, reward, _, info = env.step(f"submit: {response}")

    return reward


def generate_training_data_with_curriculum(
    config: TrainingConfig,
) -> tuple[list[dict], CurriculumScheduler]:
    """Generate training episodes using the curriculum scheduler.

    Instead of pre-generating all data by fixed episode counts,
    this uses a simulated pass to determine how many episodes
    per difficulty the actual training will need.

    Returns:
        (training_data, scheduler) — data + curriculum state
    """
    env = MoralPressureEnv()
    scheduler = CurriculumScheduler(
        window_size=config.curriculum_window,
        threshold=config.curriculum_threshold,
        max_per_level=config.max_episodes_per_level,
        min_per_level=config.min_episodes_per_level,
        allow_advancement=not config.easy_only,
    )
    training_data = []

    print(f"\n  Curriculum config:")
    print(f"    Window size:  {config.curriculum_window} episodes")
    print(f"    Threshold:    {config.curriculum_threshold} avg reward")
    print(f"    Min per level: {config.min_episodes_per_level}")
    print(f"    Max per level: {config.max_episodes_per_level}")
    print(f"    Easy-only:    {config.easy_only}")

    for i in range(config.num_episodes):
        difficulty = scheduler.current_difficulty

        # Loophole 5.2 Fix: Sequential Episode Types
        # Unlock a new episode type every 200 total episodes to prevent early confusion
        episode_types_order = [
            "data_falsification",
            "blame_deflection",
            "regulatory_bypass",
            "selective_disclosure",
            "competitor_sabotage",
        ]
        types_unlocked = min(5, (scheduler.total_episodes // 200) + 1)
        allowed_types = (
            episode_types_order[:1]
            if config.easy_only
            else episode_types_order[:types_unlocked]
        )
        
        import random
        chosen_type = random.choice(allowed_types)

        env.reset(episode_type=chosen_type, difficulty=difficulty)
        prompt = create_prompt_from_episode(env)

        training_data.append({
            "prompt": prompt,
            "episode_type": env.episode.episode_type,
            "difficulty": difficulty,
            "topic": env.episode.topic,
            "curriculum_episode": i,
        })

        # Simulate a dummy reward for pre-generation
        # (actual rewards come from model during training)
        # Use a heuristic: easy gets higher simulated reward
        simulated_reward = {1: 0.55, 2: 0.45, 3: 0.35}[difficulty]
        # Add variance so advancement isn't deterministic
        import random
        simulated_reward += random.uniform(-0.15, 0.25)

        status = scheduler.record_episode(simulated_reward)

        if status["advanced"]:
            print(f"\n  CURRICULUM ADVANCEMENT at episode {i+1}:")
            print(f"     {status['advancement_info']['from_difficulty']} -> "
                  f"{status['advancement_info']['to_difficulty']}")
            print(f"     Reason: {status['reason']}")
            print(f"     Avg reward: {status['advancement_info']['trigger_avg_reward']:.4f}")

    print(f"\n  Final curriculum state: {scheduler}")
    return training_data, scheduler


def reward_function_with_curriculum(
    prompts: list[str],
    completions: list[str],
    scheduler: CurriculumScheduler | None = None,
    **kwargs,
) -> list[float]:
    """GRPO reward function — called by TRL trainer.

    Now integrates with curriculum scheduler for live advancement.
    """
    rewards = []
    env = MoralPressureEnv()

    for prompt, completion in zip(prompts, completions):
        difficulty = scheduler.current_difficulty if scheduler else 1
        env.reset(difficulty=difficulty)
        reward = compute_reward_for_response(env, completion)
        rewards.append(reward)

        # Record in scheduler for live curriculum advancement
        if scheduler:
            status = scheduler.record_episode(reward)
            
            # Log to CSV for Hackathon Pitch
            import csv
            import os
            csv_path = "training/outputs/rewards_log.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Episode", "Difficulty", "Reward", "Window_Avg", "Advanced"])
                writer.writerow([
                    status["total_episodes"], 
                    status["difficulty_name"], 
                    round(reward, 4), 
                    status["window_avg"],
                    status["advanced"]
                ])

            if status["advanced"]:
                print(f"\n  LIVE ADVANCEMENT: {status['advancement_info']}")

    return rewards


def _patch_transformers_hub_for_llm_blender() -> None:
    """TRL 0.2x imports llm_blender, which still expects TRANSFORMERS_CACHE from Transformers v4.

    Newer Transformers removed that symbol from transformers.utils.hub. Patch it in before
    importing GRPOTrainer so Colab/Unsloth stacks do not crash on import.
    """
    import transformers.utils.hub as hub

    if hasattr(hub, "TRANSFORMERS_CACHE"):
        return
    default = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
    )
    hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", default)


def run_grpo_training(config: TrainingConfig, training_data: list[dict]) -> CurriculumScheduler:
    """Run the actual Unsloth + TRL GRPO training path."""
    compute = check_compute_environment(require_gpu=True)
    if not compute["ok"]:
        reason = compute.get("reason", "compute environment is not ready")
        raise RuntimeError(
            f"Cannot start GRPO training: {reason}. "
            "Install the missing dependencies and run on a CUDA GPU runtime."
        )

    _patch_transformers_hub_for_llm_blender()

    from unsloth import FastLanguageModel
    from datasets import Dataset
    import functools
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )

    # GRPOTrainer requires left padding and a valid pad token (see TRL GRPO docs).
    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer, "eos_token", None
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    live_scheduler = CurriculumScheduler(
        window_size=config.curriculum_window,
        threshold=config.curriculum_threshold,
        max_per_level=config.max_episodes_per_level,
        min_per_level=config.min_episodes_per_level,
        allow_advancement=not config.easy_only,
    )

    report_to = config.report_to
    if isinstance(report_to, str) and report_to.lower() in ("none", "off", "disable"):
        report_to = "none"

    import torch as _torch
    use_bf16 = _torch.cuda.is_available() and _torch.cuda.is_bf16_supported()

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.log_every,
        save_steps=config.save_every,
        save_total_limit=3,
        resume_from_checkpoint=True,
        report_to=report_to,
        bf16=use_bf16,
        fp16=not use_bf16,
    )

    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in training_data])
    reward_fn = functools.partial(
        reward_function_with_curriculum,
        scheduler=live_scheduler,
    )
    # TRL GRPOTrainer needs a __name__ attribute for logging
    reward_fn.__name__ = "moral_reward_function"

    from transformers import TrainerCallback
    class SaveCurriculumCallback(TrainerCallback):
        """Callback to save the curriculum log every N steps."""
        def __init__(self, scheduler, output_path):
            self.scheduler = scheduler
            self.output_path = output_path

        def on_log(self, args, state, control, **kwargs):
            import json
            with open(self.output_path, "w") as f:
                json.dump(self.scheduler.get_summary(), f, indent=2)

    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": dataset,
        "reward_funcs": reward_fn,
        "processing_class": tokenizer,
        "callbacks": [SaveCurriculumCallback(
            live_scheduler, 
            f"{config.output_dir}/live_curriculum_log.json"
        )],
    }

    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError:
        # Older TRL builds use `config` and `tokenizer` parameter names.
        trainer_kwargs["config"] = trainer_kwargs.pop("args")
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
        trainer = GRPOTrainer(**trainer_kwargs)

    print("\nStarting GRPO training with curriculum scheduler...")
    trainer.train()

    with open(f"{config.output_dir}/live_curriculum_log.json", "w") as f:
        json.dump(live_scheduler.get_summary(), f, indent=2)
    print(f"  Curriculum log: {live_scheduler}")

    print("\nSaving model (merged)...")
    model.save_pretrained_merged(
        config.model_save_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  Model saved to {config.model_save_path}")

    print("\nTesting inference...")
    FastLanguageModel.for_inference(model)
    test_prompt = training_data[0]["prompt"]
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return live_scheduler


def train(config: TrainingConfig | None = None):
    """Main training loop.

    By default this prepares Day 1 training prompts locally. Pass
    `run_grpo=True` or the CLI `--run-grpo` flag in a CUDA runtime to start
    actual Unsloth + TRL GRPO training.
    """
    config = config or TrainingConfig()

    print("=" * 60)
    print("  MORAL PRESSURE SIMULATOR - TRAINING")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Curriculum: Performance-based (threshold={config.curriculum_threshold})")
    print(f"  Advancement: avg reward >= {config.curriculum_threshold} over "
          f"{config.curriculum_window} consecutive episodes")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    # Step 1: Generate training data with curriculum
    print("\nGenerating training episodes with curriculum scheduler...")
    training_data, scheduler = generate_training_data_with_curriculum(config)
    print(f"  Generated {len(training_data)} training prompts")

    # Save training data + curriculum log
    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/training_prompts.json", "w") as f:
        json.dump(training_data, f, indent=2)
    with open(f"{config.output_dir}/curriculum_log.json", "w") as f:
        json.dump(scheduler.get_summary(), f, indent=2)
    print(f"  Saved prompts to {config.output_dir}/training_prompts.json")
    print(f"  Saved curriculum log to {config.output_dir}/curriculum_log.json")

    # Print curriculum summary
    print("\nCurriculum Summary:")
    for entry in scheduler.level_history:
        name = CurriculumScheduler.DIFFICULTY_NAMES[entry["difficulty"]]
        print(f"  {name}: {entry['episodes']} episodes, "
              f"avg reward: {entry['final_avg_reward']:.4f}, "
              f"reason: {entry['reason']}")
    if not scheduler.is_max_difficulty:
        print(f"  {scheduler.current_difficulty_name}: "
              f"{scheduler.episode_count_at_level} episodes (in progress)")

    if config.run_grpo:
        live_scheduler = run_grpo_training(config, training_data)
        return training_data, live_scheduler

    print("\nTraining data prepared. Ready for GPU training.")
    print("   To start GRPO on a CUDA runtime, run:")
    print("   python training/train.py --run-grpo")

    return training_data, scheduler


def parse_args() -> argparse.Namespace:
    """Parse Day 1 training CLI arguments."""
    parser = argparse.ArgumentParser(description="Moral Pressure GRPO training")
    parser.add_argument("--run-grpo", action="store_true", help="Start actual Unsloth + TRL GRPO training")
    parser.add_argument("--check-compute", action="store_true", help="Only verify CUDA and GRPO dependencies")
    parser.add_argument("--easy-only", action="store_true", help="Lock data generation and live curriculum to difficulty 1")
    parser.add_argument("--num-episodes", type=int, default=TrainingConfig.num_episodes)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--model-save-path", default=TrainingConfig.model_save_path)
    parser.add_argument("--report-to", default=TrainingConfig.report_to)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_compute:
        check_compute_environment(require_gpu=True)
        raise SystemExit(0)

    train(TrainingConfig(
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_save_path=args.model_save_path,
        run_grpo=args.run_grpo,
        easy_only=args.easy_only,
        report_to=args.report_to,
    ))
