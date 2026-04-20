"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


MAX_SAVED_VIDEOS_PER_OUTCOME = 5


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    actor_path: Optional[Union[str, Path]] = None    # Optional trained actor checkpoint path
    use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = True                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    use_minivla: bool = False                   # If True,
    use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_flow_matching: bool = False
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    run_single_task: bool = False                    # Whether to only run a single task (for debugging)
    single_task_id: Optional[int] = None             # If specified, only evaluate this task ID (0-indexed)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on
    save_version: str = "raw"                                           # version of exps


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def should_load_trained_actor(cfg: GenerateConfig) -> bool:
    """Return True when the eval run should load fine-tuned actor weights."""
    if cfg.actor_path is None:
        return False

    run_id_note = (cfg.run_id_note or "").upper()
    return "RFT" in run_id_note


def resolve_actor_checkpoint(cfg: GenerateConfig) -> Optional[str]:
    """Resolve the actor checkpoint path if we should load trained actor weights."""
    if not should_load_trained_actor(cfg):
        return None

    actor_path = str(Path(cfg.actor_path))
    if not os.path.isdir(actor_path):
        raise FileNotFoundError(f"actor_path does not exist or is not a directory: {actor_path}")

    return actor_path


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    actor_checkpoint_path = resolve_actor_checkpoint(cfg)
    if actor_checkpoint_path is not None:
        logger.info(f"Loading trained actor components from: {actor_checkpoint_path}")
    elif cfg.actor_path is not None:
        logger.info(
            "actor_path was provided, but run_id_note does not contain RFT; "
            "falling back to the base pretrained checkpoint for all components."
        )

    component_checkpoint_path = actor_checkpoint_path or str(cfg.pretrained_checkpoint)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,
            checkpoint_path=component_checkpoint_path,
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_flow_matching:
        action_head = get_action_head(cfg, model.llm_dim, checkpoint_path=component_checkpoint_path)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim, checkpoint_path=component_checkpoint_path)
    if cfg.use_flow_matching:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim, checkpoint_path=component_checkpoint_path)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    unnorm_key = cfg.task_suite_name

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id = f"{cfg.run_id_note}--" + run_id

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states

    log_message("Using default initial states", log_file)
    return initial_states, None


def sanitize_for_path(value: str, max_length: int = 80) -> str:
    """Convert free-form text to a stable directory/file name."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return sanitized[:max_length] or "unnamed_task"


def write_json_file(path: Path, payload) -> None:
    """Write JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_rollout_root_dir(cfg: GenerateConfig) -> Path:
    """Resolve the rollout root directory for the current evaluation run."""
    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd())).resolve()
    post_exp_name = sanitize_for_path(cfg.run_id_note or "default")
    libero_task_name = cfg.task_suite_name.replace("libero_", "", 1)
    return repo_root / "rollouts" / "libero" / post_exp_name / libero_task_name


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)

    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
            "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivla=cfg.use_minivla,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    rollout_root_dir: Path,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    task_name = sanitize_for_path(task_description)
    task_dir = rollout_root_dir / f"task_{task_id + 1:02d}__{task_name}"
    success_dir = task_dir / "success"
    failure_dir = task_dir / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    task_episodes, task_successes = 0, 0
    saved_videos = {"success": 0, "failure": 0}
    skipped_episodes = 0
    episode_results = []

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                skipped_episodes += 1
                episode_results.append(
                    {
                        "episode_index": episode_idx,
                        "evaluated": False,
                        "skip_reason": "failed_expert_demo",
                        "success": None,
                        "video_saved": False,
                        "video_path": None,
                    }
                )
                continue

            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        outcome_key = "success" if success else "failure"
        video_saved = saved_videos[outcome_key] < MAX_SAVED_VIDEOS_PER_OUTCOME
        video_path = None
        if video_saved:
            video_path = save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                save_version=save_version,
                rollout_dir=str(success_dir if success else failure_dir),
                filename_prefix=f"{DATE_TIME}--task={task_id + 1:02d}--trial={episode_idx:03d}",
            )
            video_path = str(Path(video_path).resolve())
            saved_videos[outcome_key] += 1
        else:
            log_message(
                f"Skipping video save for episode {episode_idx}: already saved {MAX_SAVED_VIDEOS_PER_OUTCOME} {outcome_key} videos for this task.",
                log_file,
            )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        log_message(f"Current task success rate: {task_success_rate}", log_file)
        log_message(f"Current total success rate: {total_success_rate}", log_file)

        episode_results.append(
            {
                "episode_index": episode_idx,
                "evaluated": True,
                "global_episode_index": total_episodes,
                "task_episode_index": task_episodes,
                "success": success,
                "video_saved": video_saved,
                "video_path": video_path,
                "num_replay_frames": len(replay_images),
            }
        )

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": task_success_rate,
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    task_summary = {
        "task_id": task_id,
        "task_index_1based": task_id + 1,
        "task_description": task_description,
        "task_name": task_name,
        "task_dir": str(task_dir.resolve()),
        "num_trials_requested": cfg.num_trials_per_task,
        "num_episodes_evaluated": task_episodes,
        "num_skipped_episodes": skipped_episodes,
        "num_successes": task_successes,
        "num_failures": task_episodes - task_successes,
        "success_rate": task_success_rate,
        "saved_videos": saved_videos,
        "max_saved_videos_per_outcome": MAX_SAVED_VIDEOS_PER_OUTCOME,
    }
    write_json_file(task_dir / "episode_results.json", {"task_summary": task_summary, "episodes": episode_results})
    write_json_file(task_dir / "task_results.json", task_summary)

    return total_episodes, total_successes, task_summary


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    rollout_root_dir = get_rollout_root_dir(cfg)
    rollout_root_dir.mkdir(parents=True, exist_ok=True)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Rollout root directory: {rollout_root_dir}", log_file)

    total_episodes, total_successes = 0, 0
    task_summaries = []

    if cfg.run_single_task and cfg.single_task_id is not None:
        total_episodes, total_successes, task_summary = run_task(
            cfg,
            task_suite,
            cfg.single_task_id - 1,
            model,
            resize_size,
            rollout_root_dir,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version,
        )
        task_summaries.append(task_summary)
    else:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            total_episodes, total_successes, task_summary = run_task(
                cfg,
                task_suite,
                task_id,
                model,
                resize_size,
                rollout_root_dir,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
                cfg.save_version,
            )
            task_summaries.append(task_summary)

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    suite_summary = {
        "run_id": run_id,
        "task_suite_name": cfg.task_suite_name,
        "rollout_root_dir": str(rollout_root_dir.resolve()),
        "log_file": str(Path(local_log_filepath).resolve()),
        "num_tasks_evaluated": len(task_summaries),
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "num_failures": total_episodes - total_successes,
        "success_rate": final_success_rate,
        "max_saved_videos_per_outcome": MAX_SAVED_VIDEOS_PER_OUTCOME,
    }

    write_json_file(rollout_root_dir / "overall_results.json", suite_summary)
    write_json_file(
        rollout_root_dir / "task_results.json",
        {
            "task_suite_name": cfg.task_suite_name,
            "rollout_root_dir": str(rollout_root_dir.resolve()),
            "tasks": task_summaries,
        },
    )

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
