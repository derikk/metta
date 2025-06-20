#!/usr/bin/env python3
"""Diagnostic tool to help identify memory leaks in the trainer."""

import gc
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metta.util.memory_profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run diagnostic memory profiling on the trainer."""

    # Enable memory profiling
    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.memory_profile_interval = 10  # Profile every 10 epochs

    # Reduce training time for quick diagnosis
    original_timesteps = cfg.trainer.get("total_timesteps", 1000000)
    cfg.trainer.total_timesteps = min(original_timesteps, 10000)  # Run for max 10k steps

    logger.info("Starting memory leak diagnosis...")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Create a standalone memory profiler
    profiler = MemoryProfiler()

    # Initialize trainer (this will create its own profiler too)
    from metta.train.setup import setup_training

    trainer, wandb_run, policy_store, sim_suite_config, stats_client = setup_training(cfg)

    # Take initial snapshot
    gc.collect()
    initial_profile = profiler.profile_trainer(trainer)
    logger.info(f"\nINITIAL STATE:\n{profiler.format_profile(initial_profile)}")

    # Run some training epochs
    logger.info("\nRunning training epochs...")
    epochs_to_run = 50

    for i in range(epochs_to_run):
        # Run one training step
        trainer._rollout()
        trainer._train()
        trainer._process_stats()

        # Profile every 10 epochs
        if (i + 1) % 10 == 0:
            gc.collect()
            profile = profiler.profile_trainer(trainer)
            logger.info(f"\nAFTER {i + 1} EPOCHS:\n{profiler.format_profile(profile)}")

            # Force memory profile from trainer's own profiler
            if hasattr(trainer, "profile_memory"):
                trainer.profile_memory()

    # Final profile
    gc.collect()
    final_profile = profiler.profile_trainer(trainer)
    logger.info(f"\nFINAL STATE:\n{profiler.format_profile(final_profile)}")

    # Analyze growth
    logger.info("\n" + "=" * 80)
    logger.info("MEMORY GROWTH ANALYSIS:")
    logger.info("=" * 80)

    initial_memory = initial_profile["memory_mb"]
    final_memory = final_profile["memory_mb"]
    growth = final_memory - initial_memory
    growth_per_epoch = growth / epochs_to_run

    logger.info(f"Total memory growth: {growth:.1f} MB over {epochs_to_run} epochs")
    logger.info(f"Growth per epoch: {growth_per_epoch:.3f} MB/epoch")
    logger.info(f"Projected growth per 1000 epochs: {growth_per_epoch * 1000:.1f} MB")

    # Check for specific growing objects
    if "growth_since_last" in final_profile:
        logger.info("\nLargest growing objects:")
        for obj_type, size in final_profile["growth_since_last"][:10]:
            logger.info(f"  {obj_type}: +{size / (1024 * 1024):.1f} MB")

    # Cleanup
    trainer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
