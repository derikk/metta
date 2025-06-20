import gc
import os

import numpy as np
import psutil

from mettagrid.curriculum import SingleTaskCurriculum
from mettagrid.game_builders import game_base
from mettagrid.mettagrid_env import MettaGridEnv


def test_memory_leak_fixed():
    """Test that memory usage doesn't grow unboundedly over episodes."""

    # Create a simple environment config
    env_cfg = game_base()
    env_cfg["game"]["max_steps"] = 100
    env_cfg["game"]["num_agents"] = 4

    curriculum = SingleTaskCurriculum("test_task", env_cfg)

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run many episodes
    env = MettaGridEnv(curriculum, render_mode=None)

    memory_samples = []
    episode_interval = 100

    for episode in range(1000):
        obs, _ = env.reset()

        # Run episode
        done = False
        while not done:
            actions = np.random.randint(0, 5, size=(env.num_agents, 2), dtype=np.int32)
            obs, rewards, terminals, truncations, infos = env.step(actions)
            done = terminals.all() or truncations.all()

        # Sample memory usage periodically
        if episode % episode_interval == 0:
            gc.collect()  # Force garbage collection
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            memory_samples.append(memory_growth)
            print(f"Episode {episode}: Memory growth = {memory_growth:.2f} MB")

    env.close()

    # Check that memory growth is bounded
    # Allow some growth for caches, but it shouldn't be linear
    max_allowed_growth = 50  # MB
    final_growth = memory_samples[-1]

    print(f"\nMemory growth over {len(memory_samples)} samples:")
    print("Initial: 0 MB")
    print(f"Final: {final_growth:.2f} MB")
    print(f"Max allowed: {max_allowed_growth} MB")

    # Check that growth rate decreases (not linear)
    if len(memory_samples) > 3:
        early_growth_rate = memory_samples[2] / 3
        late_growth_rate = memory_samples[-1] - memory_samples[-2]
        print(f"Early growth rate: {early_growth_rate:.2f} MB/sample")
        print(f"Late growth rate: {late_growth_rate:.2f} MB/sample")

        # Late growth should be much less than early growth
        assert late_growth_rate < early_growth_rate * 0.5, (
            f"Memory growth appears linear: early={early_growth_rate:.2f}, late={late_growth_rate:.2f}"
        )

    assert final_growth < max_allowed_growth, (
        f"Memory growth {final_growth:.2f} MB exceeds limit {max_allowed_growth} MB"
    )

    print("\nMemory leak test PASSED!")


if __name__ == "__main__":
    test_memory_leak_fixed()
