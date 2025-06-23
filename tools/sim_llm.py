#!/usr/bin/env -S uv run
"""
LLM agent simulation tool - a wrapper around sim.py for running simulations with LLM agents.

This tool simplifies running simulations with LLM agents by providing LLM-specific defaults
and avoiding the need to specify policy URIs.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run simulation with LLM agent using the specialized config."""

    # Get the path to sim.py
    sim_script = Path(__file__).parent / "sim.py"

    # Build command with LLM config
    cmd = [sys.executable, "-m", "tools.sim", "--config-name", "sim_job_llm"]

    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Run the simulation
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
