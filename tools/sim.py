# tools/sim.py
"""
Simulation driver for evaluating policies in the Metta environment.

 ▸ For every requested *policy URI*
   ▸ choose the checkpoint(s) according to selector/metric
   ▸ run the configured `SimulationSuite`
   ▸ export the merged stats DB if an output URI is provided
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.config import Config
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment

# --------------------------------------------------------------------------- #
# Config objects                                                              #
# --------------------------------------------------------------------------- #


class FetchPoliciesConfig(Config):
    """
    Configuration for fetching and filtering policies from Weights & Biases.
    """

    filter_tags: List[str] = []
    filter_metadata: Dict[str, Any] = {}
    filter_job_type: str = "train"  # Default to fetch from training runs
    filter_max_age_days: int = 7  # Default to fetch runs from last week
    entity: str = "metta-research"
    project: str = "metta"
    sort_by: str = "created_at"
    sort_order: str = "desc"  # "asc" or "desc"
    limit: int = 10
    print_only: bool = False  # If True, just print policies without running simulations
    print_details: bool = True  # If True, print detailed metadata for each policy


class SimJob(Config):
    simulation_suite: SimulationSuiteConfig
    policy_uris: List[str]
    selector_type: str = "top"
    replay_dir: str = "s3://softmax-public/replays/evals"
    stats_db_uri: str
    stats_dir: str  # The (local) directory where stats should be stored
    # If provided, automatically fetch policies from Weights & Biases
    fetch_policies_config: FetchPoliciesConfig = FetchPoliciesConfig()

    def get_policy_uris(self) -> List[str]:
        """
        Get policy URIs to evaluate.

        If fetch_policies_config is set, fetch policies based on the config.
        Otherwise, return the explicitly provided policy_uris.

        Args:
            policy_store: The PolicyStore instance to use for fetching policies

        Returns:
            List of policy URIs
        """
        if self.fetch_policies_config:
            policies = fetch_matching_policies(self.fetch_policies_config)
            return [p["uri"] for p in policies]
        return self.policy_uris


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def simulate_policy(
    sim_job: SimJob,
    policy_uri: str,
    cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.
    """

    policy_store = PolicyStore(cfg, None)
    # TODO: institutionalize this better?
    metric = sim_job.simulation_suite.name + "_score"
    policy_prs = policy_store.policies(policy_uri, sim_job.selector_type, n=1, metric=metric)

    # For each checkpoint of the policy, simulate
    for pr in policy_prs:
        logger.info("Evaluating policy %s", pr.uri)

        stats_dir = f"{sim_job.stats_dir}/{pr.name}"
        replay_dir = f"{sim_job.replay_dir}/{pr.name}"
        sim = SimulationSuite(
            config=sim_job.simulation_suite,
            policy_pr=pr,
            policy_store=policy_store,
            replay_dir=replay_dir,
            stats_dir=stats_dir,
        )
        results = sim.simulate()
        # ------------------------------------------------------------------ #
        # Export                                                             #
        # ------------------------------------------------------------------ #
        logger.info("Exporting merged stats DB → %s", sim_job.stats_db_uri)
        results.stats_db.export(sim_job.stats_db_uri)

        logger.info("Evaluation complete for policy %s", pr.uri)


def fetch_matching_policies(config: FetchPoliciesConfig) -> List[dict]:
    """
    Fetch policies from wandb based on the configured filters.
    Focuses on finding policies stored as model artifacts in runs.

    Args:
        config: The FetchPoliciesConfig instance with filtering and sorting options

    Returns:
        List of matching policy metadata dictionaries
    """
    api = wandb.Api()

    # Build filters for the runs query
    filters = {}

    # Add job type filter
    if config.filter_job_type:
        filters["jobType"] = config.filter_job_type

    # Add time filter
    if config.filter_max_age_days > 0:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=config.filter_max_age_days)
        filters["createdAt"] = {"$gte": cutoff_date.strftime("%Y-%m-%dT%H:%M:%S")}

    # Add tag filters
    if config.filter_tags:
        filters["tags"] = {"$in": config.filter_tags}

    # Add metadata filters for run config
    for key, value in config.filter_metadata.items():
        filters[f"config.{key}"] = value

    print(f"Querying runs with filters: {filters}")

    # Get runs from the project
    try:
        runs = api.runs(
            f"{config.entity}/{config.project}",
            filters=filters,
            order=f"{'-' if config.sort_order == 'desc' else ''}{config.sort_by}",
            per_page=min(100, config.limit * 2),  # Get more runs than needed since not all will have model artifacts
        )
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

    matching_policies = []

    # Process each run
    for run in runs:
        try:
            # Check if the run has model artifacts
            model_artifacts = []
            print(f"Run {run.name} has {len(run.logged_artifacts())} artifacts")

            for artifact in run.logged_artifacts():
                if artifact.type == "model":
                    print(f"  Artifact {artifact.id} is type {artifact.type}")
                    model_artifacts.append(artifact)

            # Skip runs with no model artifacts
            if not model_artifacts:
                continue

            # Process each model artifact
            for artifact in model_artifacts:
                # Create qualified name
                qualified_name = f"{config.entity}/{config.project}/{artifact.name}:v{artifact.version}"
                uri = f"wandb://{qualified_name}"

                # Create metadata from run and artifact
                metadata = {
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_url": run.url,
                    "tags": run.tags,
                    "created_at": artifact.created_at,
                    "artifact_id": artifact.id,
                    "artifact_version": artifact.version,
                    "artifact_name": artifact.name,
                    "qualified_name": qualified_name,
                }

                # Add artifact metadata
                metadata.update(artifact.metadata)

                # Add config items to metadata
                if hasattr(run, "config") and run.config:
                    for key, value in run.config.items():
                        if key not in metadata and value is not None:
                            metadata[key] = value

                policy_info = {
                    "name": f"{artifact.name}:v{artifact.version}",
                    "uri": uri,
                    "metadata": metadata,
                    "tags": run.tags,
                    "created_at": artifact.created_at,
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_url": run.url,
                }

                matching_policies.append(policy_info)

        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue

        # If we have enough policies, stop processing runs
        if len(matching_policies) >= config.limit:
            break

    # Sort policies
    if config.sort_by in ["created_at", "updated_at"]:
        matching_policies.sort(key=lambda p: p.get(config.sort_by, ""), reverse=(config.sort_order == "desc"))
    else:
        # For metadata fields
        matching_policies.sort(
            key=lambda p: p.get("metadata", {}).get(config.sort_by, 0), reverse=(config.sort_order == "desc")
        )

    # Limit the results
    return matching_policies[: config.limit]


def print_matching_policies(config: FetchPoliciesConfig) -> None:
    """
    Print information about matching policies and available metadata fields.

    Args:
        config: The FetchPoliciesConfig instance with filtering and sorting options
    """
    policies = fetch_matching_policies(config)

    if not policies:
        print("No matching policies found.")
        return

    print(f"Found {len(policies)} matching policies:")
    print(f"{'Name':<30} | {'URI':<50} | {'Run ID':<20} | {'Created At':<20}")
    print("-" * 130)
    for p in policies:
        print(f"{p['name']:<30} | {p['uri']:<50} | {p.get('run_id', 'N/A'):<20} | {p.get('created_at', 'N/A'):<20}")

    # Print available metadata fields for filtering
    if policies:
        all_metadata_keys = set()
        for p in policies:
            all_metadata_keys.update(p.get("metadata", {}).keys())

        print("\nAvailable metadata fields for filtering:")
        print(", ".join(sorted(all_metadata_keys)))

    # Print detailed metadata for each policy
    if config.print_details:
        print("\nDetailed metadata for each policy:")
        for i, p in enumerate(policies):
            print(f"\n--- Policy {i + 1}: {p['name']} ---")
            print(f"URI: {p['uri']}")
            print(f"Run: {p['run_name']} ({p['run_id']})")
            print(f"Created: {p['created_at']}")
            print("Metadata:")
            for key, value in sorted(p.get("metadata", {}).items()):
                # Format the value for better readability
                if isinstance(value, dict):
                    value_str = json.dumps(value, indent=2)
                    # Indent each line
                    value_str = "\n    ".join(value_str.splitlines())
                else:
                    value_str = str(value)
                print(f"  {key}: {value_str}")


def get_policy_uris_from_config(config: FetchPoliciesConfig) -> List[str]:
    """
    Get policy URIs based on the FetchPoliciesConfig.

    Args:
        config: The FetchPoliciesConfig instance with filtering and sorting options

    Returns:
        List of policy URIs
    """
    policies = fetch_matching_policies(config)
    return [p["uri"] for p in policies]


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="../configs", config_name="sim_job")
def main(cfg: DictConfig) -> None:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.sim")
    logger.info(f"Sim job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    sim_job = SimJob(cfg.sim_job)
    assert isinstance(sim_job, SimJob)

    print_matching_policies(sim_job.fetch_policies_config)


#    policy_uris = sim_job.get_policy_uris()

# for policy_uri in sim_job.policy_uris:
#     simulate_policy(sim_job, policy_uri, cfg, logger)


if __name__ == "__main__":
    main()
