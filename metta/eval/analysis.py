"""
metta/eval/analysis.py

Analysis utilities for evaluation results.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from tabulate import tabulate

from metta.agent.metta_agent import MettaAgent
from metta.eval.analysis_config import AnalysisConfig
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_stats_db import SimulationStatsDB
from mettagrid.util.file import local_copy

logger = logging.getLogger(__name__)


def analyze(policy_record: MettaAgent, config: AnalysisConfig) -> None:
    logger.info(f"Analyzing policy: {policy_record.uri}")
    logger.info(f"Using eval DB: {config.eval_db_uri}")

    with local_copy(config.eval_db_uri) as local_path:
        stats_db = EvalStatsDB(Path(local_path))

        sample_count = get_sample_count(stats_db, policy_record)
        if sample_count == 0:
            logger.warning(f"No samples found for policy: {policy_record.key}:v{policy_record.version}")
            return
        logger.info(f"Total sample count for specified policy/suite: {sample_count}")

        available_metrics = get_available_metrics(stats_db, policy_record)
        logger.info(f"Available metrics: {available_metrics}")

        selected_metrics = filter_metrics(available_metrics, config.metrics)
        if not selected_metrics:
            logger.warning(f"No metrics found matching patterns: {config.metrics}")
            return
        logger.info(f"Selected metrics: {selected_metrics}")

        metrics_data = get_metrics_data(stats_db, policy_record, selected_metrics, config.suite)
        print_metrics_table(metrics_data, policy_record)


# --------------------------------------------------------------------------- #
#   helpers                                                                   #
# --------------------------------------------------------------------------- #
def get_available_metrics(stats_db: EvalStatsDB, policy_record: MettaAgent) -> List[str]:
    policy_key, policy_version = policy_record.key_and_version()
    result = stats_db.query(
        f"""
        SELECT DISTINCT metric
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{policy_key}'
           AND policy_version =  {policy_version}
         ORDER BY metric
        """
    )
    return [] if result.empty else result["metric"].tolist()


def filter_metrics(available_metrics: List[str], patterns: List[str]) -> List[str]:
    if not patterns or patterns == ["*"]:
        return available_metrics
    selected = []
    for pattern in patterns:
        selected.extend(m for m in available_metrics if fnmatch.fnmatch(m, pattern))
    return list(dict.fromkeys(selected))  # dedupe, preserve order


def get_metrics_data(
    stats_db: EvalStatsDB,
    policy_record: MettaAgent,
    metrics: List[str],
    suite: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return {metric: {"mean": μ, "std": σ,
                     "count": K_recorded,
                     "samples": N_potential}}
        • μ, σ are normalised (missing values = 0).
        • K_recorded  – rows in policy_simulation_agent_metrics.
        • N_potential – total agent-episode pairs for that filter.
    """
    policy_key, policy_version = policy_record.key_and_version()
    filter_condition = f"sim_suite = '{suite}'" if suite else None

    data: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        mean = get_metric_by_policy(stats_db, m, policy_record, filter_condition)
        if mean is None:
            continue
        std = get_metric_by_policy_std(stats_db, m, policy_record, filter_condition) or 0.0

        k_recorded = get_metric_by_policy_count(stats_db, m, policy_record, filter_condition)
        n_potential = get_sample_count(stats_db, policy_record, suite)

        data[m] = {
            "mean": mean,
            "std": std,
            "count": k_recorded,
            "samples": n_potential,
        }
    return data


def print_metrics_table(metrics_data: Dict[str, Dict[str, float]], policy_record: MettaAgent) -> None:
    logger.info(f"\nMetrics for policy: {policy_record.uri}\n")
    if not metrics_data:
        logger.warning(f"No metrics data available for {policy_record.key}:v{policy_record.version}")
        return

    headers = ["Metric", "Average", "Std Dev", "Metric Samples", "Agent Samples"]
    rows = [
        [
            metric,
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            str(int(stats["count"])),
            str(int(stats["samples"])),
        ]
        for metric, stats in metrics_data.items()
    ]

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    logger.info("")


def load_eval_db(path: str) -> EvalStatsDB:
    """Load an evaluation database from a path."""
    return EvalStatsDB(Path(path))


def load_sim_db(path: str) -> SimulationStatsDB:
    """Load a simulation database from a path."""
    return SimulationStatsDB(Path(path))


def get_metric_by_policy(
    db: Union[EvalStatsDB, SimulationStatsDB],
    metric: str,
    policy_record: MettaAgent,
    filter_condition: str | None = None,
) -> Optional[float]:
    """Get the mean value of a metric for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.metric_by_policy_eval(metric, policy_record, filter_condition)
    else:
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0
        query = f"""
        SELECT AVG(e.{metric})
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = db.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None


def get_metric_by_policy_std(
    db: Union[EvalStatsDB, SimulationStatsDB],
    metric: str,
    policy_record: MettaAgent,
    filter_condition: str | None = None,
) -> Optional[float]:
    """Get the standard deviation of a metric for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.metric_by_policy_eval_std(metric, policy_record, filter_condition)
    else:
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0
        query = f"""
        SELECT STDDEV(e.{metric})
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = db.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None


def get_metric_by_policy_count(
    db: Union[EvalStatsDB, SimulationStatsDB],
    metric: str,
    policy_record: MettaAgent,
    filter_condition: str | None = None,
) -> Optional[int]:
    """Get the count of a metric for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.metric_by_policy_eval_count(metric, policy_record, filter_condition)
    else:
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0
        query = f"""
        SELECT COUNT(e.{metric})
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = db.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None


def get_sample_count(
    db: Union[EvalStatsDB, SimulationStatsDB],
    policy_record: Optional[MettaAgent] = None,
    sim_suite: Optional[str] = None,
    sim_name: Optional[str] = None,
) -> int:
    """Get the number of samples for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.sample_count(policy_record, sim_suite, sim_name)
    else:
        query = """
        SELECT COUNT(*)
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE 1=1
        """
        params = []

        if policy_record is not None:
            # For now, we'll use placeholder values for policy key and version
            pk = "metta_agent"
            pv = 0
            query += " AND s.policy_key = ? AND s.policy_version = ?"
            params.extend([pk, pv])

        if sim_suite is not None:
            query += " AND s.suite = ?"
            params.append(sim_suite)

        if sim_name is not None:
            query += " AND s.name = ?"
            params.append(sim_name)

        result = db.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else 0


def get_simulation_scores(
    db: Union[EvalStatsDB, SimulationStatsDB],
    policy_record: MettaAgent,
    metric: str,
) -> Dict[Tuple[str, str, str], float]:
    """Get the simulation scores for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.simulation_scores(policy_record, metric)
    else:
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0

        query = f"""
        WITH sim_means AS (
            SELECT s.suite, s.name, s.env, AVG(e.{metric}) as mean_metric
            FROM episodes e
            JOIN simulations s ON e.simulation_id = s.id
            WHERE s.policy_key = ? AND s.policy_version = ?
            GROUP BY s.suite, s.name, s.env
        )
        SELECT suite, name, env, mean_metric
        FROM sim_means
        ORDER BY suite, name, env
        """
        result = db.con.execute(query, (pk, pv)).fetchall()
        return {(row[0], row[1], row[2]): row[3] for row in result}


def get_simulation_breakdown(
    db: Union[EvalStatsDB, SimulationStatsDB],
    metric: str,
    policy_record: MettaAgent | None = None,
) -> pd.DataFrame:
    """Get the simulation breakdown for a policy."""
    if isinstance(db, EvalStatsDB):
        return db.simulation_breakdown(metric, policy_record)
    else:
        if policy_record is not None:
            # For now, we'll use placeholder values for policy key and version
            pk = "metta_agent"
            pv = 0
            policy_filter = "AND s.policy_key = ? AND s.policy_version = ?"
            params = (pk, pv)
        else:
            policy_filter = ""
            params = ()

        query = f"""
        SELECT s.suite, s.name, s.env,
               AVG(e.{metric}) as metric,
               STDDEV(e.{metric}) as std,
               COUNT(*) as n
        FROM episodes e
        JOIN simulations s ON e.simulation_id = s.id
        WHERE e.{metric} IS NOT NULL {policy_filter}
        GROUP BY s.suite, s.name, s.env
        ORDER BY s.suite, s.name, s.env
        """
        return pd.DataFrame(
            db.con.execute(query, params).fetchall(), columns=["suite", "name", "env", "metric", "std", "n"]
        )
