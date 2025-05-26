"""
EvalStatsDb adds views on top of SimulationStatsDb
to make it easier to query policy performance across simulations,
while handling the fact that some metrics are only logged when non‑zero.

Normalisation rule
------------------
For every query we:
1.  Count the **potential** agent‑episode samples for the policy / filter.
2.  Aggregate the recorded metric values (missing = 0).
3.  Divide by the potential count.

This yields a true mean even when zeros are omitted from logging.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from metta.agent.metta_agent import MettaAgent
from metta.sim.simulation_stats_db import SimulationStatsDB
from mettagrid.util.file import local_copy

# --------------------------------------------------------------------------- #
#   Views                                                                     #
# --------------------------------------------------------------------------- #
EVAL_DB_VIEWS: Dict[str, str] = {
    # All agent‑episode samples for every policy/simulation (regardless of metrics)
    "policy_simulation_agent_samples": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_samples AS
      SELECT
          ap.policy_key,
          ap.policy_version,
          s.suite  AS sim_suite,
          s.name   AS sim_name,
          s.env    AS sim_env,
          ap.episode_id,
          ap.agent_id
        FROM agent_policies ap
        JOIN episodes   e ON e.id = ap.episode_id
        JOIN simulations s ON s.id = e.simulation_id
    """,
    # Recorded per‑agent metrics (a subset of the above when metric ≠ 0)
    "policy_simulation_agent_metrics": """
    CREATE VIEW IF NOT EXISTS policy_simulation_agent_metrics AS
      SELECT
          ap.policy_key,
          ap.policy_version,
          s.suite  AS sim_suite,
          s.name   AS sim_name,
          s.env    AS sim_env,
          am.metric,
          am.value
        FROM agent_metrics am
        JOIN agent_policies ap
              ON ap.episode_id = am.episode_id
             AND ap.agent_id   = am.agent_id
        JOIN episodes   e ON e.id = am.episode_id
        JOIN simulations s ON s.id = e.simulation_id
    """,
}


class EvalStatsDB(SimulationStatsDB):
    # ------------------------------------------------------------------ #
    #   Construction / schema                                            #
    # ------------------------------------------------------------------ #
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    @classmethod
    @contextmanager
    def from_uri(cls, path: str):
        """Download (if remote), open, and yield an EvalStatsDB."""
        with local_copy(path) as local_path:
            db = cls(local_path)
            yield db

    @staticmethod
    def from_sim_stats_db(sim_stats_db: SimulationStatsDB) -> EvalStatsDB:
        """Create an EvalStatsDB from a SimulationStatsDB."""
        return EvalStatsDB(sim_stats_db.path)

    # Extend parent schema with the extra views
    def tables(self) -> Dict[str, str]:
        return {**super().tables(), **EVAL_DB_VIEWS}

    # ------------------------------------------------------------------ #
    #   Potential / recorded sample counters                             #
    # ------------------------------------------------------------------ #
    def _count_agent_samples(
        self,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> int:
        """Internal helper: number of agent‑episode pairs (possible samples)."""
        q = f"""
        SELECT COUNT(*) AS cnt
          FROM policy_simulation_agent_samples
         WHERE policy_key     = '{policy_key}'
           AND policy_version = {policy_version}
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        res = self.query(q)
        return int(res["cnt"][0]) if not res.empty else 0

    # Public alias (referenced by downstream code/tests)
    def potential_samples_for_metric(
        self,
        policy_key: str,
        policy_version: int,
        filter_condition: str | None = None,
    ) -> int:
        return self._count_agent_samples(policy_key, policy_version, filter_condition)

    def count_metric_agents(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        filter_condition: str | None = None,
    ) -> int:
        """How many samples actually recorded *metric* > 0."""
        q = f"""
        SELECT COUNT(*) AS cnt
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{policy_key}'
           AND policy_version = {policy_version}
           AND metric         = '{metric}'
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        res = self.query(q)
        return int(res["cnt"][0]) if not res.empty else 0

    # ------------------------------------------------------------------ #
    #   Normalised aggregations                                          #
    # ------------------------------------------------------------------ #
    def _normalised_value(
        self,
        policy_key: str,
        policy_version: int,
        metric: str,
        agg: str,  # "SUM", "AVG", or "STD"
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """Return SUM/AVG/STD after zero‑filling missing samples."""
        potential = self.potential_samples_for_metric(policy_key, policy_version, filter_condition)
        if potential == 0:
            return None

        # Aggregate only over recorded rows
        q = f"""
        SELECT
            SUM(value)       AS s1,
            SUM(value*value) AS s2,
            COUNT(*)         AS k,
            AVG(value)       AS r_avg
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{policy_key}'
           AND policy_version = {policy_version}
           AND metric         = '{metric}'
        """
        if filter_condition:
            q += f" AND {filter_condition}"
        r = self.query(q)
        if r.empty:
            return 0.0 if agg in {"SUM", "AVG"} else 0.0

        # DuckDB returns NULL→NaN when no rows match; coalesce to 0
        s1_val, s2_val, _ = r.iloc[0][["s1", "s2", "k"]]
        s1 = 0.0 if pd.isna(s1_val) else float(s1_val)
        s2 = 0.0 if pd.isna(s2_val) else float(s2_val)

        if agg == "SUM":
            return s1 / potential
        if agg == "AVG":
            return s1 / potential
        if agg == "STD":
            mean = s1 / potential
            var = (s2 / potential) - mean**2
            return math.sqrt(max(var, 0.0))
        raise ValueError(f"Unknown aggregation {agg}")

    # Convenience wrappers ------------------------------------------------
    def get_average_metric_by_filter(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = policy_record.key_and_version()
        return self._normalised_value(pk, pv, metric, "AVG", filter_condition)

    def get_sum_metric_by_filter(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = policy_record.key_and_version()
        return self._normalised_value(pk, pv, metric, "SUM", filter_condition)

    def get_std_metric_by_filter(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        pk, pv = policy_record.key_and_version()
        return self._normalised_value(pk, pv, metric, "STD", filter_condition)

    # ------------------------------------------------------------------ #
    #   Utilities                                                        #
    # ------------------------------------------------------------------ #
    def sample_count(
        self,
        policy_record: Optional[MettaAgent] = None,
        sim_suite: Optional[str] = None,
        sim_name: Optional[str] = None,
    ) -> int:
        """Return the number of samples for *policy_record*."""
        query = """
        SELECT COUNT(*)
        FROM episodes e
        JOIN evaluations s ON e.evaluation_id = s.id
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

        result = self.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else 0

    # ------------------------------------------------------------------ #
    #   Per‑simulation breakdown                                         #
    # ------------------------------------------------------------------ #
    def simulation_scores(self, policy_record: MettaAgent, metric: str) -> Dict[tuple, float]:
        """Return { (suite,name,env) : normalised mean(metric) }."""
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0

        query = f"""
        WITH sim_means AS (
            SELECT s.suite, s.name, s.env, AVG(e.{metric}) as mean_metric
            FROM episodes e
            JOIN evaluations s ON e.evaluation_id = s.id
            WHERE s.policy_key = ? AND s.policy_version = ?
            GROUP BY s.suite, s.name, s.env
        )
        SELECT suite, name, env, mean_metric
        FROM sim_means
        ORDER BY suite, name, env
        """
        result = self.con.execute(query, (pk, pv)).fetchall()
        return {(row[0], row[1], row[2]): row[3] for row in result}

    def simulation_breakdown(
        self,
        metric: str,
        policy_record: MettaAgent | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
        - suite: evaluation suite name
        - name: evaluation name
        - env: environment name
        - metric: mean value of the metric
        - std: standard deviation of the metric
        - n: number of episodes
        """
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
        JOIN evaluations s ON e.evaluation_id = s.id
        WHERE e.{metric} IS NOT NULL {policy_filter}
        GROUP BY s.suite, s.name, s.env
        ORDER BY s.suite, s.name, s.env
        """
        return pd.DataFrame(
            self.con.execute(query, params).fetchall(), columns=["suite", "name", "env", "metric", "std", "n"]
        )

    def metric_by_policy_eval(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """Return the mean value of *metric* for *policy_record*."""
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0

        query = f"""
        SELECT AVG(e.{metric})
        FROM episodes e
        JOIN evaluations s ON e.evaluation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = self.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None

    def metric_by_policy_eval_std(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[float]:
        """Return the standard deviation of *metric* for *policy_record*."""
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0

        query = f"""
        SELECT STDDEV(e.{metric})
        FROM episodes e
        JOIN evaluations s ON e.evaluation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = self.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None

    def metric_by_policy_eval_count(
        self,
        metric: str,
        policy_record: MettaAgent,
        filter_condition: str | None = None,
    ) -> Optional[int]:
        """Return the count of *metric* for *policy_record*."""
        # For now, we'll use placeholder values for policy key and version
        pk = "metta_agent"
        pv = 0

        query = f"""
        SELECT COUNT(e.{metric})
        FROM episodes e
        JOIN evaluations s ON e.evaluation_id = s.id
        WHERE s.policy_key = ? AND s.policy_version = ?
        """
        params = [pk, pv]

        if filter_condition:
            query += f" AND {filter_condition}"

        result = self.con.execute(query, params).fetchone()
        return result[0] if result and result[0] is not None else None
