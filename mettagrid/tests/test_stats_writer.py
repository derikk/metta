"""
Unit tests for the StatsWriter functionality in mettagrid.stats_writer.
"""

import pytest
from testcontainers.postgres import PostgresContainer

from mettagrid.postgres_stats_db import PostgresStatsDB
from mettagrid.stats_writer import StatsWriter


@pytest.fixture
def postgres_container():
    """Create a PostgreSQL container for testing."""
    with PostgresContainer("postgres:17", driver=None) as postgres:
        yield postgres


@pytest.fixture
def db_url(postgres_container):
    """Get the database URL for the PostgreSQL container."""
    return postgres_container.get_connection_url()


def test_episode_lifecycle(db_url):
    """Test the full lifecycle of an episode."""
    writer = StatsWriter(db_url, eval_name="test_eval", simulation_suite="test_suite")

    with PostgresStatsDB(db_url) as db:
        policy_id_1 = db.create_policy("test_policy_1", "test_description", "test_url", None)
        policy_id_2 = db.create_policy("test_policy_2", "test_description", "test_url", None)

    # Set agent policies (required for the new schema)
    agent_policies = {0: policy_id_1, 1: policy_id_2}  # agent_id -> policy_id mapping
    writer.set_agent_policies(agent_policies)

    # Episode attributes
    attributes = {"seed": "12345", "map_w": "10", "map_h": "20", "meta": '{"key": "value"}'}

    # Metrics
    agent_metrics = {0: {"reward": 10.5, "steps": 50.0}, 1: {"reward": 8.2, "steps": 45.0}}

    # Replay URL
    replay_url = "https://example.com/replay.json"

    # Record the complete episode
    writer.record_episode(agent_metrics, replay_url, attributes)

    # Verify data in database using PostgresStatsDB
    with PostgresStatsDB(db_url) as db:
        # Check episode exists
        result = db.query("SELECT id FROM episodes WHERE replay_url = %s", (replay_url,))
        assert len(result) == 1
        episode_id = result[0][0]

        # Check episode attributes (stored as JSONB)
        result = db.query("SELECT attributes FROM episodes WHERE id = %s", (episode_id,))
        assert len(result) == 1
        stored_attributes = result[0][0]
        for attr, value in attributes.items():
            assert stored_attributes[attr] == value

        # Check agent metrics
        for agent_id, metrics in agent_metrics.items():
            for metric, value in metrics.items():
                result = db.query(
                    "SELECT value FROM episode_agent_metrics WHERE episode_id = %s AND agent_id = %s AND metric = %s",
                    (episode_id, agent_id, metric),
                )
                assert len(result) == 1
                assert abs(result[0][0] - value) < 1e-6  # Compare floats with tolerance

        # Check agent policies
        for agent_id, policy_id in agent_policies.items():
            result = db.query(
                "SELECT policy_id FROM episode_agent_policies WHERE episode_id = %s AND agent_id = %s",
                (episode_id, agent_id),
            )
            assert len(result) == 1
            assert result[0][0] == policy_id

        # Check eval_name and simulation_suite
        result = db.query("SELECT eval_name, simulation_suite FROM episodes WHERE id = %s", (episode_id,))
        assert len(result) == 1
        assert result[0][0] == "test_eval"
        assert result[0][1] == "test_suite"

        # Check replay URL
        result = db.query("SELECT replay_url FROM episodes WHERE id = %s", (episode_id,))
        assert len(result) == 1
        assert result[0][0] == replay_url
