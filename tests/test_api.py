"""Smoke tests for the FastAPI app — Redis is mocked."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_redis():
    fake = MagicMock()
    fake.get.return_value = None
    fake.ping.return_value = True
    with patch("src.cache.redis_client.get_redis", return_value=fake):
        yield fake


@pytest.fixture
def client(mock_redis):
    from src.main import app
    return TestClient(app)


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "the-factory"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_picks_pending_when_no_cache(client, mock_redis):
    mock_redis.get.return_value = None
    r = client.get("/picks")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["picks"] == []


def test_picks_returns_cached(client, mock_redis):
    today = date.today().isoformat()
    cached_picks = [{"batter_name": "Aaron Judge", "team": "NYY"}]

    def fake_get(key):
        if key == f"picks:{today}":
            return json.dumps(cached_picks)
        return None

    mock_redis.get.side_effect = fake_get
    r = client.get("/picks")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cached"
    assert body["picks"][0]["batter_name"] == "Aaron Judge"
