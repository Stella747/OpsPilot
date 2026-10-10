
import numpy as np
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from database import Base
from models import IncidentDB


@pytest.fixture
def client():
    # Create a temporary in-memory database for this test
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )

    Base.metadata.create_all(bind=test_engine)

    db = TestSessionLocal()
    db.add(
        IncidentDB(
            id="TEST-001",
            service="Payment API",
            severity="High",
            description="Payment requests are experiencing high latency.",
        )
    )
    db.commit()
    db.close()

    # Make the API use our temporary database
    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    main.app.dependency_overrides[main.get_db] = override_get_db

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_weak_evidence_skips_ollama(client, monkeypatch):
    # Force all historical similarity scores below 0.40
    monkeypatch.setattr(
        main,
        "cosine_similarity",
        lambda query, historical: np.array(
            [[0.20, 0.30, 0.10, 0.15, 0.25]]
        ),
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("Ollama should not be called for weak evidence")

    monkeypatch.setattr(main.ollama, "chat", fail_if_called)

    response = client.post(
        "/incidents/TEST-001/investigate"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["sufficient_historical_evidence"] is False
    assert result["best_similarity"] == 0.30
    assert "Insufficient historical evidence" in (
        result["investigation_report"]
    )


def test_strong_evidence_calls_ollama(client, monkeypatch):
    # Force one strong historical match
    monkeypatch.setattr(
        main,
        "cosine_similarity",
        lambda query, historical: np.array(
            [[0.80, 0.30, 0.20, 0.10, 0.40]]
        ),
    )

    # Return a fake response instead of running the local LLM
    def fake_ollama_chat(*args, **kwargs):
        return {
            "message": {
                "content": "Mock investigation report"
            }
        }

    monkeypatch.setattr(
        main.ollama,
        "chat",
        fake_ollama_chat,
    )

    response = client.post(
        "/incidents/TEST-001/investigate"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["sufficient_historical_evidence"] is True
    assert result["best_similarity"] == 0.80
    assert result["investigation_report"] == (
        "Mock investigation report"
    )
