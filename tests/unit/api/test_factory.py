from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rfc_atlas.api.factory import create_app


@patch("rfc_atlas.api.factory.SynthesisOrchestrator")
def test_app_lifespan_and_health_check(mock_orchestrator_class: MagicMock) -> None:
    app = create_app()

    with TestClient(app) as client:
        mock_orchestrator_class.assert_called_once()
        assert hasattr(app.state, "engine")
        assert app.state.engine is not None

        response = client.get("/health")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
        assert response.json() == {"status": "healthy", "service": "rfc-atlas-api"}  # pyright: ignore[reportUnknownMemberType]

    assert getattr(app.state, "engine", None) is None
