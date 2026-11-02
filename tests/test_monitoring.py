"""
Unit and integration tests for Prometheus metrics and Grafana dashboard configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_metrics_endpoint_exposed():
    """Verify that /metrics endpoint returns Prometheus formatted metrics."""
    # First invoke a request to ensure counters are incremented
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    content = response.text
    # Verify core Prometheus metrics are exposed
    assert "http_requests_total" in content
    assert "http_request_duration_seconds" in content


def test_grafana_dashboard_json_structure():
    """Verify that Grafana dashboard JSON file exists, is valid, and contains all required panels."""
    dashboard_file = PROJECT_ROOT / "monitoring" / "grafana" / "estateiq-dashboard.json"
    assert dashboard_file.exists(), "Grafana dashboard JSON missing"

    with open(dashboard_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("uid") == "estateiq-monitoring"
    assert "EstateIQ" in data.get("title", "")

    # Collect all panel titles and queries
    panel_titles = []
    expressions = []
    for p in data.get("panels", []):
        if p.get("title"):
            panel_titles.append(p["title"])
        for t in p.get("targets", []):
            if t.get("expr"):
                expressions.append(t["expr"])

    # Verify required panels
    assert any("Total HTTP Requests" in t for t in panel_titles)
    assert any("Pod Availability" in t for t in panel_titles)
    assert any("Failed Requests" in t for t in panel_titles)
    assert any("Latency" in t for t in panel_titles)
    assert any("Rate" in t for t in panel_titles)
    assert any("Duration" in t for t in panel_titles)
    assert any("Status Code" in t for t in panel_titles)
    assert any("CPU" in t for t in panel_titles)
    assert any("Memory" in t for t in panel_titles)

    # Verify PromQL expressions target estateiq
    assert any("http_requests_total" in expr for expr in expressions)
    assert any("http_request_duration_seconds" in expr for expr in expressions)


def test_servicemonitor_manifest():
    """Verify that ServiceMonitor manifest has correct labels, endpoints, and selectors."""
    sm_file = PROJECT_ROOT / "k8s" / "servicemonitor.yaml"
    assert sm_file.exists(), "ServiceMonitor manifest missing"

    content = sm_file.read_text(encoding="utf-8")
    assert "kind: ServiceMonitor" in content
    assert "release: monitoring" in content
    assert "port: http" in content
    assert "path: /metrics" in content

