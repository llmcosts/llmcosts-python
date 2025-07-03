"""Alert management functions."""

from __future__ import annotations

from typing import Any, Dict

from .client import LLMCostsClient


def list_alerts(client: LLMCostsClient) -> Any:
    return client.get("/alerts")


def get_alert(client: LLMCostsClient, alert_id: str) -> Any:
    return client.get(f"/alerts/{alert_id}")


def create_alert(client: LLMCostsClient, data: Dict[str, Any]) -> Any:
    return client.post("/alerts", json=data)


def update_alert(client: LLMCostsClient, alert_id: str, data: Dict[str, Any]) -> Any:
    return client.put(f"/alerts/{alert_id}", json=data)


def delete_alert(client: LLMCostsClient, alert_id: str) -> Any:
    return client.delete(f"/alerts/{alert_id}")
