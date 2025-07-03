"""Cost event helper functions."""

from __future__ import annotations

from typing import Any, Dict

from .client import LLMCostsClient


def list_events(client: LLMCostsClient) -> Any:
    return client.get("/events")


def get_event(client: LLMCostsClient, event_id: str) -> Any:
    return client.get(f"/events/{event_id}")


def create_event(client: LLMCostsClient, data: Dict[str, Any]) -> Any:
    return client.post("/events", json=data)


def update_event(client: LLMCostsClient, event_id: str, data: Dict[str, Any]) -> Any:
    return client.put(f"/events/{event_id}", json=data)


def delete_event(client: LLMCostsClient, event_id: str) -> Any:
    return client.delete(f"/events/{event_id}")
