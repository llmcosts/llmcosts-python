"""Customer limit management."""

from __future__ import annotations

from typing import Any, Dict

from .client import LLMCostsClient


def list_limits(client: LLMCostsClient) -> Any:
    return client.get("/limits")


def get_limit(client: LLMCostsClient, limit_id: str) -> Any:
    return client.get(f"/limits/{limit_id}")


def create_limit(client: LLMCostsClient, data: Dict[str, Any]) -> Any:
    return client.post("/limits", json=data)


def update_limit(client: LLMCostsClient, limit_id: str, data: Dict[str, Any]) -> Any:
    return client.put(f"/limits/{limit_id}", json=data)


def delete_limit(client: LLMCostsClient, limit_id: str) -> Any:
    return client.delete(f"/limits/{limit_id}")
