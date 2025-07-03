"""Customer management functions."""

from __future__ import annotations

from typing import Any, Dict

from .client import LLMCostsClient


def list_customers(client: LLMCostsClient) -> Any:
    return client.get("/customers")


def get_customer(client: LLMCostsClient, customer_id: str) -> Any:
    return client.get(f"/customers/{customer_id}")


def create_customer(client: LLMCostsClient, data: Dict[str, Any]) -> Any:
    return client.post("/customers", json=data)


def update_customer(client: LLMCostsClient, customer_id: str, data: Dict[str, Any]) -> Any:
    return client.put(f"/customers/{customer_id}", json=data)


def delete_customer(client: LLMCostsClient, customer_id: str) -> Any:
    return client.delete(f"/customers/{customer_id}")
