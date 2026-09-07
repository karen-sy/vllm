# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import ExternalBlockHash
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope

SUPPORTED_PROTOCOL_VERSION = "1.0"
SUPPORTED_ACTION_VERSION = "1.0"
REQUEST_COMPLETION = "request_completion"
EVICT_ACTION_TYPE = "kv.evict"
RETAIN_ACTION_TYPE = "kv.retain"


@dataclass(frozen=True, slots=True)
class EvictBlocksAction:
    action_id: str
    block_hashes: tuple[ExternalBlockHash, ...]
    include_current_request: bool


@dataclass(frozen=True, slots=True)
class RetainBlocksAction:
    action_id: str
    block_hashes: tuple[ExternalBlockHash, ...]
    include_current_request: bool
    priority: int
    ttl_seconds: float


BlockAction: TypeAlias = EvictBlocksAction | RetainBlocksAction


def supports_envelope(envelope: KvHintsEnvelope) -> bool:
    """Return whether this consumer supports the envelope version."""
    return envelope.protocol_version == SUPPORTED_PROTOCOL_VERSION


def parse_block_action(action: KvHintAction) -> BlockAction | None:
    """Parse one G1 block action, or return None for unsupported actions."""
    if action.action_version != SUPPORTED_ACTION_VERSION:
        return None
    if action.action_type not in (EVICT_ACTION_TYPE, RETAIN_ACTION_TYPE):
        return None
    execute_at = action.payload.get("execute_at")
    if execute_at != REQUEST_COMPLETION:
        raise ValueError(
            f"{action.action_type} execute_at must be {REQUEST_COMPLETION!r}"
        )
    include_current_request = action.payload.get("include_current_request", False)
    if not isinstance(include_current_request, bool):
        raise ValueError(
            f"{action.action_type} include_current_request must be a boolean"
        )
    if action.action_type == EVICT_ACTION_TYPE:
        return EvictBlocksAction(
            action_id=action.action_id,
            block_hashes=_parse_block_hashes(action),
            include_current_request=include_current_request,
        )
    if action.action_type == RETAIN_ACTION_TYPE:
        priority = action.payload.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("kv.retain priority must be an integer")
        if priority < 0:
            raise ValueError("kv.retain priority must be non-negative")

        ttl_seconds = action.payload.get("ttl_seconds")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError("kv.retain ttl_seconds must be numeric")
        ttl_seconds = float(ttl_seconds)
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("kv.retain ttl_seconds must be positive and finite")

        return RetainBlocksAction(
            action_id=action.action_id,
            block_hashes=_parse_block_hashes(action),
            include_current_request=include_current_request,
            priority=priority,
            ttl_seconds=ttl_seconds,
        )
    return None


def _parse_block_hashes(action: KvHintAction) -> tuple[ExternalBlockHash, ...]:
    values = action.payload.get("block_hashes")
    if not isinstance(values, list):
        raise ValueError(f"{action.action_type} block_hashes must be a list")
    return tuple(_parse_block_hash(value) for value in values)


def _parse_block_hash(value: object) -> ExternalBlockHash:
    if isinstance(value, bool):
        raise ValueError("block hashes cannot be booleans")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer block hashes must be non-negative")
        return value
    if isinstance(value, bytes):
        if not value:
            raise ValueError("byte block hashes cannot be empty")
        return value
    if isinstance(value, str):
        if value.startswith("hex:"):
            try:
                block_hash = bytes.fromhex(value.removeprefix("hex:"))
            except ValueError as exc:
                raise ValueError("invalid hex block hash") from exc
            if not block_hash:
                raise ValueError("byte block hashes cannot be empty")
            return block_hash
        try:
            block_hash = int(value, 10)
        except ValueError as exc:
            raise ValueError(
                "string block hashes must be decimal integers or hex:<bytes>"
            ) from exc
        if block_hash < 0:
            raise ValueError("integer block hashes must be non-negative")
        return block_hash
    raise ValueError("block hashes must be integers, bytes, or encoded strings")
