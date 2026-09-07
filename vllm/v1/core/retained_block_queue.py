# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import itertools
import time
from collections.abc import Hashable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(frozen=True, slots=True)
class RetentionLease:
    priority: int
    expires_at: float


class RetainedBlockQueue:
    """Eviction queue for free blocks with one or more retention leases."""

    def __init__(self) -> None:
        self._leases: dict[int, dict[Hashable, RetentionLease]] = {}
        self._free_blocks: dict[int, tuple[KVCacheBlock, int]] = {}
        self._versions: dict[int, int] = {}
        self._priority_heap: list[tuple[int, int, int, int, KVCacheBlock]] = []
        self._expiry_heap: list[tuple[float, int, int, Hashable, RetentionLease]] = []
        self._counter = itertools.count()

    def retain(
        self,
        block: KVCacheBlock,
        lease_id: Hashable,
        priority: int,
        ttl_seconds: float,
    ) -> None:
        now = time.monotonic()
        lease = RetentionLease(priority, now + ttl_seconds)
        self._leases.setdefault(block.block_id, {})[lease_id] = lease
        heapq.heappush(
            self._expiry_heap,
            (lease.expires_at, next(self._counter), block.block_id, lease_id, lease),
        )
        if block.ref_cnt == 0:
            if block.block_id not in self._free_blocks:
                self._free_blocks[block.block_id] = (block, next(self._counter))
            self._refresh_free_block(block)

    def add_free_block(self, block: KVCacheBlock) -> bool:
        """Queue a newly free block if it has a live retention lease."""
        if not self._leases.get(block.block_id):
            return False
        assert block.block_id not in self._free_blocks
        self._free_blocks[block.block_id] = (block, next(self._counter))
        self._refresh_free_block(block)
        return True

    def remove_free_block(self, block: KVCacheBlock) -> bool:
        """Remove a free retained block before it becomes active."""
        if self._free_blocks.pop(block.block_id, None) is None:
            return False
        self._versions[block.block_id] = self._versions.get(block.block_id, 0) + 1
        return True

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop the lowest-priority retained block, using LRU for ties."""
        while self._priority_heap:
            priority, _, block_id, version, block = heapq.heappop(self._priority_heap)
            free_entry = self._free_blocks.get(block_id)
            if free_entry is None or self._versions.get(block_id) != version:
                continue
            if priority != self._effective_priority(block_id):
                self._refresh_free_block(block)
                continue
            del self._free_blocks[block_id]
            self._versions[block_id] = version + 1
            return block
        return None

    def expire(self) -> list[KVCacheBlock]:
        """Expire elapsed leases and return blocks that rejoin ordinary LRU."""
        return self._expire(time.monotonic())

    def clear_block(self, block: KVCacheBlock) -> bool:
        """Clear all leases and return whether the block was free here."""
        self._leases.pop(block.block_id, None)
        was_free = self._free_blocks.pop(block.block_id, None) is not None
        self._versions[block.block_id] = self._versions.get(block.block_id, 0) + 1
        return was_free

    def move_leases(self, src: KVCacheBlock, dst: KVCacheBlock) -> None:
        """Move retention with prefix-cache metadata during copy-on-write."""
        assert src.block_id not in self._free_blocks
        assert dst.block_id not in self._free_blocks
        leases = self._leases.pop(src.block_id, None)
        if not leases:
            return
        dst_leases = self._leases.setdefault(dst.block_id, {})
        for lease_id, lease in leases.items():
            current = dst_leases.get(lease_id)
            if current is None or (lease.priority, lease.expires_at) > (
                current.priority,
                current.expires_at,
            ):
                dst_leases[lease_id] = lease
                heapq.heappush(
                    self._expiry_heap,
                    (
                        lease.expires_at,
                        next(self._counter),
                        dst.block_id,
                        lease_id,
                        lease,
                    ),
                )

    def drain(self) -> list[KVCacheBlock]:
        """Remove all retention state and return retained free blocks in LRU order."""
        blocks = [entry for entry in self._free_blocks.values()]
        blocks.sort(key=lambda entry: entry[1])
        free_blocks = [block for block, _ in blocks]
        self.__init__()
        return free_blocks

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._free_blocks

    def __len__(self) -> int:
        return len(self._free_blocks)

    def _effective_priority(self, block_id: int) -> int:
        leases = self._leases.get(block_id)
        if not leases:
            return 0
        return max(lease.priority for lease in leases.values())

    def _refresh_free_block(self, block: KVCacheBlock) -> None:
        _, free_order = self._free_blocks[block.block_id]
        version = self._versions.get(block.block_id, 0) + 1
        self._versions[block.block_id] = version
        heapq.heappush(
            self._priority_heap,
            (
                self._effective_priority(block.block_id),
                free_order,
                block.block_id,
                version,
                block,
            ),
        )

    def _expire(self, now: float) -> list[KVCacheBlock]:
        changed_blocks: set[int] = set()
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            _, _, block_id, lease_id, lease = heapq.heappop(self._expiry_heap)
            leases = self._leases.get(block_id)
            if leases is None or leases.get(lease_id) != lease:
                continue
            del leases[lease_id]
            if not leases:
                del self._leases[block_id]
            changed_blocks.add(block_id)

        released_blocks: list[tuple[KVCacheBlock, int]] = []
        for block_id in changed_blocks:
            free_entry = self._free_blocks.get(block_id)
            if free_entry is not None:
                if self._leases.get(block_id):
                    self._refresh_free_block(free_entry[0])
                else:
                    del self._free_blocks[block_id]
                    self._versions[block_id] = self._versions.get(block_id, 0) + 1
                    released_blocks.append(free_entry)
        released_blocks.sort(key=lambda entry: entry[1])
        return [block for block, _ in released_blocks]
