"""Memory profiling utilities for debugging memory leaks."""

import gc
import sys
import tracemalloc
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch


class MemoryProfiler:
    """Comprehensive memory profiler to track object sizes and growth."""

    def __init__(self):
        self.previous_snapshot: Optional[Dict[str, int]] = None
        self.start_memory_mb = self._get_memory_mb()

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_tensor_memory_mb(self) -> Dict[str, float]:
        """Get PyTorch tensor memory usage."""
        stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"cuda:{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / 1024 / 1024
                stats[f"cuda:{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / 1024 / 1024

        # Count CPU tensors
        cpu_tensor_bytes = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == "cpu":
                    cpu_tensor_bytes += obj.element_size() * obj.nelement()
            except:
                pass
        stats["cpu_tensor_mb"] = cpu_tensor_bytes / 1024 / 1024

        return stats

    def _sizeof_fmt(self, num: float, suffix: str = "B") -> str:
        """Format bytes to human readable string."""
        for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Pi{suffix}"

    def _get_object_size(self, obj: Any) -> int:
        """Recursively calculate object size including nested structures."""
        seen_ids = set()

        def sizeof(o):
            if id(o) in seen_ids:
                return 0
            seen_ids.add(id(o))

            size = sys.getsizeof(o)

            if isinstance(o, dict):
                size += sum(sizeof(k) + sizeof(v) for k, v in o.items())
            elif hasattr(o, "__dict__"):
                size += sizeof(o.__dict__)
            elif hasattr(o, "__iter__") and not isinstance(o, (str, bytes, bytearray)):
                try:
                    size += sum(sizeof(item) for item in o)
                except:
                    pass

            return size

        return sizeof(obj)

    def get_largest_objects(self, limit: int = 20) -> List[Tuple[str, int, str]]:
        """Get the largest objects in memory, optimized for speed."""
        objects_by_type = defaultdict(lambda: {"count": 0, "total_size": 0})

        # Avoid iterating over all objects if possible
        gc.collect()
        all_objects = gc.get_objects()

        # Focus on potentially large object types first
        large_types = (list, dict, set, tuple, np.ndarray, torch.Tensor)

        for obj in all_objects:
            # Skip small objects quickly
            if not isinstance(obj, large_types) and sys.getsizeof(obj) < 1024:
                continue

            try:
                obj_type = type(obj).__name__
                obj_module = type(obj).__module__
                full_type = f"{obj_module}.{obj_type}"

                size = self._get_object_size(obj)

                if size > 1024:  # Only track objects > 1KB
                    objects_by_type[full_type]["count"] += 1
                    objects_by_type[full_type]["total_size"] += size
            except:
                pass

        # Sort by total size
        sorted_types = sorted(
            [(k, v["count"], v["total_size"]) for k, v in objects_by_type.items()], key=lambda x: x[2], reverse=True
        )

        return sorted_types[:limit]

    def get_attribute_sizes(self, obj: Any, name: str = "object") -> Dict[str, int]:
        """Get sizes of all attributes of an object."""
        sizes = {}

        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                try:
                    size = self._get_object_size(attr_value)
                    sizes[f"{name}.{attr_name}"] = size
                except:
                    sizes[f"{name}.{attr_name}"] = -1

        return sizes

    def profile_trainer(self, trainer: Any) -> Dict[str, Any]:
        """Profile memory usage of a trainer object."""
        profile = {
            "memory_mb": self._get_memory_mb(),
            "memory_growth_mb": self._get_memory_mb() - self.start_memory_mb,
            "tensor_memory": self._get_tensor_memory_mb(),
            "largest_objects": self.get_largest_objects(30),
            "trainer_attributes": self.get_attribute_sizes(trainer, "trainer"),
        }

        # Check specific trainer components
        components_to_check = [
            ("experience", "experience"),
            ("policy", "policy"),
            ("optimizer", "optimizer"),
            ("losses", "losses"),
            ("stats", "stats"),
            ("wandb_run", "wandb_run"),
            ("policy_store", "policy_store"),
            ("vecenv", "vecenv"),
            ("_curriculum", "_curriculum"),
            ("kickstarter", "kickstarter"),
            ("timer", "timer"),
        ]

        for attr_name, display_name in components_to_check:
            if hasattr(trainer, attr_name):
                obj = getattr(trainer, attr_name)
                if obj is not None:
                    profile[f"{display_name}_size"] = self._get_object_size(obj)
                    profile[f"{display_name}_attributes"] = self.get_attribute_sizes(obj, display_name)

                    # Special handling for Stopwatch
                    if attr_name == "timer" and hasattr(obj, "_timers"):
                        timer_info = {}
                        for timer_name, timer in obj._timers.items():
                            timer_info[timer_name] = {
                                "checkpoints": len(timer.checkpoints),
                                "references": len(timer.references),
                                "total_elapsed": timer.total_elapsed,
                            }
                        profile["timer_details"] = timer_info

        # Track growth since last snapshot
        current_snapshot = {k: v["total_size"] for k, v in self.get_largest_objects(100)}
        if self.previous_snapshot:
            growth = {}
            for obj_type, size in current_snapshot.items():
                prev_size = self.previous_snapshot.get(obj_type, 0)
                if size - prev_size > 1024 * 1024:  # Only show growth > 1MB
                    growth[obj_type] = size - prev_size
            profile["growth_since_last"] = sorted(growth.items(), key=lambda x: x[1], reverse=True)[:20]

        self.previous_snapshot = current_snapshot

        return profile

    def format_profile(self, profile: Dict[str, Any]) -> str:
        """Format profile data into a readable string."""
        lines = [
            "=" * 80,
            "MEMORY PROFILE",
            f"Total Memory: {profile['memory_mb']:.1f} MB (Growth: {profile['memory_growth_mb']:.1f} MB)",
            "=" * 80,
        ]

        # Tensor memory
        lines.append("\nTENSOR MEMORY:")
        for k, v in profile["tensor_memory"].items():
            lines.append(f"  {k}: {v:.1f} MB")

        # Largest objects
        lines.append("\nLARGEST OBJECTS IN MEMORY:")
        for obj_type, count, total_size in profile["largest_objects"][:20]:
            lines.append(f"  {obj_type}: {count} objects, {self._sizeof_fmt(total_size)}")

        # Component sizes
        lines.append("\nTRAINER COMPONENT SIZES:")
        for key in profile:
            if key.endswith("_size") and not key.endswith("_attributes_size"):
                component = key.replace("_size", "")
                size = profile[key]
                lines.append(f"  {component}: {self._sizeof_fmt(size)}")

        # Growth since last snapshot
        if "growth_since_last" in profile and profile["growth_since_last"]:
            lines.append("\nGROWTH SINCE LAST SNAPSHOT:")
            for obj_type, growth in profile["growth_since_last"][:10]:
                lines.append(f"  {obj_type}: +{self._sizeof_fmt(growth)}")

        # Large trainer attributes
        lines.append("\nLARGE TRAINER ATTRIBUTES (>1MB):")
        for key in profile:
            if key.endswith("_attributes"):
                component = key.replace("_attributes", "")
                attrs = profile[key]
                large_attrs = [(k, v) for k, v in attrs.items() if v > 1024 * 1024]
                if large_attrs:
                    lines.append(f"  {component}:")
                    for attr_name, size in sorted(large_attrs, key=lambda x: x[1], reverse=True)[:5]:
                        lines.append(f"    {attr_name}: {self._sizeof_fmt(size)}")

        # Timer details
        if "timer_details" in profile:
            lines.append("\nTIMER DETAILS:")
            for timer_name, details in profile["timer_details"].items():
                lines.append(
                    f"  {timer_name}: {details['checkpoints']} checkpoints, {details['references']} references"
                )

        lines.append("=" * 80)
        return "\n".join(lines)


def find_memory_leaks(num_iterations: int = 5) -> None:
    """Run garbage collection and check for leaked objects."""

    tracemalloc.start()

    # Take initial snapshot
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    # Run some iterations
    for i in range(num_iterations):
        gc.collect()

    # Take second snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    print("[ Top 10 memory allocations ]")
    for stat in top_stats[:10]:
        print(stat)

    tracemalloc.stop()


def check_circular_references() -> List[List[Any]]:
    """Find circular references that prevent garbage collection."""
    gc.collect()

    # Find all objects that have circular references
    circular_refs = []
    for obj in gc.garbage:
        referrers = gc.get_referrers(obj)
        if obj in referrers:
            circular_refs.append([obj, referrers])

    return circular_refs
