#!/usr/bin/env python3
# claude-prime/tools/recency-weighted-recall.py
"""
Recency-Weighted Memory Retrieval

Implements recency decay on top of semantic memory search results.
Combines semantic similarity with temporal relevance using exponential decay.

The core idea: memories fade over time unless reinforced. A 30-day half-life
means a memory's recency score drops to 0.5 after a month. Combined with
semantic similarity, this pushes stale but keyword-matching results down
in favor of recent, relevant ones.

Presets:
    balanced       - 30-day half-life, 70/30 semantic/recency (default)
    favor_recent   - 14-day half-life, 50/50 split
    favor_semantic - 60-day half-life, 80/20 split
    recent_only    - 7-day half-life, filters out memories older than 30 days

Usage:
    python recency-weighted-recall.py "search query" --preset balanced
    python recency-weighted-recall.py "search query" --preset favor_recent --limit 20

    As a library:
        from recency_weighted_recall import apply_recency_decay
        reranked = apply_recency_decay(memories, half_life_days=30.0)
"""

import json
import subprocess
from datetime import datetime, timezone
from math import exp
from typing import List, Dict, Any, Optional


def calculate_recency_score(timestamp_str: str, half_life_days: float = 30.0) -> float:
    """
    Calculate recency score using exponential decay.

    Args:
        timestamp_str: ISO timestamp of memory creation/update
        half_life_days: Days until score decays to 0.5 (default: 30)

    Returns:
        Recency score between 0 and 1 (1 = just created, 0.5 = half-life ago)
    """
    try:
        memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)

        age_days = (current_time - memory_time).total_seconds() / 86400

        # Exponential decay: score = e^(-lambda*t) where lambda = ln(2)/half_life
        decay_constant = 0.693147 / half_life_days  # ln(2) ~ 0.693147
        recency_score = exp(-decay_constant * age_days)

        return recency_score
    except (ValueError, AttributeError):
        # If timestamp parsing fails, assume old (score = 0.5)
        return 0.5


def memory_recall_with_recency(
    query: str,
    instance_id: str = "prime",
    limit: int = 20,
    project: Optional[str] = None,
    memory_type: Optional[str] = None,
    half_life_days: float = 30.0,
    recency_weight: float = 0.3,
    semantic_weight: float = 0.7,
    min_recency_days: Optional[int] = None
) -> Dict[str, Any]:
    """
    Recall memories with recency decay applied.

    Args:
        query: Search query
        instance_id: Memory palace instance
        limit: Max results to return
        project: Filter by project
        memory_type: Filter by memory type
        half_life_days: Days for recency to decay to 0.5
        recency_weight: Weight for recency score (0-1)
        semantic_weight: Weight for semantic similarity (0-1)
        min_recency_days: If set, only return memories from last N days

    Returns:
        Dict with reranked results and metadata
    """
    # Build memory_recall MCP call
    mcp_request = {
        "instance_id": instance_id,
        "query": query,
        "limit": limit * 2,  # Get more results to rerank
        "synthesize": False,  # Get raw results for reranking
        "include_graph": True
    }

    if project:
        mcp_request["project"] = project
    if memory_type:
        mcp_request["memory_type"] = memory_type

    # Call memory-palace via MCP
    # Note: This assumes claude-code MCP integration
    # In practice, this would be called via the MCP protocol
    # For now, we'll structure it to be called from within Claude Code context

    # Placeholder for actual MCP call - in real usage this would be:
    # result = mcp_call("memory-palace", "memory_recall", mcp_request)

    # For now, return structure for integration
    return {
        "query": query,
        "recency_config": {
            "half_life_days": half_life_days,
            "recency_weight": recency_weight,
            "semantic_weight": semantic_weight
        },
        "note": "This is a wrapper - needs to be called from Claude Code context with MCP access"
    }


def apply_recency_decay(
    memories: List[Dict[str, Any]],
    half_life_days: float = 30.0,
    recency_weight: float = 0.3,
    semantic_weight: float = 0.7,
    min_recency_days: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Apply recency decay to a list of memories and rerank.

    Args:
        memories: List of memory dicts with 'created' or 'updated' timestamps
        half_life_days: Days for recency to decay to 0.5
        recency_weight: Weight for recency score (0-1)
        semantic_weight: Weight for semantic similarity (0-1)
        min_recency_days: If set, filter out memories older than N days

    Returns:
        Reranked list of memories with recency scores added
    """
    current_time = datetime.now(timezone.utc)
    results = []

    for memory in memories:
        # Get timestamp (prefer 'updated', fallback to 'created')
        timestamp = memory.get('updated') or memory.get('created')
        if not timestamp:
            continue

        # Apply recency filter if specified
        if min_recency_days:
            memory_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age_days = (current_time - memory_time).total_seconds() / 86400
            if age_days > min_recency_days:
                continue

        # Calculate recency score
        recency_score = calculate_recency_score(timestamp, half_life_days)

        # Get semantic similarity (assume it's in the memory dict)
        semantic_score = memory.get('similarity', memory.get('score', 0.5))

        # Combined score
        combined_score = (
            semantic_weight * semantic_score +
            recency_weight * recency_score
        )

        # Add scores to memory
        memory['recency_score'] = recency_score
        memory['semantic_score'] = semantic_score
        memory['combined_score'] = combined_score

        results.append(memory)

    # Sort by combined score (descending)
    results.sort(key=lambda m: m['combined_score'], reverse=True)

    return results


def format_recency_results(memories: List[Dict[str, Any]]) -> str:
    """
    Format recency-weighted results for display.

    Args:
        memories: List of reranked memories with scores

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 46)
    lines.append("RECENCY-WEIGHTED MEMORY RECALL")
    lines.append("=" * 46 + "\n")

    for i, memory in enumerate(memories, 1):
        subject = memory.get('subject', 'Untitled')
        memory_id = memory.get('id', '?')

        semantic = memory.get('semantic_score', 0)
        recency = memory.get('recency_score', 0)
        combined = memory.get('combined_score', 0)

        lines.append(f"{i}. [{memory_id}] {subject}")
        lines.append(f"   Semantic: {semantic:.3f} | Recency: {recency:.3f} | Combined: {combined:.3f}")

        timestamp = memory.get('updated') or memory.get('created', 'unknown')
        lines.append(f"   Last updated: {timestamp}")
        lines.append("")

    return "\n".join(lines)


# Example usage and configuration
RECENCY_PRESETS = {
    "balanced": {
        "half_life_days": 30.0,
        "recency_weight": 0.3,
        "semantic_weight": 0.7
    },
    "favor_recent": {
        "half_life_days": 14.0,
        "recency_weight": 0.5,
        "semantic_weight": 0.5
    },
    "favor_semantic": {
        "half_life_days": 60.0,
        "recency_weight": 0.2,
        "semantic_weight": 0.8
    },
    "recent_only": {
        "half_life_days": 7.0,
        "recency_weight": 0.6,
        "semantic_weight": 0.4,
        "min_recency_days": 30
    }
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recency-weighted memory retrieval for AI agent memory systems"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--preset",
        choices=list(RECENCY_PRESETS.keys()),
        default="balanced",
        help="Recency weighting preset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max results to return"
    )

    args = parser.parse_args()

    config = RECENCY_PRESETS[args.preset]

    print(f"Query: {args.query}")
    print(f"Preset: {args.preset}")
    print(f"Config: {config}\n")
    print("Note: This script needs to be integrated with Claude Code MCP access")
    print("to call memory-palace. Use it as a library from within Claude Code context.")
