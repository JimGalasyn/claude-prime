#!/usr/bin/env python3
# claude-prime/tools/attestation.py
"""
Memory Attestation Utilities

Provides lightweight cryptographic attestation for memory integrity.
Uses SHA256 hashing of memory content + metadata to detect tampering.

How it works:
    When storing a memory, compute a hash over its content, type, instance ID,
    session ID, timestamp, and other metadata. Store the hash as a tag
    (e.g., "attestation:abc123..."). On retrieval, recompute the hash and
    compare -- any mismatch indicates the memory was modified after storage.

This requires no database schema changes -- attestation hashes are stored
in the existing tags array.

Usage as library:
    from attestation import add_attestation_to_memory_params, verify_memory_attestation

    # Before storing
    params = add_attestation_to_memory_params(memory_params)

    # After retrieval
    result = verify_memory_attestation(memory)
    if not result["valid"]:
        print(f"Tamper detected: {result['error']}")
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


def get_session_id() -> str:
    """
    Get or generate a session ID for the current conversation.

    Uses CLAUDE_SESSION_ID env var if set, otherwise generates one based on PID + timestamp.
    In production, Claude Code should set CLAUDE_SESSION_ID per conversation.

    Returns:
        Session ID string
    """
    session_id = os.environ.get("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    # Fallback: generate session ID from PID + timestamp
    import time
    pid = os.getpid()
    ts = int(time.time())
    return f"session-{pid}-{ts}"


def compute_attestation_hash(
    content: str,
    memory_type: str,
    instance_id: str,
    session_id: str,
    created_at: datetime,
    subject: Optional[str] = None,
    source_type: Optional[str] = None,
) -> str:
    """
    Compute SHA256 attestation hash for a memory.

    Hash includes:
    - Core content (content, memory_type, subject)
    - Provenance (instance_id, session_id, source_type)
    - Timestamp (created_at)

    This ensures any modification to these fields invalidates the hash.

    Args:
        content: Memory content text
        memory_type: Type of memory (fact, decision, insight, etc.)
        instance_id: AI instance ID (e.g., "prime")
        session_id: Conversation session ID
        created_at: Creation timestamp
        subject: Optional memory subject
        source_type: Optional source type (explicit, inferred, etc.)

    Returns:
        Hex string SHA256 hash
    """
    # Normalize timestamp to ISO format
    if isinstance(created_at, datetime):
        timestamp_str = created_at.isoformat()
    else:
        timestamp_str = str(created_at)

    # Build canonical representation
    attestation_data = {
        "content": content,
        "memory_type": memory_type,
        "subject": subject or "",
        "instance_id": instance_id,
        "session_id": session_id,
        "source_type": source_type or "",
        "created_at": timestamp_str,
    }

    # Serialize with sorted keys for consistency
    canonical_json = json.dumps(attestation_data, sort_keys=True)

    # Compute SHA256
    hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_obj.hexdigest()


def format_attestation_tag(attestation_hash: str) -> str:
    """
    Format attestation hash as a tag string.

    Format: "attestation:HASH"
    This allows storing attestation in the existing tags array without schema changes.

    Args:
        attestation_hash: SHA256 hash hex string

    Returns:
        Formatted tag string
    """
    return f"attestation:{attestation_hash}"


def extract_attestation_from_tags(tags: Optional[List[str]]) -> Optional[str]:
    """
    Extract attestation hash from tags array.

    Args:
        tags: List of tag strings (may be None)

    Returns:
        Attestation hash if found, None otherwise
    """
    if not tags:
        return None

    for tag in tags:
        if tag.startswith("attestation:"):
            return tag.split(":", 1)[1]

    return None


def verify_memory_attestation(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify a memory's attestation hash.

    Args:
        memory: Memory dict with fields: content, memory_type, subject, instance_id,
                created_at, source_session_id, source_type, tags

    Returns:
        Dict with verification result:
        {
            "valid": bool,
            "has_attestation": bool,
            "stored_hash": Optional[str],
            "computed_hash": Optional[str],
            "error": Optional[str]
        }
    """
    result = {
        "valid": False,
        "has_attestation": False,
        "stored_hash": None,
        "computed_hash": None,
        "error": None
    }

    # Extract stored attestation
    stored_hash = extract_attestation_from_tags(memory.get("tags"))

    if not stored_hash:
        result["error"] = "No attestation tag found"
        return result

    result["has_attestation"] = True
    result["stored_hash"] = stored_hash

    # Validate required fields
    required_fields = ["content", "memory_type", "instance_id", "created_at"]
    missing = [f for f in required_fields if not memory.get(f)]

    if missing:
        result["error"] = f"Missing required fields: {', '.join(missing)}"
        return result

    # Get session_id (from source_session_id field)
    session_id = memory.get("source_session_id", "")
    if not session_id:
        result["error"] = "Missing source_session_id"
        return result

    # Compute expected hash
    try:
        computed_hash = compute_attestation_hash(
            content=memory["content"],
            memory_type=memory["memory_type"],
            instance_id=memory["instance_id"],
            session_id=session_id,
            created_at=memory["created_at"],
            subject=memory.get("subject"),
            source_type=memory.get("source_type")
        )
        result["computed_hash"] = computed_hash

        # Verify match
        if computed_hash == stored_hash:
            result["valid"] = True
        else:
            result["error"] = "Hash mismatch - memory may have been tampered"

    except Exception as e:
        result["error"] = f"Hash computation failed: {str(e)}"

    return result


def add_attestation_to_memory_params(
    memory_params: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add attestation to memory parameters before storing.

    This function should be called when preparing memory_set parameters.
    It computes the attestation hash and adds it to the tags array.

    Args:
        memory_params: Dict with memory fields (content, memory_type, etc.)
        session_id: Optional session ID (auto-generated if not provided)

    Returns:
        Modified memory_params dict with attestation added to tags
    """
    # Use provided session_id or generate one
    if not session_id:
        session_id = get_session_id()

    # Ensure source_session_id is set
    if "source_session_id" not in memory_params:
        memory_params["source_session_id"] = session_id

    # Get or create timestamp
    created_at = memory_params.get("created_at", datetime.now())

    # Compute attestation
    attestation_hash = compute_attestation_hash(
        content=memory_params["content"],
        memory_type=memory_params["memory_type"],
        instance_id=memory_params["instance_id"],
        session_id=session_id,
        created_at=created_at,
        subject=memory_params.get("subject"),
        source_type=memory_params.get("source_type")
    )

    # Add to tags
    tags = memory_params.get("tags", [])
    if tags is None:
        tags = []

    # Remove any existing attestation tag (in case of re-computation)
    tags = [t for t in tags if not t.startswith("attestation:")]

    # Add new attestation
    tags.append(format_attestation_tag(attestation_hash))
    memory_params["tags"] = tags

    return memory_params


# Example usage
if __name__ == "__main__":
    # Example: Create memory with attestation
    from datetime import datetime

    # Set created_at BEFORE adding attestation (important!)
    created_at = datetime.now()

    memory_params = {
        "content": "User prefers concise responses without unnecessary verbosity",
        "memory_type": "preference",
        "subject": "Communication style preference",
        "instance_id": "prime",
        "source_type": "explicit",
        "tags": ["communication", "preferences"],
        "created_at": created_at  # Include timestamp before attestation
    }

    # Add attestation
    memory_params = add_attestation_to_memory_params(memory_params)

    print("Memory parameters with attestation:")
    print(json.dumps(memory_params, indent=2, default=str))

    # Verify attestation
    result = verify_memory_attestation(memory_params)

    print("\nVerification result:")
    print(json.dumps(result, indent=2))
