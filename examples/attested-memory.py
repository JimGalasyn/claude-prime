#!/usr/bin/env python3
"""
Example: Storing a memory with cryptographic attestation.

Memory attestation adds a SHA256 hash to each memory, covering its content,
type, creator, session, and timestamp. If any of these fields are modified
after storage, the hash won't match — providing tamper detection.

This is useful for AI agents that maintain persistent memory across sessions.
Without attestation, a compromised database or prompt injection could silently
alter stored memories, changing the agent's beliefs and behavior.

Usage:
    python3 attested-memory.py
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


def compute_attestation_hash(
    content: str,
    memory_type: str,
    instance_id: str,
    session_id: str,
    created_at: datetime,
    subject: Optional[str] = None,
    source_type: Optional[str] = None,
) -> str:
    """Compute SHA256 hash over memory content + metadata."""
    if isinstance(created_at, datetime):
        timestamp_str = created_at.isoformat()
    else:
        timestamp_str = str(created_at)

    attestation_data = {
        "content": content,
        "memory_type": memory_type,
        "subject": subject or "",
        "instance_id": instance_id,
        "session_id": session_id,
        "source_type": source_type or "",
        "created_at": timestamp_str,
    }

    canonical_json = json.dumps(attestation_data, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_attestation(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a memory's attestation hash matches its content."""
    tags = memory.get("tags", [])
    stored_hash = None
    for tag in tags:
        if tag.startswith("attestation:"):
            stored_hash = tag.split(":", 1)[1]
            break

    if not stored_hash:
        return {"valid": False, "error": "No attestation found"}

    computed_hash = compute_attestation_hash(
        content=memory["content"],
        memory_type=memory["memory_type"],
        instance_id=memory["instance_id"],
        session_id=memory["session_id"],
        created_at=memory["created_at"],
        subject=memory.get("subject"),
        source_type=memory.get("source_type"),
    )

    if computed_hash == stored_hash:
        return {"valid": True, "hash": computed_hash}
    else:
        return {
            "valid": False,
            "error": "HASH MISMATCH — memory may have been tampered",
            "stored": stored_hash,
            "computed": computed_hash,
        }


# --- Demo ---

if __name__ == "__main__":
    print("=== Memory Attestation Demo ===\n")

    # 1. Create a memory
    created_at = datetime.now()
    memory = {
        "content": "User prefers MIT license for all public repositories",
        "memory_type": "preference",
        "subject": "License preference",
        "instance_id": "prime",
        "session_id": f"session-{os.getpid()}-demo",
        "source_type": "explicit",
        "created_at": created_at,
        "tags": ["preferences", "github"],
    }

    # 2. Compute and attach attestation
    attestation_hash = compute_attestation_hash(
        content=memory["content"],
        memory_type=memory["memory_type"],
        instance_id=memory["instance_id"],
        session_id=memory["session_id"],
        created_at=memory["created_at"],
        subject=memory["subject"],
        source_type=memory["source_type"],
    )
    memory["tags"].append(f"attestation:{attestation_hash}")

    print(f"Memory: {memory['subject']}")
    print(f"Content: {memory['content']}")
    print(f"Attestation: {attestation_hash[:16]}...")
    print()

    # 3. Verify — should pass
    result = verify_attestation(memory)
    print(f"Verification (original): {'PASS' if result['valid'] else 'FAIL'}")

    # 4. Tamper with the memory
    tampered = dict(memory)
    tampered["content"] = "User prefers GPL license for all public repositories"

    result = verify_attestation(tampered)
    print(f"Verification (tampered): {'PASS' if result['valid'] else 'FAIL'}")
    if not result["valid"]:
        print(f"  Error: {result['error']}")

    print("\n=== Key Insight ===")
    print("The attestation hash covers content + metadata + timestamp.")
    print("Any modification — even a single character — invalidates the hash.")
    print("This protects against silent memory corruption or injection attacks.")
