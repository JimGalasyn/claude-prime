# Examples

Minimal, self-contained examples showing key patterns from the memory architecture.

## attested-memory.py

Demonstrates cryptographic attestation for AI agent memories. Each memory gets a SHA256 hash covering its content, metadata, and provenance. If anything is modified after storage, the hash breaks — providing tamper detection.

```
python3 attested-memory.py
```

**Why this matters:** AI agents with persistent memory are vulnerable to memory injection attacks — an adversary (or a hallucination) could alter stored memories to change the agent's beliefs and behavior. Attestation makes tampering detectable.
