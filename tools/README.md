# Tools

Memory management and analysis tools for AI agent systems. These tools work with [memory-palace](https://github.com/example/memory-palace) or similar memory backends.

## Tool Reference

| Tool | Purpose | Usage |
|------|---------|-------|
| `anomaly-detector.py` | Detect contradictions, behavioral drift, and pattern violations in memories | `python anomaly-detector.py check --text1 "..." --text2 "..."` |
| `consolidate-memories.py` | REM-sleep-inspired consolidation: replay, pattern extraction, pruning, wisdom | `python consolidate-memories.py --memories-file export.json --dry-run` |
| `recency-weighted-recall.py` | Apply exponential time-decay to rerank memory search results | `python recency-weighted-recall.py "query" --preset balanced` |
| `decision-tracker.py` | Track decisions with rationale, alternatives, and causal chains | `python decision-tracker.py track --decision "..." --rationale "..."` |
| `extract-memories.py` | Auto-extract memory-worthy content from Claude Code .jsonl transcripts | `python extract-memories.py --transcript session.jsonl` |
| `attestation.py` | Cryptographic (SHA256) tamper detection for stored memories | Library: `from attestation import add_attestation_to_memory_params` |
| `snapshot-graph.py` | Snapshot the memory-palace knowledge graph for time-lapse visualization | `python snapshot-graph.py --db-path ~/.memory-palace/memories.db` |

## Configuration

Most tools use environment variables for path configuration with sensible defaults:

| Variable | Used By | Default |
|----------|---------|---------|
| `MEMORY_DIR` | anomaly-detector, consolidate-memories | `./memory` |
| `DECISIONS_DIR` | decision-tracker | `./decisions` |
| `PROJECT_DIR` | extract-memories | `.` (current directory) |
| `MEMORY_PALACE_DB` | snapshot-graph | `~/.memory-palace/memories.db` |
| `SNAPSHOTS_DIR` | snapshot-graph | `./graph-snapshots` |
| `GRAPH_HTML` | snapshot-graph | (optional) path to HTML template |
| `CLAUDE_SESSION_ID` | attestation | auto-generated if not set |

All tools also accept CLI arguments to override these defaults. Run any tool with `--help` for details.

## Dependencies

These tools use only Python 3 standard library modules (json, hashlib, sqlite3, re, etc.). No pip install required.

## Typical Workflow

1. **During sessions**: Use `attestation.py` to sign memories before storing them
2. **After sessions**: Run `extract-memories.py` on conversation transcripts to find unstored memories
3. **Weekly**: Run `consolidate-memories.py` on a memory export to find patterns and prune stale entries
4. **On demand**: Use `anomaly-detector.py` to check for contradictions or drift
5. **Daily**: Run `snapshot-graph.py` to track knowledge graph growth over time
6. **For audits**: Use `decision-tracker.py` to trace why decisions were made
