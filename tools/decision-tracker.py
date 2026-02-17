#!/usr/bin/env python3
# claude-prime/tools/decision-tracker.py
"""
Decision Chain Tracker

Track decisions with rationale, alternatives, and causal links.
Enables "why did we do this?" queries and decision chain visualization.

Each decision records:
- What was decided and why
- What alternatives were considered
- Which prior decisions enabled this one (causal chain)
- Optional checkpoint snapshots for rollback context

Decisions are stored as daily JSONL files for easy grep/review.

Configuration:
    Set DECISIONS_DIR environment variable to point to your decisions directory.
    Defaults to ./decisions if not set.

Usage:
    python decision-tracker.py track --decision "Use SQLite" --rationale "Simple, no server"
    python decision-tracker.py why "SQLite"
    python decision-tracker.py impact d-2026-02-15-001
    python decision-tracker.py list --limit 10
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque


class DecisionTracker:
    """Manages decision tracking and querying."""

    def __init__(self, decisions_dir: str = None):
        if decisions_dir is None:
            decisions_dir = os.environ.get("DECISIONS_DIR", "./decisions")

        self.decisions_dir = Path(decisions_dir)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.decisions_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def generate_decision_id(self) -> str:
        """Generate unique decision ID: d-YYYY-MM-DD-NNN"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.decisions_dir / f"{today}.jsonl"

        # Count decisions today
        count = 0
        if log_file.exists():
            with open(log_file, 'r') as f:
                count = sum(1 for _ in f)

        return f"d-{today}-{count:03d}"

    def track_decision(
        self,
        decision: str,
        rationale: str,
        type: str = "implementation",
        alternatives: List[str] = None,
        enabled_by: List[str] = None,
        checkpoint: bool = False,
        context: str = None
    ) -> str:
        """Track a decision with its rationale and context."""

        decision_id = self.generate_decision_id()
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": decision_id,
            "timestamp": timestamp,
            "type": type,
            "decision": decision,
            "rationale": rationale,
            "alternatives": alternatives or [],
            "enabled_by": enabled_by or [],
            "led_to": [],
            "context": context or "",
            "checkpoint": None,
            "outcome": None
        }

        # Create checkpoint if requested
        if checkpoint:
            checkpoint_id = self.create_checkpoint(decision_id, entry)
            entry["checkpoint"] = checkpoint_id

        # Append to today's decision log
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.decisions_dir / f"{today}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Update parent decisions
        if enabled_by:
            self.update_parent_decisions(enabled_by, decision_id)

        return decision_id

    def create_checkpoint(self, decision_id: str, decision_entry: Dict) -> str:
        """Create a checkpoint snapshot at decision point."""
        checkpoint_id = f"checkpoint-{decision_id}"
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        # Capture context snapshot
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision_entry["decision"],
            "context": self.capture_context_snapshot(),
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint_id

    def capture_context_snapshot(self) -> Dict[str, Any]:
        """Capture current state for checkpoint."""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Try to read NOW.md if it exists (common in AI agent setups)
        now_md = Path("NOW.md")
        if now_md.exists():
            with open(now_md, 'r') as f:
                context["now_md_snippet"] = f.read(500)  # First 500 chars

        # Could add: git status, file list, etc.
        return context

    def update_parent_decisions(self, parent_ids: List[str], child_id: str):
        """Add child_id to parent decisions' led_to field."""
        for parent_id in parent_ids:
            decision = self.get_decision(parent_id)
            if decision:
                if child_id not in decision.get("led_to", []):
                    decision["led_to"].append(child_id)
                    self.update_decision(decision)

    def get_decision(self, decision_id: str) -> Optional[Dict]:
        """Retrieve a decision by ID."""
        # Extract date from ID (d-YYYY-MM-DD-NNN)
        parts = decision_id.split('-')
        if len(parts) < 5:
            return None

        date = f"{parts[1]}-{parts[2]}-{parts[3]}"
        log_file = self.decisions_dir / f"{date}.jsonl"

        if not log_file.exists():
            return None

        # Search for decision in log
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['id'] == decision_id:
                    return entry

        return None

    def update_decision(self, decision: Dict):
        """Update a decision entry in its log file."""
        decision_id = decision['id']
        parts = decision_id.split('-')
        date = f"{parts[1]}-{parts[2]}-{parts[3]}"
        log_file = self.decisions_dir / f"{date}.jsonl"

        # Read all decisions
        decisions = []
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['id'] == decision_id:
                    decisions.append(decision)  # Updated version
                else:
                    decisions.append(entry)

        # Rewrite file
        with open(log_file, 'w') as f:
            for d in decisions:
                f.write(json.dumps(d) + '\n')

    def search_decisions(self, query: str) -> List[Dict]:
        """Search decisions by text match."""
        results = []

        # Search all decision log files
        for log_file in sorted(self.decisions_dir.glob("*.jsonl")):
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Search in decision text, rationale, context
                    searchable = f"{entry['decision']} {entry['rationale']} {entry.get('context', '')}"
                    if query.lower() in searchable.lower():
                        results.append(entry)

        return results

    def get_decision_chain(self, decision_id: str, direction="backward") -> List[Dict]:
        """Get decision chain (backward to root causes or forward to outcomes)."""
        chain = []
        visited = set()
        queue = deque([decision_id])

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue

            visited.add(current_id)
            decision = self.get_decision(current_id)

            if not decision:
                continue

            chain.append(decision)

            # Add parents or children depending on direction
            if direction == "backward":
                for parent_id in decision.get("enabled_by", []):
                    if parent_id not in visited:
                        queue.append(parent_id)
            else:  # forward
                for child_id in decision.get("led_to", []):
                    if child_id not in visited:
                        queue.append(child_id)

        return chain

    def format_decision_chain(self, chain: List[Dict]) -> str:
        """Format decision chain for display."""
        if not chain:
            return "No decisions found."

        lines = []
        lines.append("=" * 60)
        lines.append("DECISION CHAIN")
        lines.append("=" * 60)
        lines.append("")

        # Build tree structure
        for decision in chain:
            indent = "  " * self.get_depth(decision, chain)
            lines.append(f"{indent}[{decision['id']}] {decision['decision']}")
            lines.append(f"{indent}|- Type: {decision['type']}")
            lines.append(f"{indent}|- Rationale: {decision['rationale']}")

            if decision.get('alternatives'):
                lines.append(f"{indent}|- Alternatives: {', '.join(decision['alternatives'])}")

            if decision.get('enabled_by'):
                lines.append(f"{indent}+- Enabled by: {', '.join(decision['enabled_by'])}")

            lines.append("")

        return "\n".join(lines)

    def get_depth(self, decision: Dict, chain: List[Dict]) -> int:
        """Calculate depth of decision in chain."""
        if not decision.get('enabled_by'):
            return 0

        # Find max depth of parents + 1
        max_parent_depth = 0
        for parent_id in decision['enabled_by']:
            parent = next((d for d in chain if d['id'] == parent_id), None)
            if parent:
                parent_depth = self.get_depth(parent, chain)
                max_parent_depth = max(max_parent_depth, parent_depth)

        return max_parent_depth + 1

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Retrieve a checkpoint by ID."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, 'r') as f:
            return json.load(f)

    def format_checkpoint(self, checkpoint: Dict) -> str:
        """Format checkpoint for display."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CHECKPOINT: {checkpoint['checkpoint_id']}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Decision: {checkpoint['decision']}")
        lines.append(f"Timestamp: {checkpoint['timestamp']}")
        lines.append("")
        lines.append("Context Snapshot:")
        lines.append(json.dumps(checkpoint['context'], indent=2))

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decision chain tracker for AI agent systems")
    parser.add_argument("--decisions-dir", help="Path to decisions directory (default: $DECISIONS_DIR or ./decisions)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track a new decision")
    track_parser.add_argument("--decision", required=True, help="Decision statement")
    track_parser.add_argument("--rationale", required=True, help="Why this decision")
    track_parser.add_argument("--type", default="implementation", help="Decision type")
    track_parser.add_argument("--alternatives", help="Comma-separated alternatives considered")
    track_parser.add_argument("--enabled-by", help="Comma-separated parent decision IDs")
    track_parser.add_argument("--context", help="Additional context")
    track_parser.add_argument("--checkpoint", action="store_true", help="Create checkpoint")

    # Why command
    why_parser = subparsers.add_parser("why", help="Trace decision chain backward")
    why_parser.add_argument("query", help="Decision to find (text search)")

    # Impact command
    impact_parser = subparsers.add_parser("impact", help="Show downstream impact")
    impact_parser.add_argument("decision_id", help="Decision ID")

    # View checkpoint command
    checkpoint_parser = subparsers.add_parser("checkpoint", help="View checkpoint")
    checkpoint_parser.add_argument("checkpoint_id", help="Checkpoint ID")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent decisions")
    list_parser.add_argument("--limit", type=int, default=10, help="Number to show")

    args = parser.parse_args()

    tracker = DecisionTracker(decisions_dir=args.decisions_dir)

    if args.command == "track":
        alternatives = args.alternatives.split(',') if args.alternatives else []
        enabled_by = args.enabled_by.split(',') if args.enabled_by else []

        decision_id = tracker.track_decision(
            decision=args.decision,
            rationale=args.rationale,
            type=args.type,
            alternatives=alternatives,
            enabled_by=enabled_by,
            checkpoint=args.checkpoint,
            context=args.context
        )

        print(f"Tracked decision: {decision_id}")
        print(f"  Decision: {args.decision}")
        print(f"  Rationale: {args.rationale}")
        if args.checkpoint:
            print(f"  Checkpoint: checkpoint-{decision_id}")

    elif args.command == "why":
        decisions = tracker.search_decisions(args.query)

        if not decisions:
            print(f"No decisions found matching: {args.query}")
            return

        # Get chain for first match
        decision = decisions[0]
        chain = tracker.get_decision_chain(decision['id'], direction="backward")
        print(tracker.format_decision_chain(chain))

    elif args.command == "impact":
        decision = tracker.get_decision(args.decision_id)

        if not decision:
            print(f"Decision not found: {args.decision_id}")
            return

        chain = tracker.get_decision_chain(args.decision_id, direction="forward")
        print(tracker.format_decision_chain(chain))

    elif args.command == "checkpoint":
        checkpoint = tracker.get_checkpoint(args.checkpoint_id)

        if not checkpoint:
            print(f"Checkpoint not found: {args.checkpoint_id}")
            return

        print(tracker.format_checkpoint(checkpoint))

    elif args.command == "list":
        # Get all decisions from recent files
        all_decisions = []
        for log_file in sorted(tracker.decisions_dir.glob("*.jsonl"), reverse=True):
            with open(log_file, 'r') as f:
                for line in f:
                    all_decisions.append(json.loads(line))

            if len(all_decisions) >= args.limit:
                break

        all_decisions = all_decisions[:args.limit]

        print("=" * 60)
        print("RECENT DECISIONS")
        print("=" * 60)
        print("")

        for d in all_decisions:
            print(f"[{d['id']}] {d['decision']}")
            print(f"  Type: {d['type']}")
            print(f"  Rationale: {d['rationale'][:100]}...")
            print("")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
