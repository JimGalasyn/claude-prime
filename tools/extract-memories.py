#!/usr/bin/env python3
# claude-prime/tools/extract-memories.py
"""
Automated Memory Extraction from Conversation Transcripts

Parses Claude Code .jsonl transcripts and extracts memory-worthy content:
- User preferences and requirements
- Decisions made (with rationale)
- Problems solved
- User corrections
- Patterns discovered

Feed it a transcript file and it will identify content worth storing
as persistent memories. Useful for reducing manual memory_set calls.

Usage:
    python extract-memories.py --transcript path/to/session.jsonl
    python extract-memories.py --project-dir /path/to/project --max-sessions 5
    python extract-memories.py --transcript session.jsonl --output extracted.json
"""

import json
import re
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class MemoryExtractor:
    """Extracts memories from conversation transcripts."""

    # Extraction patterns for different memory types
    PATTERNS = {
        "preference": [
            r"I (?:prefer|like|want|need) (?:to )?(.+)",
            r"(?:please|can you) (?:use|make|ensure) (.+)",
            r"(?:always|never) (.+)",
            r"my (?:preference|requirement) is (.+)",
        ],
        "requirement": [
            r"(?:please|can you) (?:implement|add|create|build) (.+)",
            r"(?:this|the .+) (?:should|must|needs to) (.+)",
            r"requirements?: (.+)",
        ],
        "correction": [
            r"(?:no|actually|wait),? (.+)",
            r"that's (?:not right|wrong|incorrect) (?:because )?(.+)",
            r"(?:fix|change|correct) (.+)",
            r"I (?:meant|said) (.+)",
        ],
        "decision": [
            r"(?:I|we) (?:decided|chose) (?:to )?(.+)",
            r"going with (.+) because (.+)",
            r"(?:will use|using) (.+) (?:for|because) (.+)",
        ],
        "problem_solved": [
            r"(?:fixed|solved|resolved) (?:by|via|using) (.+)",
            r"(?:the )?(?:solution|fix|workaround) (?:is|was) (.+)",
            r"(?:error|bug|issue) (?:fixed|resolved) (.+)",
        ],
    }

    # Importance signals
    IMPORTANCE_SIGNALS = {
        "high": [
            "important", "critical", "must", "always", "never",
            "requirement", "breaking", "security", "bug", "error"
        ],
        "medium": [
            "should", "prefer", "recommend", "better", "improvement"
        ],
    }

    def __init__(self, transcript_path: str, project: str = "life"):
        self.transcript_path = Path(transcript_path)
        self.project = project
        self.memories: List[Dict[str, Any]] = []
        self.session_id = self.transcript_path.stem

    def load_transcript(self) -> List[Dict[str, Any]]:
        """Load .jsonl transcript into list of messages."""
        messages = []
        with open(self.transcript_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return messages

    def extract_user_messages(self, messages: List[Dict]) -> List[Tuple[str, str]]:
        """Extract user messages with timestamps."""
        user_messages = []
        for msg in messages:
            if msg.get('type') == 'user' and 'message' in msg:
                content = msg['message'].get('content', [])
                timestamp = msg.get('timestamp', '')

                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            text = block.get('text', '')
                            # Filter out system tags
                            if not text.startswith('<') or not text.endswith('>'):
                                text_parts.append(text)

                if text_parts:
                    full_text = ' '.join(text_parts).strip()
                    user_messages.append((timestamp, full_text))

        return user_messages

    def extract_assistant_decisions(self, messages: List[Dict]) -> List[Tuple[str, str, str]]:
        """Extract decisions/insights from assistant thinking + text."""
        decisions = []

        for msg in messages:
            if msg.get('type') == 'assistant' and 'message' in msg:
                content = msg['message'].get('content', [])
                timestamp = msg.get('timestamp', '')

                thinking = None
                text = None

                # Extract thinking and text blocks
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'thinking':
                            thinking = block.get('thinking', '')
                        elif block.get('type') == 'text':
                            text = block.get('text', '')

                # Look for decision markers in thinking
                if thinking:
                    # "I decided to...", "I'll use...", "The approach is..."
                    if any(marker in thinking.lower() for marker in ['decided', 'approach', 'strategy', 'will use', 'going to']):
                        # Extract the decision context
                        decisions.append((timestamp, 'decision', thinking[:500]))

        return decisions

    def assess_importance(self, text: str) -> str:
        """Assess importance based on signal words."""
        text_lower = text.lower()

        high_count = sum(1 for signal in self.IMPORTANCE_SIGNALS['high'] if signal in text_lower)
        medium_count = sum(1 for signal in self.IMPORTANCE_SIGNALS['medium'] if signal in text_lower)

        if high_count >= 2:
            return "high"
        elif high_count >= 1:
            return "medium"
        elif medium_count >= 2:
            return "medium"
        else:
            return "low"

    def match_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Match text against extraction patterns."""
        matches = []

        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract the matched content
                    captured = match.group(1) if match.groups() else match.group(0)
                    matches.append((memory_type, captured.strip()))

        return matches

    def create_memory_entry(
        self,
        memory_type: str,
        content: str,
        timestamp: str,
        importance: str = "medium",
        subject: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a memory entry dict."""
        # Generate subject from content if not provided
        if not subject:
            # Take first ~50 chars as subject
            subject = content[:50].split('\n')[0].strip()
            if len(content) > 50:
                subject += "..."

        return {
            "memory_type": memory_type,
            "content": content,
            "subject": subject,
            "project": self.project,
            "source_type": "observation",  # Extracted from conversation
            "source_session_id": self.session_id,
            "source_context": f"Auto-extracted from session {self.session_id}",
            "tags": [f"importance:{importance}", "auto-extracted"],
            "timestamp": timestamp,
            "foundational": (importance == "high"),
        }

    def extract_memories(self) -> List[Dict[str, Any]]:
        """Main extraction pipeline."""
        messages = self.load_transcript()
        user_messages = self.extract_user_messages(messages)
        assistant_decisions = self.extract_assistant_decisions(messages)

        extracted = []

        # Process user messages
        for timestamp, text in user_messages:
            # Skip very short messages
            if len(text) < 20:
                continue

            # Match against patterns
            matches = self.match_patterns(text)

            for memory_type, content in matches:
                importance = self.assess_importance(content)

                # Only extract if medium+ importance
                if importance != "low":
                    memory = self.create_memory_entry(
                        memory_type=memory_type,
                        content=f"{timestamp}: {content}",
                        timestamp=timestamp,
                        importance=importance
                    )
                    extracted.append(memory)

        # Process assistant decisions (from thinking blocks)
        for timestamp, decision_type, thinking in assistant_decisions:
            # Only extract significant decisions (>100 chars)
            if len(thinking) > 100:
                importance = self.assess_importance(thinking)

                if importance != "low":
                    memory = self.create_memory_entry(
                        memory_type="insight",
                        content=f"{timestamp}: Assistant decision/approach: {thinking}",
                        timestamp=timestamp,
                        importance=importance
                    )
                    extracted.append(memory)

        return extracted

    def deduplicate(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate memories."""
        seen = set()
        unique = []

        for mem in memories:
            # Create a dedup key from content
            key = (mem['memory_type'], mem['content'][:100])
            if key not in seen:
                seen.add(key)
                unique.append(mem)

        return unique


def extract_from_transcript(transcript_path: str, project: str = "life") -> List[Dict[str, Any]]:
    """Extract memories from a single transcript file."""
    extractor = MemoryExtractor(transcript_path, project)
    memories = extractor.extract_memories()
    memories = extractor.deduplicate(memories)
    return memories


def extract_from_recent_sessions(
    project_dir: str,
    max_sessions: int = 5,
    project: str = "life"
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract from the N most recent session transcripts."""
    transcripts_dir = Path(project_dir) / ".claude" / "projects" / f"-{project_dir.replace('/', '-')}"

    if not transcripts_dir.exists():
        return {}

    # Get all .jsonl files sorted by modification time
    transcripts = sorted(
        transcripts_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    results = {}
    for transcript in transcripts[:max_sessions]:
        memories = extract_from_transcript(str(transcript), project)
        if memories:
            results[transcript.name] = memories

    return results


def format_extraction_report(results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format extraction results for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("AUTOMATED MEMORY EXTRACTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    total = 0
    for session_id, memories in results.items():
        lines.append(f"Session: {session_id}")
        lines.append(f"Extracted: {len(memories)} memories")
        lines.append("")

        for mem in memories:
            lines.append(f"  [{mem['memory_type']}] {mem['subject']}")
            lines.append(f"  Tags: {', '.join(mem['tags'])}")
            lines.append("")

        total += len(memories)

    lines.append("=" * 60)
    lines.append(f"TOTAL: {total} memories extracted")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract memories from Claude Code conversation transcripts"
    )
    parser.add_argument(
        "--transcript",
        help="Path to specific .jsonl transcript file"
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("PROJECT_DIR", "."),
        help="Project directory (finds transcripts in .claude/projects/). Default: $PROJECT_DIR or ."
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Number of recent sessions to extract from"
    )
    parser.add_argument(
        "--project",
        default="life",
        help="Project tag for memories"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file (optional, prints to stdout if not specified)"
    )

    args = parser.parse_args()

    if args.transcript:
        # Extract from specific transcript
        memories = extract_from_transcript(args.transcript, args.project)
        results = {Path(args.transcript).name: memories}
    else:
        # Extract from recent sessions
        results = extract_from_recent_sessions(
            args.project_dir,
            args.max_sessions,
            args.project
        )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Wrote extraction results to {args.output}")
    else:
        print(format_extraction_report(results))
