#!/usr/bin/env python3
# claude-prime/tools/consolidate-memories.py
"""
REM Memory Consolidation

Automated memory consolidation inspired by biological REM sleep cycles.
Processes accumulated memories through five phases:

1. Experience Replay - Score memories by importance signals
2. Pattern Extraction - Find repeated problem-solution pairs, decision heuristics
3. Memory Integration - Suggest new links between related memories
4. Memory Pruning - Identify low-value memories for archival
5. Wisdom Extraction - Convert high-confidence patterns into distilled insights

Designed for use with memory-palace or similar AI memory systems.
Feed it a JSON export of memories and it will analyze patterns and suggest actions.

Configuration:
    Set MEMORY_DIR environment variable to point to your memory directory.
    Defaults to ./memory if not set.

Usage:
    python consolidate-memories.py --memories-file memories.json --dry-run
    python consolidate-memories.py --memories-file memories.json --force
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter


class MemoryConsolidator:
    """Consolidates memories like REM sleep."""

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.environ.get("MEMORY_DIR", "./memory")

        self.memory_dir = Path(memory_dir)
        self.consolidation_dir = self.memory_dir / "consolidation"
        self.consolidation_dir.mkdir(exist_ok=True)

        self.last_run_file = self.consolidation_dir / "last_run.txt"
        self.last_run = self.load_last_run_time()

    def load_last_run_time(self) -> datetime:
        """Load timestamp of last consolidation run."""
        if self.last_run_file.exists():
            with open(self.last_run_file, 'r') as f:
                timestamp = f.read().strip()
                return datetime.fromisoformat(timestamp)
        else:
            # Default to 7 days ago
            return datetime.now(timezone.utc) - timedelta(days=7)

    def save_last_run_time(self):
        """Save current timestamp as last run time."""
        with open(self.last_run_file, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())

    def should_run(self, force: bool = False) -> bool:
        """Check if consolidation should run."""
        if force:
            return True

        days_since_last = (datetime.now(timezone.utc) - self.last_run).days

        # Run weekly
        if days_since_last >= 7:
            return True

        # Or if many new memories accumulated
        # (Would need memory-palace access to check this)

        return False

    # ==========================================
    # Phase 1: Experience Replay
    # ==========================================

    def replay_experiences(self, memories: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Score recent memories by importance.

        Importance signals:
        - Access count (frequently retrieved)
        - Foundational flag
        - Recent activity
        - Keywords indicating importance
        """

        scored = []

        for memory in memories:
            score = self.calculate_importance_score(memory)
            scored.append((memory, score))

        # Sort by importance
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def calculate_importance_score(self, memory: Dict) -> float:
        """Calculate memory importance based on multiple signals."""
        score = 0.0

        # Access count (how often retrieved)
        access_count = memory.get('access_count', 0)
        score += access_count * 2.0

        # Foundational memories are important
        if memory.get('foundational', False):
            score += 10.0

        # Recent access boosts score
        last_accessed = memory.get('last_accessed_at')
        if last_accessed:
            try:
                accessed_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                days_since = (datetime.now(timezone.utc) - accessed_time).days
                recency_boost = max(0, 7 - days_since)
                score += recency_boost
            except:
                pass

        # Importance tags
        tags = memory.get('tags', [])
        if 'importance:high' in tags:
            score += 5.0
        elif 'importance:medium' in tags:
            score += 2.0

        # Memory type weighting
        mem_type = memory.get('memory_type', '')
        if mem_type in ['architecture', 'decision', 'insight']:
            score += 3.0
        elif mem_type in ['solution', 'gotcha']:
            score += 2.0

        return score

    # ==========================================
    # Phase 2: Pattern Extraction
    # ==========================================

    def extract_patterns(self, memories: List[Dict]) -> List[Dict]:
        """
        Extract patterns from similar experiences.

        Pattern types:
        - Problem-solution pairs (repeated fixes)
        - Decision heuristics (repeated choices)
        - Topic clusters (related concepts)
        """

        patterns = []

        # Group by memory type
        by_type = defaultdict(list)
        for m in memories:
            by_type[m.get('memory_type', 'unknown')].append(m)

        # Extract solution patterns
        if 'solution' in by_type:
            solution_patterns = self.extract_solution_patterns(by_type['solution'])
            patterns.extend(solution_patterns)

        # Extract decision patterns
        if 'decision' in by_type:
            decision_patterns = self.extract_decision_patterns(by_type['decision'])
            patterns.extend(decision_patterns)

        # Extract insight clusters
        if 'insight' in by_type:
            insight_clusters = self.cluster_insights(by_type['insight'])
            patterns.extend(insight_clusters)

        return patterns

    def extract_solution_patterns(self, solutions: List[Dict]) -> List[Dict]:
        """Find repeated problem-solution patterns."""
        patterns = []

        # Group by keywords in content
        keyword_groups = defaultdict(list)

        for solution in solutions:
            content = solution.get('content', '')
            keywords = self.extract_keywords(content)

            # Group by common keywords
            key = tuple(sorted(keywords[:3]))  # Top 3 keywords
            keyword_groups[key].append(solution)

        # Find patterns (3+ similar solutions)
        for keywords, group in keyword_groups.items():
            if len(group) >= 3:
                patterns.append({
                    "type": "problem_solution",
                    "keywords": list(keywords),
                    "instances": len(group),
                    "memory_ids": [m['id'] for m in group],
                    "confidence": min(1.0, len(group) / 10.0),
                    "observation": self.generalize_solutions(group)
                })

        return patterns

    def extract_decision_patterns(self, decisions: List[Dict]) -> List[Dict]:
        """Find repeated decision patterns."""
        patterns = []

        # Look for common decision types or contexts
        by_context = defaultdict(list)

        for decision in decisions:
            # Extract context keywords
            content = decision.get('content', '')
            context_keywords = self.extract_keywords(content)

            for keyword in context_keywords[:2]:  # Top 2 keywords
                by_context[keyword].append(decision)

        # Find patterns
        for context, group in by_context.items():
            if len(group) >= 3:
                patterns.append({
                    "type": "decision_heuristic",
                    "context": context,
                    "instances": len(group),
                    "memory_ids": [m['id'] for m in group],
                    "confidence": min(1.0, len(group) / 8.0),
                    "observation": f"Decision pattern related to {context}"
                })

        return patterns

    def cluster_insights(self, insights: List[Dict]) -> List[Dict]:
        """Cluster related insights."""
        # Simplified: group by common keywords
        clusters = []

        keyword_groups = defaultdict(list)
        for insight in insights:
            keywords = self.extract_keywords(insight.get('content', ''))
            for kw in keywords[:2]:
                keyword_groups[kw].append(insight)

        for keyword, group in keyword_groups.items():
            if len(group) >= 2:
                clusters.append({
                    "type": "insight_cluster",
                    "theme": keyword,
                    "instances": len(group),
                    "memory_ids": [m['id'] for m in group],
                    "confidence": min(1.0, len(group) / 5.0)
                })

        return clusters

    def generalize_solutions(self, solutions: List[Dict]) -> str:
        """Generate generalized observation from multiple solutions."""
        # Extract common words/phrases
        all_words = []
        for solution in solutions:
            content = solution.get('content', '')
            words = self.extract_keywords(content)
            all_words.extend(words)

        # Find most common words
        common = Counter(all_words).most_common(3)
        keywords = [w for w, count in common]

        return f"Repeated pattern involving {', '.join(keywords)}"

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}

        # Lowercase and tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return keywords[:10]  # Top 10 keywords

    # ==========================================
    # Phase 3: Memory Integration
    # ==========================================

    def suggest_links(self, memories: List[Dict]) -> List[Dict]:
        """
        Suggest new links between memories.

        This would normally use semantic similarity from memory-palace,
        but we'll use keyword overlap as a proxy.
        """

        suggested_links = []

        # For each memory, find candidates for linking
        for i, mem1 in enumerate(memories):
            keywords1 = set(self.extract_keywords(mem1.get('content', '')))

            for mem2 in memories[i+1:]:
                keywords2 = set(self.extract_keywords(mem2.get('content', '')))

                # Calculate keyword overlap
                overlap = keywords1 & keywords2
                overlap_score = len(overlap) / max(len(keywords1), len(keywords2), 1)

                if overlap_score > 0.5 and len(overlap) >= 2:
                    suggested_links.append({
                        "source_id": mem1['id'],
                        "target_id": mem2['id'],
                        "relation_type": "relates_to",
                        "strength": overlap_score,
                        "reason": f"Common keywords: {', '.join(list(overlap)[:3])}"
                    })

        return suggested_links

    # ==========================================
    # Phase 4: Memory Pruning
    # ==========================================

    def identify_pruning_candidates(self, memories: List[Dict]) -> Dict[str, List[int]]:
        """Identify memories to archive or delete."""
        to_archive = []

        for memory in memories:
            # Never prune foundational
            if memory.get('foundational', False):
                continue

            # Low access + old = candidate
            age_days = self.get_age_days(memory)
            access_count = memory.get('access_count', 0)

            if age_days > 90 and access_count < 2:
                to_archive.append(memory['id'])

        return {
            "to_archive": to_archive,
            "to_delete": []  # Conservative: don't auto-delete
        }

    def get_age_days(self, memory: Dict) -> int:
        """Calculate memory age in days."""
        created = memory.get('created_at') or memory.get('created')
        if not created:
            return 0

        try:
            created_time = datetime.fromisoformat(created.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - created_time).days
            return age
        except:
            return 0

    # ==========================================
    # Phase 5: Wisdom Extraction
    # ==========================================

    def extract_wisdom(self, patterns: List[Dict], confidence_threshold: float = 0.75) -> List[Dict]:
        """Convert high-confidence patterns into wisdom entries."""
        wisdom_entries = []

        for pattern in patterns:
            confidence = pattern.get('confidence', 0)

            if confidence < confidence_threshold:
                continue

            if pattern['instances'] < 3:
                continue

            # Generate wisdom entry
            wisdom = self.generate_wisdom_entry(pattern)
            wisdom_entries.append(wisdom)

        return wisdom_entries

    def generate_wisdom_entry(self, pattern: Dict) -> Dict:
        """Format pattern as wisdom.md entry."""
        confidence_label = "high" if pattern['confidence'] > 0.85 else "medium"

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pattern_name": pattern.get('context', pattern.get('theme', 'Unnamed pattern')),
            "observation": pattern.get('observation', 'Pattern observed across multiple instances'),
            "evidence": f"{pattern['instances']} instances (Memory IDs: {', '.join(map(str, pattern['memory_ids'][:3]))})",
            "confidence": confidence_label,
            "reasoning": f"Pattern detected via consolidation with {pattern['instances']} supporting instances",
            "hypothesis": f"This pattern suggests a consistent approach for {pattern.get('type', 'this scenario')}"
        }

    # ==========================================
    # Main Consolidation Pipeline
    # ==========================================

    def run_consolidation(self, memories: List[Dict], dry_run: bool = False) -> Dict:
        """Execute full consolidation pipeline."""

        print("=" * 60)
        print("REM MEMORY CONSOLIDATION")
        print("=" * 60)
        print()

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "phases": {}
        }

        # Phase 1: Replay
        print("Phase 1: Replaying recent experiences...")
        scored_memories = self.replay_experiences(memories)
        important = [m for m, score in scored_memories if score > 5.0]
        print(f"  Reviewed {len(memories)} memories")
        print(f"  Identified {len(important)} important memories (score > 5.0)")

        results['phases']['replay'] = {
            "total_reviewed": len(memories),
            "important_identified": len(important),
            "top_scores": [score for _, score in scored_memories[:5]]
        }

        # Phase 2: Extract patterns
        print()
        print("Phase 2: Extracting patterns...")
        patterns = self.extract_patterns(memories)
        print(f"  Extracted {len(patterns)} patterns")

        by_type = Counter(p['type'] for p in patterns)
        for ptype, count in by_type.items():
            print(f"    - {ptype}: {count}")

        results['phases']['pattern_extraction'] = {
            "patterns_found": len(patterns),
            "by_type": dict(by_type)
        }

        # Phase 3: Integration
        print()
        print("Phase 3: Suggesting new connections...")
        suggested_links = self.suggest_links(important)
        print(f"  Suggested {len(suggested_links)} new links")

        results['phases']['integration'] = {
            "links_suggested": len(suggested_links)
        }

        # Phase 4: Pruning
        print()
        print("Phase 4: Identifying pruning candidates...")
        pruning = self.identify_pruning_candidates(memories)
        print(f"  Marked {len(pruning['to_archive'])} for archival")

        results['phases']['pruning'] = {
            "to_archive": len(pruning['to_archive']),
            "to_delete": len(pruning['to_delete'])
        }

        # Phase 5: Wisdom extraction
        print()
        print("Phase 5: Extracting wisdom...")
        wisdom = self.extract_wisdom(patterns, confidence_threshold=0.75)
        print(f"  Generated {len(wisdom)} wisdom entries")

        results['phases']['wisdom'] = {
            "entries_created": len(wisdom)
        }

        print()
        print("=" * 60)
        print("CONSOLIDATION COMPLETE")
        print("=" * 60)

        # Save results
        if not dry_run:
            self.save_consolidation_report(results, patterns, suggested_links, wisdom, pruning)
            self.save_last_run_time()
            print()
            print(f"Reports saved to: {self.consolidation_dir}/")

        return results

    def save_consolidation_report(
        self,
        results: Dict,
        patterns: List[Dict],
        links: List[Dict],
        wisdom: List[Dict],
        pruning: Dict
    ):
        """Save consolidation results to files."""

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")

        # Main report
        report_file = self.consolidation_dir / f"{timestamp}-report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Patterns
        if patterns:
            patterns_file = self.consolidation_dir / f"{timestamp}-patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)

        # Suggested links
        if links:
            links_file = self.consolidation_dir / f"{timestamp}-links.json"
            with open(links_file, 'w') as f:
                json.dump(links, f, indent=2)

        # Wisdom entries
        if wisdom:
            wisdom_file = self.consolidation_dir / f"{timestamp}-wisdom.json"
            with open(wisdom_file, 'w') as f:
                json.dump(wisdom, f, indent=2)

        # Pruning candidates
        if pruning['to_archive']:
            pruning_file = self.consolidation_dir / f"{timestamp}-pruning.json"
            with open(pruning_file, 'w') as f:
                json.dump(pruning, f, indent=2)


def load_sample_memories() -> List[Dict]:
    """
    Load memories for consolidation.

    In real usage, this would call memory_palace MCP to get recent memories.
    For now, return sample structure.
    """

    # This is a placeholder - in real usage would be:
    # memories = memory_recall(query="", limit=200, synthesize=False)

    return [
        {
            "id": 1,
            "subject": "Example memory",
            "content": "This is sample content",
            "memory_type": "fact",
            "created_at": "2026-02-08T10:00:00Z",
            "access_count": 5,
            "foundational": False,
            "tags": ["importance:medium"]
        }
        # ... more memories
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="REM memory consolidation for AI agent memories")
    parser.add_argument("--dry-run", action="store_true", help="Don't save results")
    parser.add_argument("--force", action="store_true", help="Run even if not scheduled")
    parser.add_argument("--memories-file", help="JSON file with memories to consolidate")
    parser.add_argument("--memory-dir", help="Path to memory directory (default: $MEMORY_DIR or ./memory)")

    args = parser.parse_args()

    consolidator = MemoryConsolidator(memory_dir=args.memory_dir)

    # Check if should run
    if not consolidator.should_run(force=args.force):
        print("Consolidation not due yet. Use --force to run anyway.")
        return

    # Load memories
    if args.memories_file:
        with open(args.memories_file, 'r') as f:
            memories = json.load(f)
    else:
        print("Note: Using sample memories. In real usage, pass --memories-file")
        print("      with JSON export from memory-palace.")
        print()
        memories = load_sample_memories()

    # Run consolidation
    results = consolidator.run_consolidation(memories, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
