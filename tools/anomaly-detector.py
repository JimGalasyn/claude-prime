#!/usr/bin/env python3
# claude-prime/tools/anomaly-detector.py
"""
Semantic Anomaly Detector

Detects contradictions, drift, and pattern violations in AI agent memory systems.
Uses keyword-based semantic similarity and pattern analysis to find:
- Contradictions between memories (negation detection)
- Behavioral drift from identity principles
- Violations of stated preferences and directives

Configuration:
    Set MEMORY_DIR environment variable to point to your memory directory.
    Defaults to ./memory if not set.

Usage:
    python anomaly-detector.py check --text1 "always use X" --text2 "never use X"
    python anomaly-detector.py report --days 7
    python anomaly-detector.py drift --action "skipped checkpoint on architectural decision"
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter


class AnomalyDetector:
    """Detects semantic anomalies in memories, decisions, and behavior."""

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.environ.get("MEMORY_DIR", "./memory")

        self.memory_dir = Path(memory_dir)
        self.anomaly_log = self.memory_dir / "anomalies.jsonl"
        self.config = self.load_config()

        # Cache identity principles
        self.identity_principles = self.load_identity_principles()

    def load_config(self) -> Dict:
        """Load anomaly detection configuration."""
        default_config = {
            "enabled": True,
            "thresholds": {
                "contradiction_keywords": 0.5,  # Keyword overlap threshold
                "drift_threshold": 0.6,
                "pattern_confidence": 0.75,
                "min_pattern_instances": 3
            },
            "severity_actions": {
                "high": "block",
                "medium": "flag",
                "low": "log"
            },
            "checks": {
                "memory_contradictions": True,
                "behavioral_drift": True,
                "pattern_violations": True,
                "semantic_outliers": False
            }
        }

        config_file = self.memory_dir.parent / "config/anomaly-detection.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def load_identity_principles(self) -> List[Dict]:
        """Extract principles from identity files."""
        principles = []

        # Look for identity files relative to the memory directory
        identity_files = [
            self.memory_dir.parent / "CLAUDE.md",
            self.memory_dir.parent / "MEMORY.md",
            self.memory_dir / "autonomous-protocol.md"
        ]

        for file in identity_files:
            if not file.exists():
                continue

            with open(file, 'r') as f:
                content = f.read()

            # Extract principle statements
            patterns = [
                (r"(?:I|We) (?:should|must|will|always|never) (.+)", "directive"),
                (r"(?:Prefer|Avoid|Use|Don't|Do not) (.+)", "preference"),
                (r"^\s*-\s*\*\*(.+?)\*\*", "principle"),  # Markdown bold
            ]

            for pattern, ptype in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    text = match.strip()
                    if len(text) > 10 and len(text) < 200:  # Reasonable length
                        principles.append({
                            "text": text,
                            "type": ptype,
                            "source": file.name,
                            "keywords": self.extract_keywords(text)
                        })

        return principles

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Lowercase and tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return list(set(keywords))  # Deduplicate

    def check_contradiction(self, text1: str, text2: str) -> Optional[Dict]:
        """
        Check if two texts contradict each other.

        Uses keyword analysis + negation detection.
        """

        # Extract keywords from both texts
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))

        # Check keyword overlap (must be talking about similar topics)
        overlap = keywords1 & keywords2
        overlap_score = len(overlap) / max(len(keywords1), len(keywords2), 1)

        if overlap_score < self.config['thresholds']['contradiction_keywords']:
            return None  # Not about same topic

        # Check for negation patterns
        negation_pairs = [
            ("always", "never"),
            ("must", "must not"),
            ("must", "mustn't"),
            ("should", "should not"),
            ("should", "shouldn't"),
            ("prefer", "avoid"),
            ("use", "don't use"),
            ("use", "do not use"),
            ("required", "forbidden"),
            ("allowed", "prohibited"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        for pos, neg in negation_pairs:
            if pos in text1_lower and neg in text2_lower:
                return {
                    "type": "contradiction",
                    "severity": "high",
                    "text1": text1,
                    "text2": text2,
                    "pattern": f"{pos} vs {neg}",
                    "overlap_score": overlap_score,
                    "overlapping_keywords": list(overlap)
                }

            # Check reverse
            if neg in text1_lower and pos in text2_lower:
                return {
                    "type": "contradiction",
                    "severity": "high",
                    "text1": text1,
                    "text2": text2,
                    "pattern": f"{neg} vs {pos}",
                    "overlap_score": overlap_score,
                    "overlapping_keywords": list(overlap)
                }

        return None

    def check_behavioral_drift(self, action: str, context: str = "") -> List[Dict]:
        """Check if action drifts from identity principles."""

        violations = []

        action_keywords = set(self.extract_keywords(action))

        for principle in self.identity_principles:
            principle_keywords = set(principle['keywords'])

            # Check keyword overlap
            overlap = action_keywords & principle_keywords
            overlap_score = len(overlap) / max(len(action_keywords), len(principle_keywords), 1)

            if overlap_score > self.config['thresholds']['drift_threshold']:
                # Check if action violates principle
                violation = self.check_principle_violation(action, principle)
                if violation:
                    violations.append({
                        "type": "drift",
                        "severity": "medium",
                        "action": action,
                        "principle": principle['text'],
                        "source": principle['source'],
                        "overlap_score": overlap_score,
                        "violation_type": violation
                    })

        return violations

    def check_principle_violation(self, action: str, principle: Dict) -> Optional[str]:
        """Check if action violates a specific principle."""

        action_lower = action.lower()
        principle_lower = principle['text'].lower()

        # Check "never X" violations
        if "never" in principle_lower:
            # Extract what should never be done
            match = re.search(r'never (.+)', principle_lower)
            if match:
                forbidden = match.group(1).strip()
                if forbidden in action_lower:
                    return f"never_violation: {forbidden}"

        # Check "always X" violations
        if "always" in principle_lower:
            # Extract what should always be done
            match = re.search(r'always (.+)', principle_lower)
            if match:
                required = match.group(1).strip()
                # Check if action is related but missing required behavior
                # (harder to detect, would need semantic understanding)
                pass

        # Check "prefer X" violations
        if "prefer" in principle_lower and "avoid" in action_lower:
            return "preference_violation"

        return None

    def learn_decision_patterns(self, decisions: List[Dict]) -> List[Dict]:
        """
        Extract patterns from decision history.

        Looks for:
        - Repeated choices (always choose X over Y)
        - Conditional patterns (when context C, choose X)
        """

        patterns = []

        if len(decisions) < self.config['thresholds']['min_pattern_instances']:
            return patterns

        # Group by decision type
        by_type = defaultdict(list)
        for d in decisions:
            by_type[d['type']].append(d)

        # Pattern: "Always checkpoint architectural decisions"
        if 'architectural' in by_type:
            arch_decisions = by_type['architectural']
            checkpointed = [d for d in arch_decisions if d.get('checkpoint')]
            checkpoint_rate = len(checkpointed) / len(arch_decisions)

            if checkpoint_rate > 0.8 and len(arch_decisions) >= 3:
                patterns.append({
                    "pattern": "Always checkpoint architectural decisions",
                    "confidence": checkpoint_rate,
                    "applies_to": lambda d: d['type'] == 'architectural',
                    "check": lambda d: d.get('checkpoint') is not None,
                    "evidence": [d['id'] for d in checkpointed]
                })

        # Pattern: Alternative preferences
        # Look for repeated rejection of same alternatives
        alternative_counts = defaultdict(lambda: {"rejected": 0, "chosen": 0})

        for d in decisions:
            decision_text = d['decision'].lower()
            for alt in d.get('alternatives', []):
                alt_lower = alt.lower()
                if alt_lower in decision_text:
                    alternative_counts[alt]["chosen"] += 1
                else:
                    alternative_counts[alt]["rejected"] += 1

        for alt, counts in alternative_counts.items():
            total = counts["rejected"] + counts["chosen"]
            if total >= 3:
                reject_rate = counts["rejected"] / total
                if reject_rate > 0.8:
                    patterns.append({
                        "pattern": f"Prefer not to use {alt}",
                        "confidence": reject_rate,
                        "applies_to": lambda d, a=alt: a.lower() in ' '.join(d.get('alternatives', [])).lower(),
                        "check": lambda d, a=alt: a.lower() not in d['decision'].lower(),
                        "evidence": []  # Would need to track decision IDs
                    })

        return patterns

    def detect_anomalies_in_memory(self, new_memory: Dict, existing_memories: List[Dict]) -> List[Dict]:
        """Detect anomalies when storing a new memory."""

        anomalies = []

        # Check for contradictions with existing memories
        for existing in existing_memories:
            contradiction = self.check_contradiction(
                new_memory.get('content', ''),
                existing.get('content', '')
            )

            if contradiction:
                anomalies.append({
                    **contradiction,
                    "new_memory": new_memory.get('subject', 'Untitled'),
                    "conflicts_with": {
                        "id": existing.get('id', 'unknown'),
                        "subject": existing.get('subject', 'Untitled')
                    }
                })

        return anomalies

    def log_anomaly(self, anomaly: Dict):
        """Log detected anomaly."""
        anomaly['timestamp'] = datetime.now(timezone.utc).isoformat()

        with open(self.anomaly_log, 'a') as f:
            f.write(json.dumps(anomaly) + '\n')

    def get_recent_anomalies(self, days: int = 7) -> List[Dict]:
        """Get anomalies from past N days."""
        if not self.anomaly_log.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = []

        with open(self.anomaly_log, 'r') as f:
            for line in f:
                anomaly = json.loads(line)
                timestamp = datetime.fromisoformat(anomaly['timestamp'])
                if timestamp > cutoff:
                    recent.append(anomaly)

        return recent

    def generate_report(self, days: int = 7) -> str:
        """Generate anomaly detection report."""
        anomalies = self.get_recent_anomalies(days)

        if not anomalies:
            return f"No anomalies detected in past {days} days."

        lines = []
        lines.append("=" * 60)
        lines.append(f"ANOMALY DETECTION REPORT (Past {days} days)")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Total anomalies: {len(anomalies)}")
        lines.append("")

        # Group by type
        by_type = defaultdict(list)
        for a in anomalies:
            by_type[a['type']].append(a)

        lines.append("By type:")
        for atype, items in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
            severity = items[0].get('severity', 'unknown')
            lines.append(f"  - {atype}: {len(items)} ({severity} severity)")

        lines.append("")

        # Show high severity anomalies
        high_severity = [a for a in anomalies if a.get('severity') == 'high']
        if high_severity:
            lines.append("High severity anomalies (require review):")
            for a in high_severity[:5]:  # Show max 5
                lines.append(f"  [{a['timestamp'][:10]}] {a['type']}")
                if 'pattern' in a:
                    lines.append(f"    Pattern: {a['pattern']}")
                if 'principle' in a:
                    lines.append(f"    Violates: {a['principle'][:60]}...")
                lines.append("")

        # Trends
        lines.append("Trends:")
        if len(anomalies) > 10:
            lines.append(f"  WARNING: High anomaly rate ({len(anomalies)} in {days} days)")
            lines.append("  Recommendation: Review identity files and patterns")
        else:
            lines.append(f"  Normal anomaly rate ({len(anomalies)} in {days} days)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic anomaly detection for AI agent memories")
    parser.add_argument("--memory-dir", help="Path to memory directory (default: $MEMORY_DIR or ./memory)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check for contradictions")
    check_parser.add_argument("--text1", required=True, help="First text")
    check_parser.add_argument("--text2", required=True, help="Second text")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate anomaly report")
    report_parser.add_argument("--days", type=int, default=7, help="Days to report")

    # Drift check command
    drift_parser = subparsers.add_parser("drift", help="Check for behavioral drift")
    drift_parser.add_argument("--action", required=True, help="Action to check")

    args = parser.parse_args()

    detector = AnomalyDetector(memory_dir=args.memory_dir)

    if args.command == "check":
        contradiction = detector.check_contradiction(args.text1, args.text2)

        if contradiction:
            print("CONTRADICTION DETECTED")
            print(f"   Pattern: {contradiction['pattern']}")
            print(f"   Overlap score: {contradiction['overlap_score']:.2f}")
            print(f"   Keywords: {', '.join(contradiction['overlapping_keywords'])}")
        else:
            print("No contradiction detected")

    elif args.command == "report":
        print(detector.generate_report(args.days))

    elif args.command == "drift":
        violations = detector.check_behavioral_drift(args.action)

        if violations:
            print("DRIFT DETECTED")
            for v in violations:
                print(f"   Action: {v['action']}")
                print(f"   Violates principle: {v['principle']}")
                print(f"   Source: {v['source']}")
                print(f"   Overlap score: {v['overlap_score']:.2f}")
                print()
        else:
            print("No drift detected")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
