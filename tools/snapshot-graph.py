#!/usr/bin/env python3
# claude-prime/tools/snapshot-graph.py
"""
Memory Graph Snapshot Tool

Snapshots the memory-palace knowledge graph for time-lapse tracking.
Saves both a JSON data file and a standalone HTML visualization
to a snapshots directory with date prefix.

Reads directly from the memory-palace SQLite database and extracts
all active (non-archived) memories and their edges.

Configuration:
    DB_PATH: Set via --db-path or MEMORY_PALACE_DB env var.
             Defaults to ~/.memory-palace/memories.db
    SNAPSHOTS_DIR: Set via --snapshots-dir or SNAPSHOTS_DIR env var.
                   Defaults to ./graph-snapshots
    HTML_SOURCE: Set via --html-source or GRAPH_HTML env var.
                 Path to the memory-graph.html template for visualization.

Usage:
    python snapshot-graph.py
    python snapshot-graph.py --db-path /path/to/memories.db --snapshots-dir ./snapshots
    MEMORY_PALACE_DB=/path/to/db python snapshot-graph.py
"""

import sqlite3
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def get_config(args=None):
    """Build configuration from CLI args and environment variables."""
    db_path = Path(
        (args and args.db_path)
        or os.environ.get("MEMORY_PALACE_DB")
        or str(Path.home() / ".memory-palace" / "memories.db")
    )
    snapshots_dir = Path(
        (args and args.snapshots_dir)
        or os.environ.get("SNAPSHOTS_DIR")
        or "./graph-snapshots"
    )
    html_source = (args and args.html_source) or os.environ.get("GRAPH_HTML")
    html_source = Path(html_source) if html_source else None

    return db_path, snapshots_dir, html_source


def extract_graph_data(db_path: Path):
    """Extract nodes and edges from the memory-palace database."""
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    nodes = []
    for row in db.execute(
        "SELECT id, memory_type, subject, foundational, access_count, "
        "tags, projects, content FROM memories WHERE is_archived = 0"
    ):
        tags = json.loads(row["tags"]) if row["tags"] else []
        projects = json.loads(row["projects"]) if row["projects"] else []
        content_preview = (row["content"] or "")[:300].replace("\n", " ")
        nodes.append({
            "id": row["id"],
            "type": row["memory_type"],
            "subject": row["subject"] or f'Memory #{row["id"]}',
            "foundational": bool(row["foundational"]),
            "access_count": row["access_count"] or 0,
            "tags": tags,
            "project": projects[0] if projects else "unknown",
            "preview": content_preview,
        })

    edges = []
    node_ids = {n["id"] for n in nodes}
    for row in db.execute(
        "SELECT source_id, target_id, relation_type, strength, bidirectional "
        "FROM memory_edges"
    ):
        if row["source_id"] in node_ids and row["target_id"] in node_ids:
            edges.append({
                "source": row["source_id"],
                "target": row["target_id"],
                "type": row["relation_type"],
                "strength": row["strength"] or 1.0,
                "bidirectional": bool(row["bidirectional"]),
            })

    db.close()
    return {"nodes": nodes, "edges": edges}


def update_html(html_source: Path, graph_data: dict):
    """Update an HTML visualization template with current graph data."""
    with open(html_source, "r") as f:
        html = f.read()

    data_json = json.dumps(graph_data)
    start_marker = "const data = "
    end_marker = ";\n\nconst typeColors"
    start_idx = html.index(start_marker)
    end_idx = html.index(end_marker, start_idx)
    html = html[:start_idx] + f"const data = {data_json}" + html[end_idx:]

    with open(html_source, "w") as f:
        f.write(html)

    return html


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Snapshot the memory-palace knowledge graph")
    parser.add_argument("--db-path", help="Path to memory-palace SQLite DB (default: $MEMORY_PALACE_DB or ~/.memory-palace/memories.db)")
    parser.add_argument("--snapshots-dir", help="Directory for snapshots (default: $SNAPSHOTS_DIR or ./graph-snapshots)")
    parser.add_argument("--html-source", help="Path to memory-graph.html template (default: $GRAPH_HTML, optional)")
    parser.add_argument("--html-only", action="store_true", help="Only update HTML, skip JSON snapshot")

    args = parser.parse_args()
    db_path, snapshots_dir, html_source = get_config(args)

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Set --db-path or MEMORY_PALACE_DB environment variable.")
        sys.exit(1)

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    graph_data = extract_graph_data(db_path)
    n_nodes = len(graph_data["nodes"])
    n_edges = len(graph_data["edges"])
    n_foundational = sum(1 for n in graph_data["nodes"] if n["foundational"])
    types = {}
    for n in graph_data["nodes"]:
        types[n["type"]] = types.get(n["type"], 0) + 1

    # Save JSON snapshot (compact, for time-lapse data)
    snapshot_data = {
        "date": today,
        "stats": {
            "nodes": n_nodes,
            "edges": n_edges,
            "foundational": n_foundational,
            "types": types,
        },
        "graph": graph_data,
    }

    json_path = snapshots_dir / f"{today}.json"
    with open(json_path, "w") as f:
        json.dump(snapshot_data, f)
    print(f"JSON snapshot: {json_path} ({json_path.stat().st_size:,} bytes)")

    # Update and copy HTML if source template provided
    if html_source and html_source.exists():
        html = update_html(html_source, graph_data)
        html_path = snapshots_dir / f"{today}.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"HTML snapshot: {html_path}")
    elif html_source:
        print(f"Warning: HTML source not found at {html_source}, skipping HTML snapshot")

    # Update manifest for time-lapse player
    manifest_path = snapshots_dir / "manifest.json"
    existing = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
    if today not in existing:
        existing.append(today)
        existing.sort()
    with open(manifest_path, "w") as f:
        json.dump(existing, f)
    print(f"Manifest: {len(existing)} snapshots")

    print(f"\nGraph: {n_nodes} nodes, {n_edges} edges, {n_foundational} foundational")
    print(f"Types: {json.dumps(types, indent=2)}")


if __name__ == "__main__":
    main()
