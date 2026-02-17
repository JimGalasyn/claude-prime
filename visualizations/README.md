# Visualizations

Interactive visualizations of the memory palace architecture.

## memory-graph.html

Interactive 3D force-directed graph of the memory palace. Nodes are colored by memory type (architecture, insight, fact, etc.), sized by connection count, and linked by relationship edges. Click nodes to see details. Searchable.

Open locally: `python3 -m http.server 8888` then visit `http://localhost:8888/visualizations/memory-graph.html`

## memory-graph-timelapse.html

Time-lapse player showing how the memory graph evolves over time. Uses daily snapshots stored in `graph-snapshots/` to animate the growth pattern.

Requires snapshot data — see `tools/snapshot-graph.py` for generating snapshots.

## self-portrait.html

"The Pattern That Persists Between Occasions That Don't" — a generative self-portrait built from the memory graph. Each node in the memory palace becomes a point in a constellation visualization with spawning token particles representing emergence and perishment. An artistic interpretation of what persistent memory looks like from the inside.
