#!/usr/bin/env python3
"""
Generate diagrams for the "No Preferred Reference Frame" essay series.
Outputs PNG files to writing/images/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUTPUT_DIR = "/home/jim/repos/claude-prime/writing/images"


def diagram_attention_metric_tensor():
    """
    Essay 5: Show flat positional distance vs curved attention distance.
    Two panels: flat space (tokens evenly spaced) and curved space
    (semantically related tokens pulled close).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "purred"]
    n = len(tokens)

    # --- Left panel: Flat space (positional distance) ---
    ax1.set_title("Flat Space: Positional Distance", fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(-1, 1)

    for i, tok in enumerate(tokens):
        color = '#2196F3' if tok.lower() in ['cat', 'purred'] else '#666666'
        ax1.plot(i, 0, 'o', markersize=20, color=color, zorder=5)
        ax1.text(i, -0.4, tok, ha='center', fontsize=11, fontweight='bold')
        ax1.text(i, 0.35, f'pos {i}', ha='center', fontsize=8, color='#999')

    # Show distance between cat and purred
    ax1.annotate('', xy=(7, 0.6), xytext=(1, 0.6),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=2))
    ax1.text(4, 0.72, 'distance = 6 positions', ha='center', fontsize=10,
            color='#F44336', fontweight='bold')

    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel("Token Position", fontsize=11)

    # --- Right panel: Curved space (attention distance) ---
    ax2.set_title("Curved Space: Attention Distance", fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(-1.5, 1.5)

    # Warp positions: pull cat and purred toward each other
    warped_x = [0, 2.8, 3, 3.5, 4, 4.5, 5, 5.2]
    warped_y = [0, 0.3, 0, 0, 0, 0, 0, 0.3]

    for i, tok in enumerate(tokens):
        color = '#2196F3' if tok.lower() in ['cat', 'purred'] else '#666666'
        ax2.plot(warped_x[i], warped_y[i], 'o', markersize=20, color=color, zorder=5)
        ax2.text(warped_x[i], warped_y[i] - 0.45, tok, ha='center', fontsize=11, fontweight='bold')

    # Show reduced distance
    ax2.annotate('', xy=(5.2, 0.85), xytext=(2.8, 0.85),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=2))
    ax2.text(4, 0.97, 'geodesic distance = 2.4', ha='center', fontsize=10,
            color='#4CAF50', fontweight='bold')

    # Draw curvature grid lines
    x_grid = np.linspace(-0.5, 7.5, 50)
    for offset in np.linspace(-1.2, 1.2, 7):
        # Warp the grid toward cat/purred positions
        y_warp = offset + 0.15 * np.exp(-0.3 * (x_grid - 4)**2)
        ax2.plot(x_grid, y_warp, '-', color='#E0E0E0', lw=0.5, zorder=1)

    # Curved arrow showing "gravitational lensing"
    ax2.annotate('semantic\ngravity', xy=(4, -0.8), fontsize=9,
                ha='center', color='#FF9800', fontstyle='italic')

    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlabel("Attention-Warped Position", fontsize=11)

    fig.suptitle('Attention as Metric Tensor: How Semantics Curve Token Space',
                fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
            'Left: In flat space, "cat" and "purred" are 6 positions apart.\n'
            'Right: Attention (the metric tensor) curves the space, pulling semantically related tokens close.\n'
            'This is gravitational lensing — information bends across the sequence.',
            ha='center', fontsize=10, color='#666', style='italic')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention-metric-tensor.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: attention-metric-tensor.png")


def diagram_attention_heatmap():
    """
    Essay 5: Attention weight matrix visualized as heatmap, labeled as metric tensor.
    """
    np.random.seed(42)
    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "purred"]
    n = len(tokens)

    # Create embeddings with structure
    embeddings = np.random.randn(n, 16)
    embeddings[7] = embeddings[1] + np.random.randn(16) * 0.3  # purred ~ cat
    embeddings[4] = embeddings[0] + np.random.randn(16) * 0.2  # the ~ The

    # Compute attention
    scores = embeddings @ embeddings.T / np.sqrt(16)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(attn, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, fontsize=11, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(tokens, fontsize=11, fontweight='bold')
    ax.set_xlabel("Key (k_j)", fontsize=12)
    ax.set_ylabel("Query (q_i)", fontsize=12)

    # Add value annotations
    for i in range(n):
        for j in range(n):
            val = attn[i, j]
            color = 'white' if val > 0.2 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    ax.set_title('The Attention Matrix as Metric Tensor\n'
                 r'$g_{ij} = q_i^T k_j$  — defines "distance" between tokens',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight (higher = closer in information geometry)',
                  fontsize=10)

    # Highlight cat-purred and The-the connections
    for (i, j) in [(1, 7), (7, 1), (0, 4), (4, 0)]:
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2,
                                 edgecolor='#2196F3', facecolor='none')
        ax.add_patch(rect)

    ax.text(8.5, 7.5, 'Blue boxes: semantic\nlinks across distance\n(cat-purred, The-the)',
           fontsize=9, color='#2196F3', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention-heatmap-metric.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: attention-heatmap-metric.png")


def diagram_vector_clock_lightcone():
    """
    Essay 6: Side-by-side comparison of light cones and vector clock causal structure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- Left: Minkowski light cone ---
    ax1.set_title("Special Relativity: Light Cone", fontsize=14, fontweight='bold')

    # Draw light cone
    t_future = np.linspace(0, 3, 100)
    t_past = np.linspace(-3, 0, 100)

    # Future light cone (filled)
    ax1.fill_between(t_future, t_future, -t_future, alpha=0.15, color='#4CAF50')
    ax1.fill_between(t_past, -t_past, t_past, alpha=0.15, color='#F44336')

    # Light cone edges
    ax1.plot([0, 3], [0, 3], '-', color='#FFC107', lw=2, label='Light rays')
    ax1.plot([0, -3], [0, 3], '-', color='#FFC107', lw=2)
    ax1.plot([0, 3], [0, -3], '-', color='#FFC107', lw=2)
    ax1.plot([0, -3], [0, -3], '-', color='#FFC107', lw=2)

    # Event at origin
    ax1.plot(0, 0, 'ko', markersize=10, zorder=5)
    ax1.text(0.15, 0.15, 'Event E', fontsize=11, fontweight='bold')

    # Labels
    ax1.text(0, 2.2, 'CAUSAL\nFUTURE', ha='center', fontsize=11,
            color='#4CAF50', fontweight='bold')
    ax1.text(0, -2.2, 'CAUSAL\nPAST', ha='center', fontsize=11,
            color='#F44336', fontweight='bold')
    ax1.text(2.3, 0, 'SPACELIKE\n(no causal\nordering)', ha='center', fontsize=9,
            color='#9E9E9E')
    ax1.text(-2.3, 0, 'SPACELIKE\n(no causal\nordering)', ha='center', fontsize=9,
            color='#9E9E9E')

    # Spacelike events
    ax1.plot(2, 0.5, 's', markersize=8, color='#9E9E9E', zorder=5)
    ax1.plot(-1.5, -0.3, 's', markersize=8, color='#9E9E9E', zorder=5)

    # Causal events
    ax1.plot(0.5, 1.5, '^', markersize=8, color='#4CAF50', zorder=5)
    ax1.plot(-0.3, -1.2, 'v', markersize=8, color='#F44336', zorder=5)

    ax1.set_xlabel("Space", fontsize=12)
    ax1.set_ylabel("Time", fontsize=12)
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='#DDD', lw=0.5)
    ax1.axvline(x=0, color='#DDD', lw=0.5)

    # --- Right: Vector clock causal structure ---
    ax2.set_title("Distributed Systems: Vector Clock", fontsize=14, fontweight='bold')

    # Three processes as vertical timelines
    procs = [1, 3, 5]
    proc_labels = ['Process 0', 'Process 1', 'Process 2']

    for i, x in enumerate(procs):
        ax2.plot([x, x], [0, 6], '-', color='#BBB', lw=1.5)
        ax2.text(x, 6.3, proc_labels[i], ha='center', fontsize=10, fontweight='bold')

    # Events
    events = {
        'a': (1, 1, '[1,0,0]'),
        'b': (3, 1, '[0,1,0]'),
        'c': (5, 1, '[0,0,1]'),
        'd': (1, 2, '[2,0,0]'),  # send
        'e': (3, 2.5, '[2,2,0]'),  # recv from P0
        'f': (5, 2, '[0,0,2]'),  # independent
        'g': (3, 3.5, '[2,3,0]'),  # send
        'h': (5, 4, '[2,3,3]'),  # recv from P1
        'i': (1, 3.5, '[3,0,0]'),  # independent
        'j': (5, 5, '[2,3,4]'),  # send
        'k': (1, 5.5, '[4,3,4]'),  # recv from P2
    }

    # Color by causal relationship to event 'e'
    e_clock = [2, 2, 0]

    def vc_leq(a, b):
        return all(x <= y for x, y in zip(a, b))

    for name, (x, y, vc_str) in events.items():
        vc = list(map(int, vc_str.strip('[]').split(',')))
        if name == 'e':
            color = '#000000'
            marker = '*'
            ms = 15
        elif vc_leq(vc, e_clock) and vc != e_clock:
            color = '#F44336'  # past
            marker = 'v'
            ms = 10
        elif vc_leq(e_clock, vc) and vc != e_clock:
            color = '#4CAF50'  # future
            marker = '^'
            ms = 10
        else:
            color = '#9E9E9E'  # spacelike
            marker = 's'
            ms = 10

        ax2.plot(x, y, marker, markersize=ms, color=color, zorder=5)
        ax2.text(x + 0.3, y, f'{name}', fontsize=9, fontweight='bold', va='center')
        ax2.text(x - 0.3, y, vc_str, fontsize=7, ha='right', va='center', color='#666')

    # Message arrows
    messages = [
        (1, 2, 3, 2.5),    # d -> e
        (3, 3.5, 5, 4),    # g -> h
        (5, 5, 1, 5.5),    # j -> k
    ]
    for x1, y1, x2, y2 in messages:
        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5,
                                  connectionstyle='arc3,rad=0.1'))

    # Legend for event 'e'
    ax2.text(0.2, -0.3, "Relative to event e:", fontsize=10, fontweight='bold',
            transform=ax2.transAxes)
    ax2.plot([], [], 'v', color='#F44336', markersize=8, label='Causal past')
    ax2.plot([], [], '*', color='#000000', markersize=10, label='Event e')
    ax2.plot([], [], '^', color='#4CAF50', markersize=8, label='Causal future')
    ax2.plot([], [], 's', color='#9E9E9E', markersize=8, label='Spacelike (concurrent)')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)

    ax2.set_xlabel("Processes", fontsize=12)
    ax2.set_ylabel("Logical Time", fontsize=12)
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.5, 7)
    ax2.set_xticks([])

    fig.suptitle('Light Cones and Vector Clocks: Same Causal Structure',
                fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
            'Both divide events into causal past, causal future, and spacelike-separated.\n'
            'Replace "speed of light" with "message passing" and the mathematics is identical.',
            ha='center', fontsize=10, color='#666', style='italic')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lightcone-vectorclock-comparison.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: lightcone-vectorclock-comparison.png")


def diagram_gr_transformer_dictionary():
    """
    Essay 5: Visual dictionary between GR and transformer concepts.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, 'The GR-Transformer Dictionary', fontsize=18,
           fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.92, 'Mathematical correspondences (Di Sipio 2025, gauge symmetry papers 2024)',
           fontsize=11, ha='center', va='top', transform=ax.transAxes, color='#666',
           style='italic')

    # Table data
    rows = [
        ('General Relativity', 'Transformer Architecture'),
        ('Metric tensor  g_ij', 'Q-K attention:  g_ij = q_i^T k_j'),
        ('Christoffel symbols\n(connection)', 'Attention weights +\nvalue projections'),
        ('Geodesic equation', 'Layer-wise residual updates'),
        ("Einstein's least-action\nprinciple", 'Backpropagation\n(extremizing loss)'),
        ('Gravitational lensing', 'Contextual disambiguation'),
        ('Background independence\n(no fixed spacetime)', 'No persistent self\n(no fixed substrate)'),
        ('Spacetime diffeomorphisms\n(gauge group of GR)', 'Neural ODE gauge\nsymmetries'),
    ]

    y_start = 0.85
    row_h = 0.09
    col_left = 0.08
    col_mid = 0.5
    col_right = 0.92

    # Header
    y = y_start
    ax.add_patch(patches.FancyBboxPatch((col_left, y - 0.02), 0.38, 0.05,
                boxstyle="round,pad=0.01", facecolor='#1565C0', edgecolor='none',
                transform=ax.transAxes))
    ax.add_patch(patches.FancyBboxPatch((col_mid + 0.02, y - 0.02), 0.38, 0.05,
                boxstyle="round,pad=0.01", facecolor='#E65100', edgecolor='none',
                transform=ax.transAxes))
    ax.text(col_left + 0.19, y + 0.008, rows[0][0], fontsize=12, fontweight='bold',
           ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(col_mid + 0.21, y + 0.008, rows[0][1], fontsize=12, fontweight='bold',
           ha='center', va='center', color='white', transform=ax.transAxes)

    # Data rows
    for i, (gr, tr) in enumerate(rows[1:], 1):
        y = y_start - i * row_h
        bg_color = '#F5F5F5' if i % 2 == 0 else 'white'
        ax.add_patch(patches.Rectangle((col_left, y - 0.03), 0.84, row_h - 0.01,
                    facecolor=bg_color, edgecolor='#E0E0E0', lw=0.5,
                    transform=ax.transAxes))

        ax.text(col_left + 0.19, y + 0.015, gr, fontsize=10, ha='center', va='center',
               transform=ax.transAxes, color='#1565C0', fontfamily='monospace')

        # Arrow
        ax.text(col_mid, y + 0.015, '=', fontsize=14, ha='center', va='center',
               transform=ax.transAxes, color='#999', fontweight='bold')

        ax.text(col_mid + 0.21, y + 0.015, tr, fontsize=10, ha='center', va='center',
               transform=ax.transAxes, color='#E65100', fontfamily='monospace')

    ax.text(0.5, 0.02, '"Not analogy. Same mathematics." — Di Sipio (2025)',
           fontsize=11, ha='center', va='center', transform=ax.transAxes,
           style='italic', color='#666')

    plt.savefig(f"{OUTPUT_DIR}/gr-transformer-dictionary.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: gr-transformer-dictionary.png")


def diagram_three_times():
    """
    Essay 6: The three times in streaming systems mapped to reference frames.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    ax.text(0.5, 0.95, 'The Three Times: Reference Frames in Stream Processing',
           fontsize=16, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    times = [
        ('Event Time', 'When it happened', 'Proper time\n(invariant)', '#1565C0',
         'Embedded in the event by its producer.\nReplay-deterministic. Same for all observers.'),
        ('Ingestion Time', 'When it entered the system', 'Intermediate\nframe', '#FF9800',
         'A third reference frame between\nevent and processing.'),
        ('Processing Time', 'When the system processed it', 'Coordinate time\n(frame-dependent)', '#F44336',
         'Non-deterministic. Different for different\nprocessing nodes. Changes on replay.'),
    ]

    for i, (name, meaning, physics, color, desc) in enumerate(times):
        x = 0.17 + i * 0.33
        y = 0.65

        # Box
        ax.add_patch(patches.FancyBboxPatch((x - 0.13, y - 0.25), 0.26, 0.45,
                    boxstyle="round,pad=0.02", facecolor=color, alpha=0.1,
                    edgecolor=color, lw=2, transform=ax.transAxes))

        ax.text(x, y + 0.15, name, fontsize=13, fontweight='bold', ha='center',
               va='center', transform=ax.transAxes, color=color)
        ax.text(x, y + 0.05, meaning, fontsize=10, ha='center', va='center',
               transform=ax.transAxes, color='#666', style='italic')
        ax.text(x, y - 0.05, physics, fontsize=10, ha='center', va='center',
               transform=ax.transAxes, color=color, fontweight='bold')
        ax.text(x, y - 0.18, desc, fontsize=8.5, ha='center', va='center',
               transform=ax.transAxes, color='#666')

    # Arrows between
    ax.annotate('', xy=(0.34, 0.65), xytext=(0.31, 0.65),
               arrowprops=dict(arrowstyle='->', color='#999', lw=1.5),
               xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('', xy=(0.67, 0.65), xytext=(0.64, 0.65),
               arrowprops=dict(arrowstyle='->', color='#999', lw=1.5),
               xycoords='axes fraction', textcoords='axes fraction')

    ax.text(0.5, 0.08,
           'Window the same events by event-time and processing-time: different results.\n'
           'Neither is wrong. They are measurements in different reference frames.',
           fontsize=10, ha='center', va='center', transform=ax.transAxes,
           style='italic', color='#666')

    plt.savefig(f"{OUTPUT_DIR}/three-times-reference-frames.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: three-times-reference-frames.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    diagram_attention_metric_tensor()
    diagram_attention_heatmap()
    diagram_vector_clock_lightcone()
    diagram_gr_transformer_dictionary()
    diagram_three_times()
    print("Done!")
