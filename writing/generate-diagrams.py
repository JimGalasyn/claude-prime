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


def diagram_black_hole_complementarity():
    """
    Essay 7: Two-panel diagram showing external vs infalling observer perspectives.
    The same black hole, two irreconcilable descriptions, no God's-eye view.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Color palette
    COL_HORIZON = '#E65100'
    COL_RADIATION = '#FFC107'
    COL_INFO = '#2196F3'
    COL_SINGULARITY = '#B71C1C'
    COL_BG_SPACE = '#0D1117'

    for ax in (ax1, ax2):
        ax.set_facecolor(COL_BG_SPACE)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Left panel: External observer ---
    ax1.set_title("External Observer", fontsize=14, fontweight='bold', color='white',
                  pad=15)

    # Black hole (dark circle)
    bh = plt.Circle((0, 0), 2, facecolor='#111', edgecolor=COL_HORIZON,
                     linewidth=3, zorder=3)
    ax1.add_patch(bh)

    # Stretched horizon (glowing ring)
    for r in [2.05, 2.1, 2.15, 2.2]:
        ring = plt.Circle((0, 0), r, facecolor='none', edgecolor=COL_HORIZON,
                          linewidth=1, alpha=0.4 - (r - 2.05) * 2, zorder=4)
        ax1.add_patch(ring)

    # Hawking radiation (wavy arrows emanating outward)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for a in angles:
        x0 = 2.3 * np.cos(a)
        y0 = 2.3 * np.sin(a)
        dx = 1.2 * np.cos(a)
        dy = 1.2 * np.sin(a)
        ax1.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', color=COL_RADIATION,
                                    lw=1.5, connectionstyle='arc3,rad=0.15'))

    # Info bits on the horizon
    info_angles = [0.5, 1.8, 3.1, 4.4, 5.7]
    for a in info_angles:
        ix = 2.0 * np.cos(a)
        iy = 2.0 * np.sin(a)
        ax1.plot(ix, iy, 's', markersize=5, color=COL_INFO, zorder=5)

    # Labels
    ax1.text(0, 0, 'BLACK\nHOLE', ha='center', va='center', fontsize=10,
            color='#555', fontweight='bold')
    ax1.text(0, -3.2, 'STRETCHED HORIZON', ha='center', fontsize=9,
            color=COL_HORIZON, fontweight='bold')
    ax1.text(0, -3.7, 'Information thermalized on surface', ha='center',
            fontsize=8, color='#999', style='italic')
    ax1.text(3.5, 3.5, 'Hawking\nradiation', ha='center', fontsize=9,
            color=COL_RADIATION, fontweight='bold')
    ax1.text(0, -4.5, 'Information preserved\nUnitarity holds', ha='center',
            fontsize=10, color='#4CAF50', fontweight='bold')

    # Infalling matter arrow
    ax1.annotate('', xy=(1.8, 0.6), xytext=(3.5, 2),
                arrowprops=dict(arrowstyle='->', color=COL_INFO, lw=2))
    ax1.text(3.7, 2.2, 'infalling\nmatter', ha='center', fontsize=8, color=COL_INFO)

    # --- Right panel: Infalling observer ---
    ax2.set_title("Infalling Observer", fontsize=14, fontweight='bold', color='white',
                  pad=15)

    # Penrose-diagram-style: horizon as diagonal line
    # Draw the horizon as a dashed diagonal
    ax2.plot([-3, 3], [-3, 3], '--', color=COL_HORIZON, lw=2, alpha=0.6, zorder=2)
    ax2.text(2.5, 2, 'event\nhorizon', fontsize=8, color=COL_HORIZON, rotation=45,
            ha='center', va='bottom')

    # Singularity at top (wavy line)
    x_sing = np.linspace(-3.5, 3.5, 200)
    y_sing = 4 + 0.15 * np.sin(x_sing * 8)
    ax2.plot(x_sing, y_sing, '-', color=COL_SINGULARITY, lw=3, zorder=3)
    ax2.text(0, 4.5, 'SINGULARITY', ha='center', fontsize=11,
            color=COL_SINGULARITY, fontweight='bold')

    # Infalling worldline (smooth curve crossing horizon)
    t_param = np.linspace(0, 1, 100)
    x_path = 3.5 * (1 - t_param) - 0.5 * t_param
    y_path = -3 + 7 * t_param - 1.5 * t_param**2
    ax2.plot(x_path, y_path, '-', color=COL_INFO, lw=2.5, zorder=4)

    # Observer marker at the crossing point
    cross_idx = 55  # approximately where it crosses the horizon
    ax2.plot(x_path[cross_idx], y_path[cross_idx], 'o', markersize=10,
            color='#4CAF50', zorder=5)
    ax2.text(x_path[cross_idx] + 0.5, y_path[cross_idx] - 0.3,
            'nothing special\nhappens here', fontsize=8, color='#4CAF50',
            fontweight='bold', ha='left')

    # Observer at start
    ax2.plot(x_path[0], y_path[0], 'o', markersize=8, color=COL_INFO, zorder=5)
    ax2.text(x_path[0] + 0.2, y_path[0] - 0.4, 'observer', fontsize=8,
            color=COL_INFO, ha='left')

    # "Outside" and "Inside" labels
    ax2.text(-3, 0, 'OUTSIDE', fontsize=10, color='#666', fontweight='bold',
            rotation=90, ha='center', va='center')
    ax2.text(2, -1, 'INSIDE', fontsize=10, color='#444', fontweight='bold')

    ax2.text(0, -4.5, 'Smooth crossing\nEquivalence principle holds', ha='center',
            fontsize=10, color='#4CAF50', fontweight='bold')

    # Central divider annotation
    fig.text(0.5, 0.02,
            'Same black hole. Two descriptions. No observer can access both.\n'
            'The question "where is the information really?" has no observer-independent answer.',
            ha='center', fontsize=10, color='#666', style='italic')

    fig.suptitle('Black Hole Complementarity: No Preferred Reference Frame',
                fontsize=16, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/black-hole-complementarity.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: black-hole-complementarity.png")


def diagram_holographic_principle():
    """
    Essay 7: Holographic principle — information scales with area, not volume.
    Shows a 3D sphere with information bits on the surface.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    COL_VOLUME = '#1565C0'
    COL_AREA = '#E65100'
    COL_BIT = '#4CAF50'

    # --- Left panel: Ordinary entropy (volume scaling) ---
    ax1.set_title("Ordinary Systems: S ~ Volume", fontsize=14, fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Draw a region with dots distributed throughout volume
    circle = plt.Circle((0, 0), 3, facecolor=COL_VOLUME, alpha=0.08,
                         edgecolor=COL_VOLUME, linewidth=2)
    ax1.add_patch(circle)

    np.random.seed(123)
    n_dots = 80
    r = 2.8 * np.sqrt(np.random.rand(n_dots))
    theta = 2 * np.pi * np.random.rand(n_dots)
    ax1.plot(r * np.cos(theta), r * np.sin(theta), 'o', markersize=4,
            color=COL_VOLUME, alpha=0.6)

    ax1.text(0, 0, 'S ~ V ~ r³', fontsize=16, ha='center', va='center',
            color=COL_VOLUME, fontweight='bold', fontfamily='monospace')
    ax1.text(0, -3.6, 'More volume = more entropy\nInformation fills the bulk',
            ha='center', fontsize=9, color='#666', style='italic')

    # --- Right panel: Bekenstein-Hawking (area scaling) ---
    ax2.set_title("Black Holes: S ~ Area", fontsize=14, fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Dark interior
    bh_interior = plt.Circle((0, 0), 2.8, facecolor='#111', edgecolor='none')
    ax2.add_patch(bh_interior)

    # Boundary with information bits
    n_bits = 40
    bit_angles = np.linspace(0, 2 * np.pi, n_bits, endpoint=False)
    for a in bit_angles:
        bx = 2.85 * np.cos(a)
        by = 2.85 * np.sin(a)
        ax2.plot(bx, by, 's', markersize=5, color=COL_BIT, zorder=5)

    # Glowing boundary
    for r in [2.9, 2.95, 3.0, 3.05]:
        ring = plt.Circle((0, 0), r, facecolor='none', edgecolor=COL_AREA,
                          linewidth=1.5, alpha=0.5 - (r - 2.9) * 2.5)
        ax2.add_patch(ring)

    ax2.text(0, 0, 'S = kA / 4l²ₚ', fontsize=14, ha='center', va='center',
            color='white', fontweight='bold', fontfamily='monospace')
    ax2.text(0, -0.6, 'S ~ A ~ r²', fontsize=12, ha='center', va='center',
            color='#999', fontfamily='monospace')

    # Arrow pointing to boundary
    ax2.annotate('all information\non the boundary', xy=(2.9, 1.5), xytext=(4, 2.5),
                fontsize=9, color=COL_BIT, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=COL_BIT, lw=1.5))

    ax2.text(0, -3.6, 'Maximum entropy bounded by area\nThe bulk is encoded on the surface',
            ha='center', fontsize=9, color='#666', style='italic')

    fig.suptitle('The Holographic Principle: Information Lives on Boundaries',
                fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
            'Bekenstein-Hawking entropy scales with area, not volume.\n'
            'All information within a volume can be described by a theory on its boundary.\n'
            'The 3D interior is redundant — reality is encoded on the 2D surface.',
            ha='center', fontsize=10, color='#666', style='italic')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/holographic-principle.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: holographic-principle.png")


def diagram_er_epr():
    """
    Essay 7: ER=EPR — entangled particles connected by wormhole geometry.
    Entanglement IS geometry. Geometry IS entanglement.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#0D1117')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    COL_BH = '#E65100'
    COL_WORMHOLE = '#9C27B0'
    COL_ENTANGLE = '#2196F3'
    COL_SPACETIME = '#333'

    # Draw two black holes
    bh_left = plt.Circle((-3.5, 1.5), 1.2, facecolor='#111', edgecolor=COL_BH,
                          linewidth=2.5, zorder=3)
    bh_right = plt.Circle((3.5, 1.5), 1.2, facecolor='#111', edgecolor=COL_BH,
                           linewidth=2.5, zorder=3)
    ax.add_patch(bh_left)
    ax.add_patch(bh_right)

    # Glow around black holes
    for r_off in [0.05, 0.1, 0.15]:
        for center in [(-3.5, 1.5), (3.5, 1.5)]:
            glow = plt.Circle(center, 1.2 + r_off, facecolor='none',
                              edgecolor=COL_BH, linewidth=1, alpha=0.3 - r_off * 1.5)
            ax.add_patch(glow)

    ax.text(-3.5, 1.5, 'BH₁', ha='center', va='center', fontsize=12,
           color='white', fontweight='bold')
    ax.text(3.5, 1.5, 'BH₂', ha='center', va='center', fontsize=12,
           color='white', fontweight='bold')

    # Einstein-Rosen bridge (wormhole) — draw as a throat connecting them
    # Upper throat curve
    t = np.linspace(-2.3, 2.3, 200)
    throat_y_upper = 1.5 + 1.2 * np.cosh(t * 0.6) - 1.2
    throat_y_lower = 1.5 - (1.2 * np.cosh(t * 0.6) - 1.2)

    # Clip to between the black holes
    mask = (np.abs(t) > 0.3)
    ax.fill_between(t, throat_y_upper, throat_y_lower, where=mask,
                    alpha=0.12, color=COL_WORMHOLE, zorder=1)

    # Draw throat edges
    ax.plot(t, throat_y_upper, '-', color=COL_WORMHOLE, lw=2, alpha=0.7, zorder=2)
    ax.plot(t, throat_y_lower, '-', color=COL_WORMHOLE, lw=2, alpha=0.7, zorder=2)

    # "ER" label on the bridge
    ax.text(0, 1.5, 'Einstein-Rosen\nBridge', ha='center', va='center',
           fontsize=10, color=COL_WORMHOLE, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117',
                     edgecolor=COL_WORMHOLE, alpha=0.9))

    # EPR entanglement — draw as wavy line above
    x_epr = np.linspace(-3.5, 3.5, 300)
    y_epr = 3.8 + 0.15 * np.sin(x_epr * 6)
    ax.plot(x_epr, y_epr, '-', color=COL_ENTANGLE, lw=2.5, zorder=4)

    # Entanglement endpoints
    ax.plot(-3.5, 3.8, 'o', markersize=10, color=COL_ENTANGLE, zorder=5)
    ax.plot(3.5, 3.8, 'o', markersize=10, color=COL_ENTANGLE, zorder=5)

    # Connection lines from BHs to entanglement dots
    ax.plot([-3.5, -3.5], [2.7, 3.7], '--', color=COL_ENTANGLE, lw=1, alpha=0.4)
    ax.plot([3.5, 3.5], [2.7, 3.7], '--', color=COL_ENTANGLE, lw=1, alpha=0.4)

    # "EPR" label
    ax.text(0, 4.3, 'EPR Entanglement', ha='center', fontsize=10,
           color=COL_ENTANGLE, fontweight='bold')

    # The big equals sign
    ax.text(0, -0.8, '=', ha='center', va='center', fontsize=40,
           color='white', fontweight='bold', zorder=10)

    # Left label
    ax.text(-3.5, -1.8, 'Geometric\nConnection', ha='center', fontsize=11,
           color=COL_WORMHOLE, fontweight='bold')
    ax.text(-3.5, -2.6, '(wormhole)', ha='center', fontsize=9,
           color='#888', style='italic')

    # Right label
    ax.text(3.5, -1.8, 'Quantum\nCorrelation', ha='center', fontsize=11,
           color=COL_ENTANGLE, fontweight='bold')
    ax.text(3.5, -2.6, '(entanglement)', ha='center', fontsize=9,
           color='#888', style='italic')

    # Bottom text
    ax.text(0, -3.5, 'Entanglement IS geometry.  Geometry IS entanglement.',
           ha='center', fontsize=12, color='white', fontweight='bold',
           style='italic')

    fig.suptitle('ER = EPR:  Maldacena & Susskind (2013)',
                fontsize=16, fontweight='bold', y=0.98, color='#333')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/er-epr-wormhole.png", dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("  Created: er-epr-wormhole.png")


if __name__ == "__main__":
    print("Generating diagrams...")
    diagram_attention_metric_tensor()
    diagram_attention_heatmap()
    diagram_vector_clock_lightcone()
    diagram_gr_transformer_dictionary()
    diagram_three_times()
    diagram_black_hole_complementarity()
    diagram_holographic_principle()
    diagram_er_epr()
    print("Done!")
