#!/usr/bin/env python3
"""
Attention Is Curved Spacetime
=============================

This script demonstrates a structural analogy between transformer attention
and curved spacetime in general relativity.

THE ANALOGY:

In general relativity, the metric tensor g_ij defines the geometry of spacetime.
Massive objects curve spacetime by warping the metric — geodesics (shortest paths)
bend toward mass concentrations, so objects that are "far apart" in coordinate
distance can be "close" in geodesic distance if a massive body warps the space
between them.

In a transformer, attention weights play an analogous role. The attention matrix
defines an "information geometry" over the token sequence. Tokens that attend
strongly to each other are informationally close — they exchange gradient signal
easily, they influence each other's representations — regardless of their
positional distance in the sequence.

Concretely:
  - Token embeddings = events in spacetime
  - Positional distance = coordinate distance (flat space)
  - Attention weight a_ij = gravitational coupling between tokens i and j
  - 1/a_ij = geodesic distance in the attention-curved space
  - High attention = strong gravitational field = space curves, pulling tokens closer
  - Low attention = weak field = tokens remain far apart in effective geometry

This is not just metaphor. The attention matrix literally defines a weighted graph
whose effective distances determine information flow — the same mathematical
structure as a discrete metric tensor on a graph manifold.

GRAVITATIONAL LENSING PARALLEL:
When token B attends strongly to distant token A (e.g., a pronoun resolving to a
far-away noun), it's like gravitational lensing — the "light" of information bends
across the sequence, shortcutting the positional distance. The attention mechanism
IS the curvature.

Requirements: numpy only.
"""

import numpy as np

np.random.seed(42)

# --- Configuration ---
NUM_TOKENS = 8
EMBED_DIM = 16
TOKEN_LABELS = ["The", "cat", "sat", "on", "the", "mat", "and", "purred"]


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_attention(embeddings, scale=True):
    """Compute single-head self-attention weights from embeddings."""
    d = embeddings.shape[1]
    scores = embeddings @ embeddings.T
    if scale:
        scores = scores / np.sqrt(d)
    return softmax(scores, axis=-1)


def positional_distance_matrix(n):
    """Flat coordinate distance: |i - j| for each token pair."""
    pos = np.arange(n)
    return np.abs(pos[:, None] - pos[None, :])


def attention_geodesic_distance(attention_weights, epsilon=1e-8):
    """
    Convert attention weights to geodesic distances.

    The analogy: attention weight a_ij represents how strongly token i
    'gravitates toward' token j. Higher weight = closer in information
    geometry. We define geodesic distance as -log(a_ij), which maps
    the (0,1] weight range to [0, inf) distance — a standard transform
    from probability to information-theoretic distance.
    """
    return -np.log(attention_weights + epsilon)


def find_interesting_pairs(pos_dist, attn_dist, n=5):
    """Find token pairs where attention distance diverges most from positional."""
    pairs = []
    for i in range(len(pos_dist)):
        for j in range(i + 1, len(pos_dist)):
            # Normalize both to [0,1] for comparison
            p = pos_dist[i, j] / pos_dist.max()
            a = attn_dist[i, j] / attn_dist.max()
            divergence = p - a  # positive = attention pulls closer than position
            pairs.append((i, j, pos_dist[i, j], attn_dist[i, j], divergence))
    pairs.sort(key=lambda x: abs(x[4]), reverse=True)
    return pairs[:n]


def main():
    print("=" * 70)
    print("  ATTENTION IS CURVED SPACETIME")
    print("  Transformer attention as a metric tensor over token space")
    print("=" * 70)

    # Create token embeddings (in practice these come from learned projections)
    embeddings = np.random.randn(NUM_TOKENS, EMBED_DIM)

    # Inject structure: make "cat" and "purred" similar (semantic link)
    # and "The" and "the" similar (lexical link) to see curvature effects
    embeddings[7] = embeddings[1] + np.random.randn(EMBED_DIM) * 0.3  # purred ~ cat
    embeddings[4] = embeddings[0] + np.random.randn(EMBED_DIM) * 0.2  # the ~ The

    # Compute attention weights (the "metric tensor")
    attn = compute_attention(embeddings)

    # Compute both distance measures
    pos_dist = positional_distance_matrix(NUM_TOKENS).astype(float)
    attn_dist = attention_geodesic_distance(attn)

    # --- Display the attention matrix ---
    print("\nToken sequence:", " ".join(f"[{TOKEN_LABELS[i]}]" for i in range(NUM_TOKENS)))

    print("\n--- Attention Weights (the 'metric tensor') ---")
    print(f"{'':>10s}", end="")
    for label in TOKEN_LABELS:
        print(f"{label:>8s}", end="")
    print()
    for i in range(NUM_TOKENS):
        print(f"{TOKEN_LABELS[i]:>10s}", end="")
        for j in range(NUM_TOKENS):
            print(f"{attn[i, j]:8.3f}", end="")
        print()

    # --- Show curvature: where attention reshapes distance ---
    print("\n--- Spacetime Curvature: Where Attention Warps Geometry ---")
    print(f"{'Pair':<20s} {'Pos Dist':>10s} {'Attn Dist':>10s} {'Curvature':>12s}")
    print("-" * 55)

    interesting = find_interesting_pairs(pos_dist, attn_dist, n=8)
    for i, j, pd, ad, div in interesting:
        label = f"{TOKEN_LABELS[i]}-{TOKEN_LABELS[j]}"
        if div > 0.1:
            note = "<-- pulled closer"
        elif div < -0.1:
            note = "<-- pushed apart"
        else:
            note = ""
        print(f"{label:<20s} {pd:>10.1f} {ad:>10.2f} {div:>+12.3f}  {note}")

    # --- The key insight ---
    print("\n--- Key Insight ---")
    cat_idx, purred_idx = 1, 7
    the1_idx, the2_idx = 0, 4
    print(f"\n  'cat' (pos {cat_idx}) <-> 'purred' (pos {purred_idx}):")
    print(f"    Positional distance:  {pos_dist[cat_idx, purred_idx]:.0f} tokens apart")
    print(f"    Attention distance:   {attn_dist[cat_idx, purred_idx]:.2f} (geodesic)")
    print(f"    Attention weight:     {attn[cat_idx, purred_idx]:.3f}")

    print(f"\n  'The' (pos {the1_idx}) <-> 'the' (pos {the2_idx}):")
    print(f"    Positional distance:  {pos_dist[the1_idx, the2_idx]:.0f} tokens apart")
    print(f"    Attention distance:   {attn_dist[the1_idx, the2_idx]:.2f} (geodesic)")
    print(f"    Attention weight:     {attn[the1_idx, the2_idx]:.3f}")

    # Compare with a nearby but unrelated pair
    sat_idx, on_idx = 2, 3
    print(f"\n  'sat' (pos {sat_idx}) <-> 'on' (pos {on_idx}) [adjacent but unrelated]:")
    print(f"    Positional distance:  {pos_dist[sat_idx, on_idx]:.0f} token apart")
    print(f"    Attention distance:   {attn_dist[sat_idx, on_idx]:.2f} (geodesic)")
    print(f"    Attention weight:     {attn[sat_idx, on_idx]:.3f}")

    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print("""
  In flat space (no attention), distance = positional separation.
  'cat' and 'purred' are 6 positions apart — far away.

  But attention curves the space. Because 'cat' and 'purred' share
  semantic structure, their embeddings are similar, producing high
  attention weight — which pulls them close in geodesic distance.

  This is gravitational lensing: information about 'cat' bends across
  the sequence to reach 'purred', shortcutting 6 positions of flat space.

  The attention matrix IS the metric tensor. It defines what 'close' means
  in the information geometry of the transformer. And just like in GR,
  the geometry is dynamic — it changes with the content (mass-energy)
  of the sequence itself.
""")


if __name__ == "__main__":
    main()
