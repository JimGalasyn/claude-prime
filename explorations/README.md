# Explorations

Code that accompanies theoretical writing about connections between physics, computer science, and AI.

These are standalone, runnable Python scripts that demonstrate deep structural analogies across domains.

## attention-as-spacetime.py

Demonstrates how transformer attention weights create a "metric tensor" analogous to curved spacetime in general relativity. High-attention tokens warp the information geometry, pulling distant tokens closer — just as massive objects curve spacetime, making geodesics converge.

Run: `python3 attention-as-spacetime.py`

Accompanies the essay: *"Attention Is Curved Spacetime"*

## vector-clocks-as-lightcones.py

Shows that Lamport vector clocks in distributed systems define the same causal structure as light cones in special relativity. Both answer: "which events can causally influence which other events?" — and both reveal that some pairs of events have no definite ordering.

Run: `python3 vector-clocks-as-lightcones.py`

Accompanies the essay: *"Vector Clocks Are Light Cones"*
