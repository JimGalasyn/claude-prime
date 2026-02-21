# Simulations

Physics and mathematical simulations organized by research project.

## Directory Structure

```
simulations/
├── koide/          # Koide formula, Lorenz attractors, mass predictions
│   ├── output/     # Generated plots and animations
│   └── *.py
├── williamson/     # Williamson pivot field, torus topology, EM confinement
│   ├── output/     # Generated plots and animations
│   ├── *.py
│   └── shadertoy_pivot.glsl
├── nwt_fdtd.py             # Null worldtube FDTD (active — do not reorganize)
├── nwt_fdtd_kn.py          # NWT with Kerr-Newman metric
├── null_worldtube.py        # Null worldtube core simulation
├── null_worldtube_diagrams.py
└── nwt_*.png               # NWT output files
```

## Koide Simulations

Explores the Koide formula and its connection to chaotic dynamics:
- `plot_koide_lorenz_attractor.py` — Lorenz attractor with Koide mass ratios
- `acid_test_lineshapes.py` — Line shape analysis
- `quantum_koide_lineshapes.py` — Quantum Koide predictions
- `plot_attractor_bands.py`, `plot_basin_boundaries.py`, `plot_bifurcation.py` — Dynamical systems analysis
- `plot_decay_phase_space.py`, `plot_mass_asymmetry.py`, `plot_return_map.py` — Phase space studies
- `anim_annihilation.py`, `anim_phase_decay.py` — Animations

## Williamson Pivot Field

Implements the extended Maxwell equations from Williamson's "A new theory of light and matter" (FFP14, 2014).

**Core idea:** Standard Maxwell equations + scalar pivot field P:
```
∂B/∂t = -∇×E              (Faraday, unchanged)
∂E/∂t = ∇×B - ∇P          (Ampere + pivot gradient)
∂P/∂t = -∇·E              (pivot evolution — NEW)
```

- `williamson_pivot.py` / `williamson_pivot_3d.py` — Core simulations (1D and 3D)
- `williamson_torus.py` — Toroidal topology analysis
- `williamson_braided.py`, `williamson_nested.py` — Braided and nested configurations
- `williamson_nonlinear.py`, `williamson_radial.py` — Nonlinear and radial studies
- `williamson_charge_golden.py` — Charge quantization via golden ratio
- `williamson_lorentz.py` — Lorentz contraction of torus solutions
- `williamson_variational.py` — Variational approach
- `anim_torus_knots.py`, `anim_precession.py`, `anim_photon_*.py` — Animations
- `shadertoy_pivot.glsl` — GPU real-time visualization (see ShaderToy setup below)

### ShaderToy Setup

1. Go to https://www.shadertoy.com/new
2. Click **"+ Add Buffer"** to create **Buffer A**
3. In **Buffer A** tab: paste `BUFFER A` section, set iChannel0 → Buffer A
4. In **Image** tab: paste `IMAGE` section, set iChannel0 → Buffer A
5. Press play. Click to add EM pulses.

## Null Worldtube (NWT)

Active research — files remain in simulations root. FDTD implementation of null worldtube geometry.

## Requirements

```
numpy matplotlib scipy
```

## References

- Koide, Y. "New view of quark and lepton mass hierarchy", Phys. Rev. D 28, 252 (1983)
- Williamson, J.G. "A new theory of light and matter", PoS (FFP14) 2014
- Williamson, J.G. & van der Mark, M.B. "Is the electron a photon with toroidal topology?", Ann. Fond. L. de Broglie 22, 133 (1997)
