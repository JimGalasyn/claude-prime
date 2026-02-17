# Williamson Pivot Field Simulation

Implements the extended Maxwell equations from John Williamson's
"A new theory of light and matter" (Frontiers of Fundamental Physics 14, 2014).

## The Physics

Standard Maxwell equations are extended with a scalar **pivot field P**:

```
∂B/∂t = -∇×E              (Faraday, unchanged)
∂E/∂t = ∇×B - ∇P          (Ampere + pivot gradient)
∂P/∂t = -∇·E              (pivot evolution — NEW)
```

The energy-momentum density becomes:

```
M = ½(E² + B² + P²) + (E×B + P·E)
```

- **P²** = rest mass-energy density (mass from confined EM energy)
- **P·E** = redirects momentum toward E → enables closed flows → confinement
- Setting P=0 recovers standard Maxwell equations exactly

## Python Simulation

```bash
# Compare pivot vs standard Maxwell
python3 simulations/williamson_pivot.py --mode compare --steps 500 --save

# Animate with different initial conditions
python3 simulations/williamson_pivot.py --mode animate --init gaussian
python3 simulations/williamson_pivot.py --mode animate --init circular
python3 simulations/williamson_pivot.py --mode animate --init vortex
```

Requirements: `numpy`, `matplotlib`

## ShaderToy (GPU Real-Time)

The file `shadertoy_pivot.glsl` contains GLSL shaders for real-time visualization
on [ShaderToy](https://www.shadertoy.com/).

### Setup

1. Go to https://www.shadertoy.com/new
2. Click **"+ Add Buffer"** to create **Buffer A**
3. In **Buffer A** tab:
   - Paste the code from the `BUFFER A` section of `shadertoy_pivot.glsl`
   - Set **iChannel0** → **Buffer A** (click the iChannel0 box, select "Buffer A" — this creates the self-feedback loop)
4. In **Image** tab:
   - Paste the code from the `IMAGE` section of `shadertoy_pivot.glsl`
   - Set **iChannel0** → **Buffer A**
5. Press play

### Visualization

- **Red**: EM energy density ½(E² + B²)
- **Green**: Pivot energy ½P² (rest mass)
- **Blue**: Momentum density |E×B + P·E|
- **Click**: Add an EM pulse at cursor position

## References

- Williamson, J.G. "A new theory of light and matter", PoS (FFP14) 2014
- Williamson, J.G. & van der Mark, M.B. "Is the electron a photon with toroidal topology?", Ann. Fond. L. de Broglie 22, 133 (1997)
