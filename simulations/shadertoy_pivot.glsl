// ============================================================
// Williamson Pivot Field — ShaderToy Implementation
// ============================================================
//
// Implements the extended Maxwell equations from:
// "A new theory of light and matter" (J.G. Williamson, FFP14, 2014)
//
//   ∂B/∂t = -∇×E                (Faraday, unchanged)
//   ∂E/∂t = ∇×B - ∇P           (Ampere + pivot gradient)
//   ∂P/∂t = -∇·E               (pivot evolution — NEW)
//
// Energy-momentum: M = ½(E²+B²+P²) + (E×B + P·E)
//   P² = rest mass-energy density
//   P·E = momentum redirection → enables closed flows → confinement
//
// Setup in ShaderToy:
//   Buffer A: FDTD field storage (Ex, Ey, Bz, P) in RGBA
//   Image:    Visualization (energy density, pivot, or composite)
//
// Controls:
//   Mouse click: Add EM pulse at cursor position
//   Time:        Auto-initializes with circular pulse
//
// ============================================================

// ===================== BUFFER A ==============================
// This buffer stores and updates the electromagnetic fields.
// Each pixel = one grid cell. RGBA = (Ex, Ey, Bz, P)
//
// Paste this into "Buffer A" tab in ShaderToy.
// Set Buffer A's iChannel0 to itself (feedback).
// ============================================================

// Buffer A: Field update (paste into Buffer A tab)
// iChannel0 = Buffer A (self-feedback)

/*
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 res = iResolution.xy;
    vec2 dx = vec2(1.0, 0.0);
    vec2 dy = vec2(0.0, 1.0);

    float dt = 0.4;  // Courant number (< 1/sqrt(2) for stability)

    // Read current fields: (Ex, Ey, Bz, P)
    vec4 C  = texture(iChannel0, fragCoord / res);
    vec4 R  = texture(iChannel0, (fragCoord + dx) / res);
    vec4 L  = texture(iChannel0, (fragCoord - dx) / res);
    vec4 U  = texture(iChannel0, (fragCoord + dy) / res);
    vec4 D  = texture(iChannel0, (fragCoord - dy) / res);

    float Ex = C.x, Ey = C.y, Bz = C.z, P = C.w;

    // Finite differences
    float dEy_dx = (R.y - L.y) * 0.5;
    float dEx_dy = (U.x - D.x) * 0.5;
    float dBz_dx = (R.z - L.z) * 0.5;
    float dBz_dy = (U.z - D.z) * 0.5;
    float dP_dx  = (R.w - L.w) * 0.5;
    float dP_dy  = (U.w - D.w) * 0.5;
    float dEx_dx = (R.x - L.x) * 0.5;
    float dEy_dy = (U.y - D.y) * 0.5;

    // Williamson extended Maxwell equations:
    // ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
    Bz -= dt * (dEy_dx - dEx_dy);

    // ∂Ex/∂t = ∂Bz/∂y - ∂P/∂x
    Ex += dt * (dBz_dy - dP_dx);

    // ∂Ey/∂t = -∂Bz/∂x - ∂P/∂y
    Ey += dt * (-dBz_dx - dP_dy);

    // ∂P/∂t = -(∂Ex/∂x + ∂Ey/∂y)
    P -= dt * (dEx_dx + dEy_dy);

    // Gentle damping at boundaries
    float border = 20.0;
    float bx = min(fragCoord.x, res.x - fragCoord.x) / border;
    float by = min(fragCoord.y, res.y - fragCoord.y) / border;
    float damp = clamp(min(bx, by), 0.0, 1.0);
    Ex *= damp; Ey *= damp; Bz *= damp; P *= damp;

    // Initialize on first frame or reset
    if (iFrame < 2) {
        vec2 center = res * 0.5;
        vec2 r = fragCoord - center;
        float R_dist = length(r);
        float theta = atan(r.y, r.x);

        // Circular pulse: photon propagating around a ring
        float ring_radius = min(res.x, res.y) * 0.15;
        float ring_width = ring_radius * 0.25;
        float ring = exp(-pow(R_dist - ring_radius, 2.0) / (2.0 * ring_width * ring_width));

        float k0 = 6.2832 / ring_radius * 3.0;  // ~3 wavelengths around ring
        float phase = k0 * ring_radius * theta;

        // Radial E field (creates ∇·E ≠ 0, seeds pivot)
        float Er = ring * cos(phase);
        Ex = Er * r.x / max(R_dist, 1.0);
        Ey = Er * r.y / max(R_dist, 1.0);
        Bz = ring * sin(phase) * 0.5;
        P = 0.0;
    }

    // Mouse interaction: add pulse at click location
    if (iMouse.z > 0.0) {
        vec2 mpos = iMouse.xy;
        vec2 r = fragCoord - mpos;
        float d = length(r);
        float pulse = 0.3 * exp(-d * d / 200.0);
        // Radial pulse (divergent E)
        Ex += pulse * r.x / max(d, 1.0);
        Ey += pulse * r.y / max(d, 1.0);
    }

    fragColor = vec4(Ex, Ey, Bz, P);
}
*/

// ===================== IMAGE =================================
// This is the visualization shader. Paste into "Image" tab.
// Set iChannel0 = Buffer A.
// ============================================================

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 res = iResolution.xy;

    // Read fields from Buffer A
    vec4 fields = texture(iChannel0, uv);
    float Ex = fields.x;
    float Ey = fields.y;
    float Bz = fields.z;
    float P  = fields.w;

    // Energy density: ½(E² + B² + P²)
    float em_energy = 0.5 * (Ex*Ex + Ey*Ey + Bz*Bz);
    float pivot_energy = 0.5 * P * P;
    float total_energy = em_energy + pivot_energy;

    // Momentum density components
    float Sx = Ey * Bz + P * Ex;  // Standard Poynting + pivot redirection
    float Sy = -Ex * Bz + P * Ey;
    float mom = sqrt(Sx*Sx + Sy*Sy);

    // === Visualization mode (cycle with time or pick one) ===

    // Composite visualization:
    // Red channel:   EM energy density
    // Green channel:  Pivot energy (rest mass)
    // Blue channel:   Momentum density

    float scale_em = 15.0;
    float scale_p = 30.0;
    float scale_mom = 20.0;

    vec3 color = vec3(
        clamp(em_energy * scale_em, 0.0, 1.0),
        clamp(pivot_energy * scale_p, 0.0, 1.0),
        clamp(mom * scale_mom, 0.0, 1.0)
    );

    // Add subtle field direction indicator (E field as faint lines)
    float E_mag = sqrt(Ex*Ex + Ey*Ey);
    if (E_mag > 0.001) {
        // LIC-like effect: brightness varies with field alignment to grid
        float angle = atan(Ey, Ex);
        float pattern = 0.5 + 0.5 * sin(fragCoord.x * cos(angle) + fragCoord.y * sin(angle));
        color += vec3(0.05) * pattern * clamp(E_mag * 5.0, 0.0, 0.3);
    }

    // Pivot field sign indicator (faint blue/red overlay for P polarity)
    color += vec3(max(P * 2.0, 0.0), 0.0, max(-P * 2.0, 0.0)) * 0.1;

    // Gamma correction for better visibility
    color = pow(color, vec3(0.7));

    fragColor = vec4(color, 1.0);
}
