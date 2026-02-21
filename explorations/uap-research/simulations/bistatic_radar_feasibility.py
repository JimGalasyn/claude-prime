#!/usr/bin/env python3
"""
Back-of-the-envelope feasibility calculation:
Bistatic radar using HAARP as transmitter and a giant airborne triangle as receiver.

Theory (Jim Galasyn / Tom Mahood):
  - HAARP illuminates the ionosphere with HF energy
  - A giant triangular dirigible/balloon acts as a receiving antenna
  - Re-entry vehicles passing between the ionospheric "scattering volume" and
    the triangle receiver cast detectable shadows (forward-scatter bistatic radar)
  - This constitutes an ABM detection system that would violate the ABM Treaty

We evaluate:
  1. HAARP transmitted power and ionospheric interaction
  2. Forward-scatter cross section of an RV at relevant frequencies
  3. Power budget at the triangle receiver
  4. Signal-to-noise ratio
  5. Whether shadow detection is feasible
"""

import numpy as np

# =============================================================================
# Physical constants
# =============================================================================
c = 3e8          # speed of light, m/s
k_B = 1.38e-23  # Boltzmann constant, J/K
PI = np.pi

# =============================================================================
# HAARP parameters
# =============================================================================
haarp_power_W = 3.6e6          # transmitter power, 3.6 MW
haarp_antenna_gain_dBi = 31.0  # phased array gain ~31 dBi at optimal frequency
haarp_gain_linear = 10**(haarp_antenna_gain_dBi / 10)  # ~1259
haarp_ERP_W = haarp_power_W * haarp_gain_linear  # ~4.5 GW ERP

# HAARP frequency range: 2.8 - 10 MHz
# Most effective ionospheric heating around 3-7 MHz
haarp_freq_Hz = 5e6  # 5 MHz nominal
haarp_wavelength_m = c / haarp_freq_Hz  # 60 m

print("=" * 70)
print("BISTATIC RADAR FEASIBILITY: HAARP + TRIANGLE RECEIVER")
print("=" * 70)
print(f"\n--- HAARP Transmitter ---")
print(f"  Transmitter power:     {haarp_power_W/1e6:.1f} MW")
print(f"  Antenna gain:          {haarp_antenna_gain_dBi:.0f} dBi ({haarp_gain_linear:.0f}x)")
print(f"  Effective radiated power: {haarp_ERP_W/1e9:.1f} GW")
print(f"  Frequency:             {haarp_freq_Hz/1e6:.1f} MHz")
print(f"  Wavelength:            {haarp_wavelength_m:.1f} m")

# =============================================================================
# Geometry
# =============================================================================
# HAARP is in Gakona, Alaska (62.4°N, 145.2°W)
# Triangle operates over US Southwest (~35°N, 115°W)
# Great circle distance: ~4,200 km
# But the signal path is: HAARP -> ionosphere -> scattering -> triangle
# Ionosphere altitude: ~100-350 km (F-layer for HF reflection)

R_haarp_to_iono_km = 350          # vertical, but HAARP beam is directed up
R_iono_to_triangle_km = 4200      # rough ground distance, ionospheric hop
R_haarp_to_target_km = 2000       # RV might be anywhere in between (midcourse/terminal)

# For bistatic: transmitter-to-target and target-to-receiver distances
# In this concept, HAARP -> ionosphere -> scattered field illuminates a region
# RV passes through this illuminated region near the triangle
# The triangle sees a shadow/perturbation

# Key insight: this is more like a "radar fence" or "forward scatter barrier"
# The ionosphere acts as a giant diffuse secondary source

iono_height_km = 300  # F-layer
print(f"\n--- Geometry ---")
print(f"  HAARP to ionosphere:   {R_haarp_to_iono_km} km (vertical)")
print(f"  Ionosphere to triangle: {R_iono_to_triangle_km} km (1-2 hop skip)")
print(f"  Ionospheric layer:     {iono_height_km} km (F-layer)")

# =============================================================================
# Triangle receiver parameters
# =============================================================================
# Witness estimates: 300 ft to 1 mile across
# Let's parametrize
triangle_sizes_m = [100, 300, 500, 1000, 1600]  # 100m to 1 mile

print(f"\n--- Triangle Receiver Sizes ---")
for size in triangle_sizes_m:
    area = (np.sqrt(3) / 4) * size**2  # equilateral triangle area
    # As receiving antenna, effective aperture ≈ physical area * efficiency
    # For a filled aperture antenna at resonance, A_eff ≈ A_physical * η
    # But at 60m wavelength, a 300m triangle is only ~5 wavelengths across
    # Antenna gain G = 4π A_eff / λ²
    eta = 0.5  # aperture efficiency
    A_eff = area * eta
    G_rx = 4 * PI * A_eff / haarp_wavelength_m**2
    G_rx_dBi = 10 * np.log10(G_rx)
    print(f"  {size:5.0f} m ({size*3.28:.0f} ft): area = {area:.0f} m², "
          f"A_eff = {A_eff:.0f} m², gain = {G_rx_dBi:.1f} dBi")

# =============================================================================
# Ionospheric scattering / reflection
# =============================================================================
# HAARP heats the ionosphere, creating enhanced electron density perturbations
# These scatter/reflect HF energy. The ionosphere can act as:
# 1. A specular reflector (for frequencies below foF2, the critical frequency)
#    - foF2 typically 3-12 MHz depending on conditions
# 2. A diffuse scatterer for stimulated emissions
#
# For a simple model: assume some fraction of HAARP's ERP is scattered
# toward the ground over a large area. This is the "illumination" that
# the RV would shadow.

# Ionospheric reflection coefficient for HF near critical frequency
# Can be quite high (0.1 to 0.9 depending on frequency vs foF2)
iono_reflection_coeff = 0.3  # conservative

# The reflected/scattered power spreads over a large area on the ground
# For a 1-hop skip at 300 km height, the illuminated footprint is roughly
# an annular zone. For our purposes, assume the power is spread over a
# circular area of radius ~500 km (conservative)
iono_footprint_radius_km = 500
iono_footprint_area_m2 = PI * (iono_footprint_radius_km * 1e3)**2

# Power flux density at ground from ionospheric reflection
P_scattered_W = haarp_ERP_W * iono_reflection_coeff
# Account for free-space spreading from ionosphere back to ground
R_iono_ground_m = np.sqrt((iono_height_km * 1e3)**2 + (R_iono_to_triangle_km * 1e3 / 2)**2)
# Flux at ground (treating iono as isotropic re-radiator over hemisphere)
# More realistically, specular reflection focuses energy, so use footprint model
flux_at_ground_W_per_m2 = P_scattered_W / iono_footprint_area_m2

print(f"\n--- Ionospheric Illumination ---")
print(f"  Reflection coefficient: {iono_reflection_coeff}")
print(f"  Scattered power:       {P_scattered_W/1e9:.2f} GW")
print(f"  Footprint radius:      {iono_footprint_radius_km} km")
print(f"  Footprint area:        {iono_footprint_area_m2:.2e} m²")
print(f"  Flux at ground:        {flux_at_ground_W_per_m2:.2e} W/m²")
print(f"  Flux (dBW/m²):         {10*np.log10(flux_at_ground_W_per_m2):.1f} dBW/m²")

# =============================================================================
# RV shadow calculation
# =============================================================================
# An RV passing through this illuminated field at altitude will cast a "shadow"
# The shadow is a diffraction effect (Babinet's principle)
#
# For an object of size d in a field of wavelength λ:
# - If d >> λ: geometric shadow, sharp edges, easy to detect
# - If d ~ λ: Fresnel diffraction, partial shadow
# - If d << λ: Rayleigh regime, negligible shadow (wave diffracts around object)
#
# RV physical dimensions: cone ~0.5-2 m diameter, ~1-3 m length
# At 5 MHz (λ = 60 m): RV is ~1/30 to 1/60 of a wavelength
# This is DEEP in the Rayleigh regime — very bad for shadow detection!
#
# However, there are important considerations:
# 1. RVs have a plasma wake/sheath during re-entry that can be much larger
# 2. HAARP can also generate ELF/VLF stimulated emissions (very long λ, worse)
# 3. Could use harmonics or stimulated Brillouin scatter at higher frequencies?

rv_diameter_m = 1.0       # physical RV cone diameter
rv_length_m = 2.0         # physical RV length
rv_cross_section_m2 = PI * (rv_diameter_m/2)**2  # ~0.79 m²

# Plasma sheath during re-entry can extend several meters
# and has high electron density → large radar cross section at HF
plasma_sheath_diameter_m = 10.0  # reasonable for re-entry plasma wake
plasma_cross_section_m2 = PI * (plasma_sheath_diameter_m/2)**2

print(f"\n--- Re-entry Vehicle ---")
print(f"  Physical diameter:     {rv_diameter_m:.1f} m")
print(f"  Physical cross-section: {rv_cross_section_m2:.2f} m²")
print(f"  Size parameter (d/λ):  {rv_diameter_m/haarp_wavelength_m:.4f} (Rayleigh regime!)")
print(f"  Plasma sheath diameter: {plasma_sheath_diameter_m:.0f} m")
print(f"  Plasma cross-section:  {plasma_cross_section_m2:.1f} m²")
print(f"  Plasma size param (d/λ): {plasma_sheath_diameter_m/haarp_wavelength_m:.3f}")

# Forward scatter cross section (Babinet's principle)
# σ_fs = 4π A² / λ²  (for objects >> λ)
# For objects << λ, this formula doesn't apply — use Rayleigh scattering
# σ_Rayleigh ∝ (d/λ)^4 * geometric_cross_section — very small

# For the physical RV:
sigma_fs_rv = 4 * PI * rv_cross_section_m2**2 / haarp_wavelength_m**2
# For the plasma sheath:
sigma_fs_plasma = 4 * PI * plasma_cross_section_m2**2 / haarp_wavelength_m**2

# Rayleigh regime correction: multiply by (2πd/λ)^4 when d << λ
rayleigh_factor_rv = (2 * PI * rv_diameter_m / haarp_wavelength_m)**4
rayleigh_factor_plasma = (2 * PI * plasma_sheath_diameter_m / haarp_wavelength_m)**4

sigma_rv_actual = rv_cross_section_m2 * rayleigh_factor_rv
sigma_plasma_actual = plasma_cross_section_m2 * rayleigh_factor_plasma

print(f"\n--- Forward Scatter Cross Sections ---")
print(f"  RV (Babinet, if d>>λ):     {sigma_fs_rv:.4f} m²")
print(f"  Plasma (Babinet, if d>>λ): {sigma_fs_plasma:.1f} m²")
print(f"  Rayleigh correction (RV):   ×{rayleigh_factor_rv:.2e}")
print(f"  Rayleigh correction (plasma): ×{rayleigh_factor_plasma:.2e}")
print(f"  RV effective σ (Rayleigh):  {sigma_rv_actual:.2e} m²")
print(f"  Plasma effective σ (Rayleigh): {sigma_plasma_actual:.2e} m²")

# =============================================================================
# Signal calculation: shadow depth at the triangle
# =============================================================================
# The "shadow" is a reduction in received flux when the RV is between the
# ionospheric source and the receiver.
#
# For an object in the Rayleigh regime, the fractional shadow depth is tiny.
# More precisely, the shadow is the missing forward-scattered power.
#
# At the receiver, the RV removes power from the incident beam proportional
# to its extinction cross section (≈ scattering + absorption cross section).
# For a conducting object in Rayleigh regime:
# σ_ext ≈ σ_scatter + σ_abs
# For a plasma (re-entry sheath), absorption can be significant.

# Alternative approach: treat as a simple occultation
# Fractional power blocked = σ_ext / A_receiver (if RV is close to receiver)
# Or more precisely, the Fresnel zone analysis...

# Fresnel zone radius at distance d from RV to receiver:
# r_F = sqrt(λ * d)
# If receiver is at 20 km altitude and RV at 100 km altitude:
# separation ~ 80 km
rv_altitude_km = 100  # midcourse/early terminal phase
triangle_altitude_km = 20  # high-altitude dirigible
separation_km = rv_altitude_km - triangle_altitude_km

fresnel_radius_m = np.sqrt(haarp_wavelength_m * separation_km * 1e3)

print(f"\n--- Shadow / Fresnel Analysis ---")
print(f"  RV altitude:           {rv_altitude_km} km")
print(f"  Triangle altitude:     {triangle_altitude_km} km")
print(f"  Separation:            {separation_km} km")
print(f"  First Fresnel radius:  {fresnel_radius_m:.1f} m")
print(f"  RV diameter / Fresnel: {rv_diameter_m/fresnel_radius_m:.4f}")
print(f"  Plasma diam / Fresnel: {plasma_sheath_diameter_m/fresnel_radius_m:.4f}")

# The shadow depth (fractional reduction in signal) is approximately:
# δ ≈ (A_object / A_fresnel)² for objects << Fresnel zone
# This is because the shadow fills in via diffraction
A_fresnel = PI * fresnel_radius_m**2
shadow_depth_rv = (rv_cross_section_m2 / A_fresnel)**2
shadow_depth_plasma = (plasma_cross_section_m2 / A_fresnel)**2

print(f"  Fresnel zone area:     {A_fresnel:.0f} m²")
print(f"  Shadow depth (RV):     {shadow_depth_rv:.2e} ({10*np.log10(shadow_depth_rv):.1f} dB)")
print(f"  Shadow depth (plasma): {shadow_depth_plasma:.2e} ({10*np.log10(shadow_depth_plasma):.1f} dB)")

# =============================================================================
# Power at the triangle receiver
# =============================================================================
print(f"\n--- Power Budget ---")
print(f"{'Triangle size (m)':<20} {'P_received (dBW)':<20} {'Shadow ΔP RV (dBW)':<22} {'Shadow ΔP plasma (dBW)'}")
print("-" * 85)

for size in triangle_sizes_m:
    area = (np.sqrt(3) / 4) * size**2
    eta = 0.5
    A_eff = area * eta
    P_rx = flux_at_ground_W_per_m2 * A_eff
    P_rx_dBW = 10 * np.log10(P_rx)

    # Shadow signal = P_rx * shadow_depth
    shadow_P_rv = P_rx * shadow_depth_rv
    shadow_P_plasma = P_rx * shadow_depth_plasma

    shadow_P_rv_dBW = 10 * np.log10(shadow_P_rv) if shadow_P_rv > 0 else -999
    shadow_P_plasma_dBW = 10 * np.log10(shadow_P_plasma) if shadow_P_plasma > 0 else -999

    print(f"  {size:<18.0f} {P_rx_dBW:<20.1f} {shadow_P_rv_dBW:<22.1f} {shadow_P_plasma_dBW:.1f}")

# =============================================================================
# Noise floor
# =============================================================================
# Receiver noise at HF frequencies is dominated by galactic/atmospheric noise
# At 5 MHz, noise temperature is very high: ~10^5 to 10^6 K
# (external noise dominates over receiver noise)
T_noise_K = 1e5  # conservative for 5 MHz, quiet conditions
bandwidth_Hz = 1000  # 1 kHz processing bandwidth (matched to RV transit time)

# RV transit time across Fresnel zone:
rv_velocity_m_s = 7000  # ~7 km/s for ICBM RV
transit_time_s = 2 * fresnel_radius_m / rv_velocity_m_s  # through Fresnel zone
optimal_bandwidth_Hz = 1.0 / transit_time_s

noise_power_W = k_B * T_noise_K * bandwidth_Hz
noise_power_optimal_W = k_B * T_noise_K * optimal_bandwidth_Hz

print(f"\n--- Noise Floor ---")
print(f"  Noise temperature:     {T_noise_K:.0e} K (5 MHz galactic+atmospheric)")
print(f"  Processing bandwidth:  {bandwidth_Hz} Hz")
print(f"  Noise power (1 kHz BW): {10*np.log10(noise_power_W):.1f} dBW")
print(f"  RV velocity:           {rv_velocity_m_s/1000:.0f} km/s")
print(f"  Transit time:          {transit_time_s:.3f} s")
print(f"  Optimal bandwidth:     {optimal_bandwidth_Hz:.1f} Hz")
print(f"  Noise power (optimal): {10*np.log10(noise_power_optimal_W):.1f} dBW")

# =============================================================================
# SNR calculation
# =============================================================================
print(f"\n--- SNR Analysis (optimal bandwidth) ---")
print(f"{'Triangle (m)':<15} {'P_rx (dBW)':<15} {'Shadow_plasma (dBW)':<22} {'Noise (dBW)':<15} {'SNR (dB)'}")
print("-" * 80)

for size in triangle_sizes_m:
    area = (np.sqrt(3) / 4) * size**2
    A_eff = area * 0.5
    P_rx = flux_at_ground_W_per_m2 * A_eff
    shadow_P = P_rx * shadow_depth_plasma
    snr = shadow_P / noise_power_optimal_W
    snr_dB = 10 * np.log10(snr) if snr > 0 else -999

    print(f"  {size:<13.0f} {10*np.log10(P_rx):<15.1f} {10*np.log10(shadow_P):<22.1f} "
          f"{10*np.log10(noise_power_optimal_W):<15.1f} {snr_dB:.1f}")

# =============================================================================
# What frequency WOULD you need?
# =============================================================================
print(f"\n{'='*70}")
print("FREQUENCY ANALYSIS: What would make this work?")
print(f"{'='*70}")
print(f"\nFor shadow detection, you need the target to be comparable to or larger")
print(f"than the wavelength. Working backwards from the plasma sheath:")
print(f"  Plasma sheath: {plasma_sheath_diameter_m} m → need λ ≤ {plasma_sheath_diameter_m} m → f ≥ {c/plasma_sheath_diameter_m/1e6:.0f} MHz")
print(f"  Physical RV:   {rv_diameter_m} m → need λ ≤ {rv_diameter_m} m → f ≥ {c/rv_diameter_m/1e6:.0f} MHz")

# What about HAARP's stimulated emissions?
# HAARP can generate:
# - ELF (3-30 Hz) — way too long wavelength
# - VLF (3-30 kHz) — still too long
# - Stimulated electromagnetic emissions (SEE) near the pump frequency
# - Parametric decay instabilities can generate sidebands
# - But all of these are at or below the pump frequency (≤ 10 MHz)

# HOWEVER: What if we consider the IONOSPHERE as the scattering screen,
# not direct shadow detection? Then we're doing something different:
# detecting the RV's perturbation of the ionospheric reflected signal.

print(f"\n{'='*70}")
print("ALTERNATIVE: PERTURBATION DETECTION (not shadow)")
print(f"{'='*70}")
print(f"""
Instead of detecting a geometric shadow, consider:
The ionosphere creates a complex interference pattern on the ground
(like laser speckle). An RV with a plasma sheath is a CONDUCTING OBJECT
moving through the ionospheric HF field. It doesn't just block — it
RE-SCATTERS the field, creating a detectable doppler-shifted signal.

Key advantages:
1. Even a sub-wavelength conductor scatters (Rayleigh regime, but still)
2. The RV moves at ~7 km/s → Doppler shift: Δf = 2v/λ sin(θ)
   At 5 MHz, λ = 60 m: Δf ≈ {2*rv_velocity_m_s/haarp_wavelength_m:.0f} Hz
3. The plasma sheath is a BETTER scatterer than metal at HF
   (plasma frequency of re-entry sheath can exceed HF frequencies)
4. The scattered signal is doppler-shifted and can be separated from
   the static ionospheric reflection (clutter rejection!)
""")

# Doppler-shifted scattering from RV
doppler_shift_Hz = 2 * rv_velocity_m_s / haarp_wavelength_m  # max
print(f"  Max Doppler shift:     {doppler_shift_Hz:.0f} Hz")
print(f"  This is easily resolvable with FFT processing!")

# RV as Rayleigh scatterer of incident HF field
# For a conducting sphere of radius a << λ:
# σ_Rayleigh = (128/3) π^5 (a/λ)^4 a² ∝ a^6/λ^4
# For plasma sheath (a = 5 m, λ = 60 m):
a_plasma = plasma_sheath_diameter_m / 2
sigma_rayleigh = (128/3) * PI**5 * (a_plasma/haarp_wavelength_m)**4 * a_plasma**2

print(f"\n  Rayleigh scattering cross section (plasma sheath):")
print(f"    σ = {sigma_rayleigh:.4f} m²  ({10*np.log10(sigma_rayleigh):.1f} dBsm)")

# Now: the scattered (Doppler-shifted) power from RV
# Incident power density at RV altitude (from ionospheric reflection)
# Use same flux_at_ground approximation (RV is at ~100 km, not ground, but similar order)
flux_at_rv = flux_at_ground_W_per_m2 * 0.5  # rough: less at altitude due to geometry

P_scattered_by_rv = flux_at_rv * sigma_rayleigh  # isotropic scatter
# Power density of scattered signal at triangle (distance = separation)
flux_scattered_at_triangle = P_scattered_by_rv / (4 * PI * (separation_km * 1e3)**2)

print(f"\n  Incident flux at RV:   {flux_at_rv:.2e} W/m²")
print(f"  Power scattered by RV: {P_scattered_by_rv:.2e} W (isotropic)")
print(f"  Flux at triangle:      {flux_scattered_at_triangle:.2e} W/m²")

# Power received by triangle from RV scatter
print(f"\n  {'Triangle (m)':<15} {'P_scatter_rx (dBW)':<20} {'Doppler BW noise (dBW)':<25} {'SNR (dB)'}")
print("  " + "-" * 75)

doppler_bw = 10  # Hz, narrow filter around Doppler peak
noise_doppler = k_B * T_noise_K * doppler_bw

for size in triangle_sizes_m:
    area = (np.sqrt(3) / 4) * size**2
    A_eff = area * 0.5
    P_scatter_rx = flux_scattered_at_triangle * A_eff
    snr = P_scatter_rx / noise_doppler
    snr_dB = 10 * np.log10(snr) if snr > 0 else -999
    print(f"  {size:<15.0f} {10*np.log10(P_scatter_rx):<20.1f} "
          f"{10*np.log10(noise_doppler):<25.1f} {snr_dB:.1f}")

# =============================================================================
# Integration gain
# =============================================================================
print(f"\n--- Integration Gain ---")
print(f"  RV transit through detection zone: ~{2*500/rv_velocity_m_s*1000:.0f} ms to {2*50000/rv_velocity_m_s:.0f} s")
print(f"  With coherent integration over {transit_time_s:.2f} s at {doppler_bw} Hz BW:")
integration_gain = transit_time_s * doppler_bw  # rough
print(f"  Integration gain: ~{10*np.log10(integration_gain):.1f} dB")
# Over longer dwell (say 10 seconds in detection zone)
long_dwell_s = 10
integration_gain_long = long_dwell_s * doppler_bw
print(f"  Over {long_dwell_s}s dwell: ~{10*np.log10(integration_gain_long):.1f} dB additional")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"""
1. SHADOW DETECTION at HF (5 MHz, λ=60m): NOT FEASIBLE
   - RV and even plasma sheath are deep in Rayleigh regime (d << λ)
   - Shadow depth is negligible (~10⁻¹⁰ to 10⁻⁷)
   - No amount of antenna size compensates

2. DOPPLER-SHIFTED SCATTER DETECTION: MARGINAL TO FEASIBLE
   - RV plasma sheath scatters HF even in Rayleigh regime
   - Doppler shift (~230 Hz) separates RV signal from clutter
   - SNR is poor for small triangles, but improves with size
   - A 1000-1600m triangle could potentially detect at SNR ~ -20 to -10 dB
   - Coherent integration over dwell time could add 10-20 dB
   - Net: MARGINALLY DETECTABLE with 1+ km triangle

3. WHAT WOULD MAKE IT WORK BETTER:
   - Higher frequency illumination (30+ MHz gets plasma sheath above Rayleigh)
   - Larger plasma wake (re-entry vehicles in atmosphere have bigger wakes)
   - Multiple HAARP-like transmitters for better illumination
   - The triangle doesn't need to detect alone — multiple triangles or
     ground-based bistatic receivers could be part of the system

4. KEY INSIGHT: The system is more plausible as a DOPPLER FENCE than
   a SHADOW DETECTOR. The RV's enormous velocity creates a clear
   spectral signature that can be separated from static clutter.
""")
