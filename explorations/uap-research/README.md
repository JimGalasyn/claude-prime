# UAP Research: Meteor-Ablated Dusty Plasma Hypothesis

## Overview

This research tests the hypothesis that some UAP (Unidentified Anomalous Phenomena) sightings — particularly fireballs, orbs, and luminous spheres — may be caused by stable dusty plasma structures formed from meteor-ablated nanoparticles in the atmosphere. The work extends findings from the UK Ministry of Defence's classified Condign Report (2000), which found a Pearson correlation of r=0.62 between meteor activity and UAP sighting rates in UK data from 1996.

The research has two main threads:
1. **UAP-Meteor Correlation** — Statistical analysis of whether UAP sighting rates track meteor shower activity
2. **Kostrov Hypothesis** — Testing whether meteor dust modulates the global electric circuit with a ~12-day settling delay

## Directory Structure

```
explorations/uap-research/
├── simulations/          # Physics simulations (vortex, plasma, scattering)
├── analysis/             # Data analysis scripts (correlation, parsing)
├── legacy/               # Superseded script versions
├── output/               # All generated plots, CSVs, JSONs
│   └── rmob_correlation/ # RMOB × NUFORC correlation figures
└── README.md             # This file
```

---

## Study 1: UAP-Meteor Temporal Correlation

### Motivation

The Condign Report found that UAP sighting rates in the UK correlated with meteor activity (r=0.62 for 1996). If UAP are meteor-ablated plasma structures, this correlation should be reproducible across longer time spans and independent datasets.

### Data Sources

| Dataset | Records | Period | Description |
|---------|---------|--------|-------------|
| **NUFORC** | ~97,000 sightings | 2000–2021 | US-based sighting reports with shape, date, location |
| **IMO VMDB** | ~5,500 observer-days | 2000–2017 | International Meteor Organization visual meteor counts |
| **RMOB** | 7,742 observer-months, 582 observers | 2000–2020 | Radio meteor echo counts (24/7, weather-independent) |

### Analyses Performed

#### v1: IMO Visual × NUFORC (`legacy/uap_meteor_correlation.py`)
- Monthly correlation of IMO visual meteor counts vs NUFORC sighting rates
- Shape-filtered analysis: "plasma-consistent" shapes (fireball, sphere, orb, etc.) vs all shapes
- Shower window analysis: ±15 day windows around 7 major showers
- **Limitation:** Geographic mismatch (US NUFORC vs global IMO), seasonal confounds

#### v2: Refined IMO × NUFORC (`analysis/uap_meteor_correlation_v2.py`)
- Addresses v1 confounds with seasonal detrending, year-by-year analysis
- Fireball-specific deep dive (most direct meteor→plasma link)
- Conditional analysis: UAP rates during high vs low meteor weeks
- Detrended Fluctuation Analysis for non-stationarity
- Shower-specific: correlates ZHR and entry velocity with UAP enhancement

#### v3: RMOB Radio × NUFORC (`analysis/rmob_nuforc_correlation.py`)
The most rigorous analysis, using radio meteor data which avoids visual-observation biases:

- **RMOB parsing** (`analysis/rmob_parser.py`): Parses 7,742 raw RMOB text files into a normalized daily meteor activity index. Per-observer normalization (each observer's daily count / their own monthly median) handles the huge variation in equipment sensitivity. IQR-based outlier rejection removes noisy readings.
- **Correlation analysis**: Raw daily, deseasonalized, monthly aggregated, and year-by-year
- **Lag analysis**: Tests lags from -3 to +14 days (meteor leads UAP)
- **Shower windows**: Enhancement factor during each major shower vs outside
- **Composition-dependent analysis**: Tests whether entry velocity (a proxy for nanoparticle production efficiency) predicts UAP enhancement. High-velocity cometary showers (Leonids 70 km/s, Eta Aquariids 65 km/s) should produce more nm-scale meteoric smoke particles (MSPs) via complete vaporization, while slow asteroidal showers (Geminids 35 km/s) produce μm-scale melt droplets that don't form plasma.
- **Within-season control**: Compares shower peak to ±30 day adjacent baseline in the same season, controlling for outdoor-activity confounds while preserving the meteor-driven signal.

### Key Results

- RMOB successfully detects 3 major showers above noise: Geminids (SNR 2.19), Quadrantids (1.82), Perseids (1.75)
- 582 unique observers across ~40 countries, mean 24.6 observers/day
- The normalized activity index centers at 1.017 (close to expected 1.0), confirming the normalization works
- Year-by-year analysis shows several individually significant positive correlations
- Lag analysis probes the mesospheric transport timescale (meteors ablate at 80–100 km altitude; nanoparticles must settle to tropospheric heights)

### Output Files

| File | Description |
|------|-------------|
| `output/rmob_correlation/rmob_nuforc_correlation.png/pdf` | 9-panel figure: time series, scatter plots, lag analysis, seasonal cycles, shower enhancement |
| `output/rmob_correlation/shower_composition_analysis.png/pdf` | 4-panel figure: entry velocity vs UAP enhancement, cometary vs asteroidal comparison |
| `output/rmob_daily_meteor_flux.csv` | Daily RMOB activity index (2000–2020) |
| `output/rmob_analysis_summary.json` | Parser statistics, shower detection results |

---

## Study 2: Kostrov Hypothesis — Meteor Dust Modulates the Global Electric Circuit

### Background

Kostrov (2020) proposed that cosmic dust (including meteor stream particles) settling through the atmosphere modulates the Global Electric Circuit (GEC) — the ~250 kV potential difference between the ionosphere and Earth's surface that drives a continuous ~1–2 pA/m² fair-weather current. The mechanism:

1. Meteor streams deposit ~100 tonnes/day of material at 80–120 km altitude
2. Ablated nanoparticles (~1–100 nm) settle through the atmosphere over ~10–15 days
3. These charged particles modify atmospheric conductivity in the mesosphere and stratosphere
4. Changed conductivity modulates the fair-weather current density
5. Enhanced current density could favor stable atmospheric plasma formation

### Tests Performed

#### Test A: RMOB Meteors × NOAA StormEvents (`analysis/kostrov_cross_correlation.py`)
- Cross-correlates daily RMOB meteor activity index with US thunderstorm event counts from NOAA StormEvents database (2000–2019)
- Tests lags 0–60 days with 1000-iteration block bootstrap significance testing (30-day blocks preserve autocorrelation)
- Examines both raw and seasonally-decomposed (anomaly) correlations
- Specifically checks Kostrov's predicted 10–15 day lag window

**Results:**
- 5,133 aligned days (2000-01-02 to 2019-12-30)
- Raw correlation peak: r=0.015 at lag 26d — **not significant** (p95 threshold: 0.061)
- Anomaly correlation peak: r=0.042 at lag 26d — **not significant** (p95 threshold: 0.056)
- Kostrov window (10–15 days): r=0.011 at lag 12d (raw), r=0.006 at lag 13d (anomaly) — **not significant**
- The StormEvents data may be too coarse (discrete event reports, US-only) to detect a subtle modulation

#### Test B: RMOB Meteors × WGLC Global Lightning (`analysis/kostrov_wglc_analysis.py`)
The stronger test, using continuous global lightning stroke density from the World Wide Lightning Location Network:
- Cross-correlates RMOB meteor activity with WGLC daily global stroke density (2010–2020, 4,018 days)
- Same lag analysis and bootstrap significance testing as Test A

**Results:**
- Raw correlation: essentially flat, peak r=-0.002 at lag 13d — **not significant**
- Anomaly correlation (seasonal cycle removed): peak r=0.166 at lag **0 days** — **significant at 95%** (threshold: 0.068)
- Short-lag window (3–7 days): r=0.099 at lag 3d — **significant**
- Kostrov window (10–15 days): r=0.021 at lag 15d — **not significant**
- Extended window (8–20 days): r=0.067 at lag 19d — borderline

**Interpretation:** The strong lag-0 anomaly correlation means that *after removing seasonal cycles*, days with unusual meteor activity also tend to have unusual lightning activity. However, the signal peaks at lag 0 (simultaneous), not at the 10–15 day delay Kostrov predicts. This could mean:
- The correlation is driven by a shared seasonal confound not fully removed by the decomposition
- There's a real but non-delayed relationship (e.g., direct ionospheric effects of meteor bombardment)
- The settling delay prediction needs refinement

### Global Electric Circuit Simulation (`simulations/global_electric_circuit_sim.py`)

A computational model integrating Kostrov's dust-modulated GEC with local atmospheric conditions to predict windows where plasma stability is favored. Combines:
- Steady-state cosmic dust influx
- Time-varying meteor stream dust with 10–15 day settling delay
- Carnegie curve diurnal variation (global thunderstorm cycle)
- Local conductivity profiles (terrain-dependent: mountain, plateau, coastal, valley)
- Plasma stability assessment by terrain type across the year

### Output Files

| File | Description |
|------|-------------|
| `output/kostrov_cross_correlation.png` | 4-panel: raw and anomaly cross-correlation, 2015 time series, annual profiles |
| `output/kostrov_cross_correlation_results.json` | Full numerical results including all lag values |
| `output/kostrov_wglc_cross_correlation.png` | 4-panel: raw and anomaly cross-correlation with WGLC data |
| `output/kostrov_wglc_results.json` | Full numerical results for WGLC analysis |
| `output/gec_simulation_overview.png` | GEC model: dust loading, current density, annual variation |
| `output/gec_plasma_stability.png` | Plasma stability ratio by terrain type across the year |

---

## Physics Simulations

### Tripolar Vortex Formation
Simulates the Dharodi et al. (2014) result: a circular vortex in strongly coupled dusty plasma spontaneously evolves into a tripolar structure through viscoelastic instability. This is physically relevant because atmospheric dusty plasma would exhibit similar vortex dynamics.

| Script | Description |
|--------|-------------|
| `simulations/tripolar_shielded.py` | **Canonical**: Shielded vortex instability → tripolar (Carton & McWilliams 1989) |
| `simulations/tripolar_2d_euler.py` | **Control case**: Classical 2D Euler for comparison |
| `simulations/tripolar_vortex_final.py` | **Best visualization**: Viscoelastic instability demo |
| `legacy/tripolar_vortex.py` → `v2` → `v3` | Earlier iterations |

### Other Simulations

| Script | Description |
|--------|-------------|
| `simulations/coulomb_crystal_md.py` | Molecular dynamics of Coulomb crystal formation |
| `simulations/multi_ball_formation.py` | Multi-ball lightning formation dynamics |
| `simulations/mie_scattering.py` | Mie scattering of meteor-ablated nanoparticles → predicts orange color |
| `simulations/bistatic_radar_feasibility.py` | Bistatic radar detection feasibility for plasma structures |
| `simulations/global_electric_circuit_sim.py` | GEC model (see Kostrov section above) |

---

## References

- UK Ministry of Defence, "Unidentified Aerial Phenomena in the UK Air Defence Region" (Project Condign), 2000
- Kostrov, A.V. et al., "On the role of cosmic dust in atmospheric electricity", 2020
- Dharodi, V.S., Patel, B.G., & Kaw, P.K., Phys. Rev. E 89, 023102 (2014) — tripolar vortex in dusty plasma
- Borovicka, J. et al., "Spectroscopy of bright fireballs", Earth Moon Planets 95, 2005
- Smirnov, B.M., "Atmospheric Electricity", 2014
- Panneerselvam, C. et al., "Fair-weather atmospheric electricity", 2009
