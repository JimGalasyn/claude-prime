# Data Directory

This directory contains all source datasets used by the research scripts.

## Tracked datasets (included in git)

| Dataset | Directory | Size | Description |
|---------|-----------|------|-------------|
| NUFORC | `nuforc/` | 17 MB | All NUFORC sighting records (`All-Nuforc-Records.csv`) |
| IMO VMDB | `imo/` | 67 MB | IMO Visual Meteor Database rate files (2000-2021) |
| RMOB | `rmob/` | ~440 KB | Radio Meteor Observation Bureau survey data (2000-2020) |

## Gitignored datasets (download manually)

These datasets are too large for git. Download them and place in the appropriate subdirectory.

### WGLC Lightning Data (`wglc/`)

**File:** `wglc_timeseries_30m_daily.nc` (1.8 GB NetCDF)

Download from the World Wide Lightning Location Network / Global Lightning Climatology project:
- Source: https://doi.org/10.5065/2rc7-ww48
- Place the file at: `data/wglc/wglc_timeseries_30m_daily.nc`

### NOAA Storm Events (`storm_events/`)

**Files:** `StormEvents_details-ftp_v1.0_d{YYYY}_*.csv` for years 2000-2020 (1.1 GB total)

Download from NOAA Storm Events Database:
- Source: https://www.ncdc.noaa.gov/stormevents/ftp.jsp
- Download the "details" CSV for each year 2000-2020
- Place all CSV files in: `data/storm_events/`
