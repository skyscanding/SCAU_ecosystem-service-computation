# LandTrendr Disturbance + Ecosystem Services Coupling Analysis

Python pipeline that integrates **LandTrendr temporal disturbance detection** with **LULC classification**, **landscape metrics** (PyLandStats), **InVEST ecosystem services** (Carbon, Habitat Quality, SDR), and **statistical coupling analysis** (correlation, regression, SEM path analysis, Kruskal-Wallis phase comparison).

No GUI. Entirely command-line driven. Validated against a real-world Dabaoshan Mine Area ecological restoration case study.

## Table of Contents

- [Core Methodology](#core-methodology)
- [Case Study: Dabaoshan Mine Area](#case-study-dabaoshan-mine-area)
- [10-Step Pipeline Overview](#10-step-pipeline-overview)
- [Quick Start](#quick-start)
- [Step 1: LandTrendr Disturbance Raster Analysis](#step-1-landtrendr-disturbance-raster-analysis)
- [Step 2: LULC Classification Area Trends](#step-2-lulc-classification-area-trends)
- [Step 3: Landscape Metrics (PyLandStats)](#step-3-landscape-metrics-pylandstats)
- [Step 4: InVEST Carbon Storage](#step-4-invest-carbon-storage)
- [Step 5: InVEST Habitat Quality](#step-5-invest-habitat-quality)
- [Step 6: InVEST SDR Soil Erosion](#step-6-invest-sdr-soil-erosion)
- [Step 7: Merge Analysis Table](#step-7-merge-analysis-table)
- [Step 8: Correlation & Partial Correlation](#step-8-correlation--partial-correlation)
- [Step 9: OLS Regression Models](#step-9-ols-regression-models)
- [Step 10: SEM Path Analysis & Phase Comparison](#step-10-sem-path-analysis--phase-comparison)
- [Result Interpretation](#result-interpretation)
- [Full Config Reference](#full-config-reference)
- [Troubleshooting](#troubleshooting)
- [Validation Results](#validation-results)
- [Files](#files)

---

## Core Methodology

```
GEE LandTrendr (NBR time series, 2009-2024)
       ↓
  YOD / MAG / DUR / MPY rasters
       ↓
  Disturbance pixel extraction → Annual statistics, severity classification
       +
  SVM LULC classification (23 annual maps, 5 classes, 2000-2025)
       ↓
  Area trends → Transition matrices
       ↓
  PyLandStats: 8 landscape-level + 5 class-level metrics per year
       ↓
  InVEST Carbon: area × carbon pool density (Mg)
  InVEST Habitat Quality: suitability-weighted degradation model
  InVEST SDR: USLE-based soil erosion + sediment delivery
       ↓
  Unified analysis_table.csv (13 years × ~37 columns)
       ↓
  Pearson/Spearman correlations (FDR-corrected) + Partial correlation
       ↓
  OLS regression: Carbon ~ MAG + LPI + SHDI  (Adj R² = 0.778, p = 0.002)
       ↓
  SEM Path analysis: MAG → SHDI/MESH → carbon/HQ  (β = -0.320, p = 0.012)
       ↓
  Kruskal-Wallis phase comparison: 3 ecological restoration phases
```

**Why this integration?** Ecological restoration impacts are multi-dimensional. LandTrendr captures vegetation disturbance magnitude and timing from satellite time series. LULC classification tracks land cover transitions. Landscape metrics quantify spatial pattern changes. InVEST models translate land cover into ecosystem service flows. Statistical coupling tests the mechanistic hypothesis: *disturbance magnitude → landscape fragmentation → ecosystem service decline*. This pipeline automates the full chain from raw rasters to publishable statistics.

### The Three-Phase Framework

| Phase | Years | Description |
|-------|-------|-------------|
| **Degradation** | 2013-2016 | Active mining, high disturbance frequency, rapid landscape fragmentation |
| **Transition** | 2017-2020 | Policy intervention begins, partial recovery, mixed signals |
| **Consolidation** | 2021-2025 | Ecological restoration takes effect, disturbance stabilizes, ecosystem services rebound |

### Key Analytical Models

```
Regression:   Carbon ~ β₁·MAG + β₂·LPI + β₃·SHDI + ε
Path model:   SHDI ∼ MAG
              MESH ∼ MAG
              Carbon ∼ SHDI + MESH + MAG
              HQ ∼ SHDI + CONTAG + MAG
```

Each year gets its own disturbance, landscape, and ES values. The statistical models test whether disturbance magnitude predicts ecosystem service decline *through* landscape pattern mediation.

---

## Case Study: Dabaoshan Mine Area

This pipeline was developed and validated for the **Dabaoshan Mine Area** ecological restoration study in northern Guangdong Province, China.

| Parameter | Value |
|-----------|-------|
| Study area | 15.10 km² (1,509.48 ha) |
| CRS | EPSG:32649 (WGS 84 / UTM zone 49N) |
| Pixel resolution | 30 m (0.09 ha/pixel) |
| Total boundary pixels | 16,772 |
| LULC classes | 5 (water, built-up, unrestored, recovering, stable vegetation) |
| LULC period | 2000-2025 (23 annual maps, Landsat 5/7/8/9, SVM classifier) |
| LandTrendr period | 2009-2024, disturbance window 2013-2024 |
| LandTrendr parameters | Max Segments=6, MAG>200, DUR<5, Preval>300, MMU>11 |
| Disturbed pixels | 1,008 (6.01% of area, 90.7 ha) |
| InVEST models | Carbon Storage, Habitat Quality, SDR (Sediment Delivery Ratio) |
| Statistical methods | Pearson/Spearman (FDR-BH), Partial correlation, OLS, SEM (semopy), Kruskal-Wallis |

### Data Inputs

| Dataset | Source | Key Fields / Content |
|---------|--------|---------------------|
| `yod_2009_2024.tif` | GEE LandTrendr export | Year of Detection (int16) |
| `mag_2009_2024.tif` | GEE LandTrendr export | NBR magnitude (float32) |
| `dur_2009_2024.tif` | GEE LandTrendr export | Disturbance duration, years (int16) |
| `magperyear_2009_2024.tif` | GEE LandTrendr export | Magnitude per year (float32) |
| `Export_Output.shp` | Study boundary | Polygon, native raster CRS |
| `lulc_{year}.tif` × 23 | Landsat SVM classification | 5-class LULC map per year |
| `carbon_pools.csv` | InVEST input | C_above, C_below, C_soil, C_dead per LULC |
| `sensitivity.csv` | InVEST HQ input | Habitat suitability + threat sensitivity |
| DEM, erosivity, erodibility | SDR inputs | Raster layers for USLE computation |

### Key Findings (from Case Study Validation)

| Indicator | Degradation Mean | Consolidation Mean | Kruskal-Wallis |
|-----------|-----------------|--------------------|----------------|
| Carbon total (Mg) | 1,978,000 | 1,730,000 | H = 7.50, p = 0.024* |
| Habitat quality | 0.325 | 0.298 | - |
| SHDI (diversity) | 1.257 | 1.405 | - |
| NP (patch count) | 476 | 394 | - |
| Disturbance area (ha) | 11.86 | 5.91 | - |
| Mean magnitude | 0.440 | 0.533 | - |

**Top correlations**: Carbon × Habitat Quality r = 0.965 (FDR p < 0.001), SHDI × Carbon r = -0.858.

**Regression**: Carbon ~ MAG + LPI + SHDI → Adj R² = 0.778, F(3,8) = 13.82, p = 0.002.

**Path analysis**: MAG → Carbon β = -0.320 (p = 0.012*), SHDI → Carbon β = -0.487 (p < 0.001), MESH → Carbon β = 0.332 (p = 0.005).

---

## 10-Step Pipeline Overview

| Step | Name | What It Does | Input | Output | Core Tool |
|------|------|-------------|-------|--------|-----------|
| 1 | **LandTrendr Disturbance** | Load GEE rasters, clip, mask disturbances, annual stats, severity | YOD/MAG/DUR/MPY TIFFs | `landtrendr_summary.csv` | rasterio |
| 2 | **LULC Trends** | Read classified rasters, compute area per class per year, transition matrix | 23 LULC rasters | `lulc_area_trends.csv` | rasterio, numpy |
| 3 | **Landscape Metrics** | Compute landscape-level + class-level metrics via PyLandStats | LULC rasters | `landscape_metrics.csv` | pylandstats |
| 4 | **Carbon Storage** | Estimate total carbon from LULC area × carbon pool density | LULC areas + carbon pools | `carbon_summary.csv` | pandas |
| 5 | **Habitat Quality** | Compute area-weighted HQ with degradation from threat sources | LULC areas + sensitivity scores | `habitat_quality_summary.csv` | pandas, numpy |
| 6 | **SDR Soil Erosion** | Estimate USLE soil loss + sediment delivery per year | LULC areas + R/K/LS factors | `sdr_summary.csv` | pandas, numpy |
| 7 | **Merge Table** | Combine all outputs into unified analysis_table.csv + assign phases | All step CSVs | `analysis_table.csv` | pandas |
| 8 | **Correlation** | Pearson + Spearman matrices (FDR-corrected) + Partial correlation | `analysis_table.csv` | pingouin, scipy, statsmodels |
| 9 | **Regression** | OLS models: Carbon, HQ, Erosion ~ disturbance + landscape | `analysis_table.csv` | statsmodels |
| 10 | **Path Analysis** | SEM path analysis (semopy) + Kruskal-Wallis phase comparison | `analysis_table.csv` | semopy, scipy |

---

## Quick Start

### Prerequisites

- **Python 3.10+** with scientific stack
- **GEE-exported LandTrendr rasters**: YOD, MAG, DUR, MPY (GeoTIFF)
- **LULC classification rasters**: One per year, integer class codes (GeoTIFF)
- **Study boundary**: Shapefile (polygon)

```bash
# Install dependencies
pip install -r requirements.txt
```

### Verify your environment

```bash
# Check core packages
python -c "import numpy, scipy, pandas, rasterio, geopandas; print('Core OK')"

# Check statistics
python -c "import pingouin, statsmodels, semopy; print('Stats OK')"

# Check landscape
python -c "import pylandstats; print('PyLandStats OK')"
```

### 1. Prepare your data

At minimum you need:

| Layer | Format | Required Fields | Notes |
|-------|--------|-----------------|-------|
| **Study boundary** | Shapefile (.shp) | - | Single polygon, projected CRS |
| **YOD raster** | GeoTIFF | - | Year of Detection from GEE LandTrendr |
| **MAG raster** | GeoTIFF | - | NBR magnitude from GEE LandTrendr |
| **DUR raster** | GeoTIFF | - | Duration (years), reclassified |
| **MPY raster** | GeoTIFF | - | Magnitude per year |
| **LULC rasters** | GeoTIFF, one per year | - | Integer class codes (1-5) |

### 2. Create a config file

Copy `config_template.json` and fill in your data paths:

```json
{
  "output_dir": "D:/MyProject/output",
  "crs": "EPSG:32649",
  "steps": ["dist", "class", "land", "carbon", "hq", "sdr", "merge", "corr", "regress", "path"],

  "study_area": {
    "boundary_shp": "D:/data/boundary.shp",
    "pixel_area_ha": 0.09,
    "raster_res_m": 30
  },

  "landtrendr": {
    "raster_dir": "D:/data/LandTrendr_export",
    "yod_raster": "yod_2005_2025.tif",
    "mag_raster": "mag_NBR_2005_2025.tif",
    "yod_range": [2013, 2024]
  },

  "lulc": {
    "raster_dir": "D:/data/LULC_classified",
    "raster_pattern": "lulc_{year}.tif",
    "year_range": [2000, 2025]
  }
}
```

> **Field Mapping**: Use `field_mapping` to declare your dataset's actual column names. The merge step auto-detects columns; if names differ, set them explicitly.

### 3. Run

```bash
# Full pipeline (all 10 steps)
python scripts/master_pipeline.py my_config.json

# Specific steps only
python scripts/master_pipeline.py my_config.json --steps dist,merge,corr,regress,path

# Standalone: run a single step with CLI arguments
python scripts/step1_landtrendr_disturbance.py \
    --raster-dir D:/data/LT/ --boundary D:/data/boundary.shp \
    --output-dir D:/output/ --crs EPSG:32649
```

### 4. View results

All outputs land in the output directory:

| File | Contents |
|------|----------|
| `summary_statistics.csv` | Annual disturbance: Pixel Count, Area (ha), Area (km²), Mean MAG, Median MAG, Std MAG, Mean Rate, duration breakdown (Dur_1yr_ha, Dur_2yr_ha, Dur_3yr_ha), intensity classes (Low_ha, Moderate_ha, High_ha, Very_High_ha), Severe_pct |
| `lulc_area_trends.csv` | LULC area (ha) per class per year |
| `landscape_metrics.csv` | NP, PD, LPI, ED, LSI, SHDI, CONTAG, MESH per year |
| `class_metrics.csv` | Class-level metrics per LULC type per year |
| `carbon_summary.csv` | Total carbon (Mg) + density (Mg/ha) per year |
| `habitat_quality_summary.csv` | Mean habitat quality per year |
| `sdr_summary.csv` | USLE total + sediment export per year |
| `analysis_table.csv` | **Master table**: all above merged, 13 years × ~37 columns |
| `transition_matrix_2000_2025.csv` | Begin→end LULC transition matrix (ha) |
| `transition_probability_2000_2025.csv` | Transition probabilities |
| `Statistics/corr_pearson_r.csv` | Pearson correlation matrix |
| `Statistics/corr_spearman_r.csv` | Spearman correlation matrix |
| `Statistics/corr_pearson_p_fdr.csv` | All pairwise r, p_raw, p_fdr, significant flags |
| `Statistics/corr_significant.csv` | Only significant pairs (FDR α = 0.05) |
| `Statistics/partial_correlation.csv` | Partial correlation results |
| `Statistics/descriptive_stats.csv` | Mean, std, min, max, median, skew, kurtosis |
| `Statistics/regression_summary.csv` | Per-model OLS: R², Adj R², F, coefficients, p-values |
| `Statistics/path_analysis_estimates.csv` | SEM path coefficients + significance |
| `Statistics/phase_summary.csv` | 3-phase mean ± std for all variables |
| `Statistics/phase_kruskal.csv` | Kruskal-Wallis H statistics + p-values |

---

## Step 1: LandTrendr Disturbance Raster Analysis

Loads GEE-exported LandTrendr rasters (YOD, MAG, DUR, MPY), clips to study boundary, generates disturbance masks, computes annual statistics, and classifies disturbance severity.

**Script**: `scripts/step1_landtrendr_disturbance.py`

### Why This Matters

LandTrendr detects vegetation disturbance and recovery from the temporal trajectory of the Normalized Burn Ratio (NBR). The GEE implementation fits piecewise linear segments to each pixel's NBR time series (2009-2024). Disturbance is identified as a segment with negative NBR magnitude exceeding thresholds. The four exported rasters capture:

- **YOD** (Year of Detection): When the strongest disturbance occurred
- **MAG** (Magnitude): NBR drop magnitude of that disturbance
- **DUR** (Duration): Number of years the disturbance persisted
- **MPY** (Magnitude Per Year): Annualized magnitude = MAG / DUR

This step extracts only pixels within the 2013-2024 disturbance window inside the study boundary.

### GEE LandTrendr Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `maxSegments` | 6 | Maximum trajectory segments per pixel |
| `spikeThreshold` | 0.9 | Dampen spikes (>90% change) |
| `vertexCountOvershoot` | 3 | Prevent over-segmentation |
| `preventOneYearRecovery` | true | Exclude 1-year false recoveries |
| `recoveryThreshold` | 0.25 | Minimum NBR gain for recovery |
| `pvalThreshold` | 0.05 | F-test p-value for vertex addition |
| `bestModelProportion` | 0.75 | p-value scaling factor |
| `minObservationsNeeded` | 6 | Minimum clear observations |
| Magnitude filter | > 200 | NBR drop × 1000 threshold |
| Duration filter | < 5 | Maximum disturbance years |
| Preval filter | > 300 | Pre-disturbance NBR threshold |
| MMU filter | > 11 | Minimum mapping unit (pixels) |

### Input Schema

| File | Format | Required | Description |
|------|--------|----------|-------------|
| `yod_raster` | GeoTIFF (int16) | Yes | Year of Detection, 0 = no disturbance |
| `mag_raster` | GeoTIFF (float32) | Yes | NBR Magnitude |
| `dur_raster` | GeoTIFF (int16) | No | Duration reclassified (1/2/3 years) |
| `mpy_raster` | GeoTIFF (float32) | No | Magnitude per Year |
| `boundary_shp` | Shapefile | Yes | Study area polygon (reprojected to raster CRS automatically) |

**CRS handling** (Nanling convention): The boundary is reprojected to the raster's native CRS (read from YOD). All rasters are clipped in their native CRS - no warping is applied. Rasters must share the same CRS.

### Output Schema

Naming follows the Nanling analysis convention. Output file: `summary_statistics.csv`.

| Field | Type | Description |
|-------|------|-------------|
| `Year` | INT | Disturbance year (2010-2024) |
| `Pixel Count` | INT | Number of disturbed pixels |
| `Area (ha)` | FLOAT | Disturbed area (hectares) |
| `Area (km²)` | FLOAT | Disturbed area (km²) |
| `Mean MAG` | FLOAT | Mean NBR magnitude |
| `Median MAG` | FLOAT | Median NBR magnitude |
| `Std MAG` | FLOAT | Std deviation of magnitude |
| `Mean Rate` | FLOAT | Mean magnitude per year (NBR/yr) |
| `Dur_1yr_ha` | FLOAT | Area with 1-year duration |
| `Dur_2yr_ha` | FLOAT | Area with 2-year duration |
| `Dur_3yr_ha` | FLOAT | Area with ≥3-year duration |
| `Low_ha` | FLOAT | Low intensity area (MAG 0.2-0.35) |
| `Moderate_ha` | FLOAT | Moderate intensity area (MAG 0.35-0.5) |
| `High_ha` | FLOAT | High intensity area (MAG 0.5-0.65) |
| `Very_High_ha` | FLOAT | Very high intensity area (MAG >0.65) |
| `Severe_pct` | FLOAT | % of year total that is High + Very High |

### Intensity Classification (4-class, Nanling convention)

| Intensity Level | Magnitude Range | Color | Description |
|-----------------|-----------------|-------|-------------|
| **Low** | 0.20 - 0.35 | `#f1c40f` | Minor vegetation stress |
| **Moderate** | 0.35 - 0.50 | `#e67e22` | Noticeable canopy loss |
| **High** | 0.50 - 0.65 | `#e74c3c` | Major vegetation removal |
| **Very High** | 0.65+ | `#8b0000` | Complete vegetation loss |

### Severity Classification (3-class, for spatial map only)

| Severity Level | Magnitude Range | Color | Description |
|----------------|-----------------|-------|-------------|
| **Low** | 0.20 - 0.35 | `#f1c40f` | Minor stress |
| **Moderate** | 0.35 - 0.55 | `#e67e22` | Significant loss |
| **High** | 0.55+ | `#c0392b` | Severe loss |

**Note**: The 4-class intensity scheme is used for CSV export and statistical analysis. The 3-class severity scheme is used for the spatial classification map (Figure 13).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `raster_dir` | - | Directory with GEE-exported TIFFs |
| `boundary_shp` | - | Study boundary shapefile |
| `yod_range` | [2010, 2024] | Disturbance analysis window |
| `pixel_area_ha` | 0.09 | Pixel area in hectares (30m × 30m) |
| `nodata_val` | 0 | Fill value for masked pixels |
| `intensity_bins` | 4-class (see above) | Override intensity thresholds |
| `severity_bins` | 3-class (see above) | Override severity thresholds |
| `generate_figures` | true | Whether to generate 16 diagnostic figures |

### Usage

```bash
# Via master pipeline
python scripts/master_pipeline.py config.json --steps dist

# Standalone
python scripts/step1_landtrendr_disturbance.py \
    --raster-dir D:/data/LT/ \
    --boundary D:/data/boundary.shp \
    --output-dir D:/output/ \
    --crs EPSG:32649 \
    --yod-start 2013 --yod-end 2024 \
    --pixel-ha 0.09 \
    --figures
```

### Pitfalls

1. **CRS mismatch**: The script reprojects the boundary shapefile to match the raster's native CRS automatically. If the boundary CRS readout shows unexpected values, verify your shapefile projection.
2. **Empty disturbance mask**: If `disturbed.sum() == 0`, verify that `yod_range` covers years with actual disturbance. Check the unique YOD values in the loaded raster.
3. **Figure generation memory**: 16 figures at 300 dpi for large rasters can consume significant RAM. Set `generate_figures: false` in config to skip figures.
4. **GEE export filenames**: Default filenames follow Nanling convention (`yod_2009_2024.tif`, `mag_2009_2024.tif`, `dur_2009_2024.tif`, `magperyear_2009_2024.tif`). Update `yod_raster`/`mag_raster`/etc. in config if yours differ.
5. **Duration raster values**: The script expects DUR values 1, 2, 3 only. If GEE exported raw duration (1-5), reclassify before running or the cross-tabulation will produce incorrect counts.

---

## Step 2: LULC Classification Area Trends

Reads classified LULC rasters (one per year), computes area per class per year, generates inter-annual change tables and begin→end transition matrices.

**Script**: `scripts/step2_lulc_trends.py`

### Why This Matters

LULC provides the spatial template for all downstream ES models. Each InVEST model maps LULC classes to biophysical parameters (carbon pools, habitat suitability, C/P factors). Accurate LULC area time series are the foundation of every subsequent analysis step.

### Input Schema

| File | Format | Required | Description |
|------|--------|----------|-------------|
| `lulc_{year}.tif` | GeoTIFF (int) | Yes | Per-year classified raster |
| `boundary_shp` | Shapefile | Yes | Study boundary |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `Year` | INT | Year (2000-2025) |
| `area_{class}_ha` | FLOAT | Area per LULC class (ha) |
| `area_{class}_pct` | FLOAT | Area % per LULC class |
| `area_total_ha` | FLOAT | Total area within boundary |

Also outputs `transition_matrix_{y1}_{y2}.csv` and `transition_probability_{y1}_{y2}.csv`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `raster_dir` | - | Directory with LULC TIFFs |
| `raster_pattern` | `lulc_{year}.tif` | Filename pattern with `{year}` |
| `year_range` | [2000, 2025] | Start and end year |
| `classes` | {water:1, built_up:2, ...} | Class name → raster code mapping |

### Pitfalls

1. **Raster not found**: If your rasters follow a different naming convention, update `raster_pattern`. Common alternatives: `LULC_{year}_aligned.tif`, `classified_{year}.tif`.
2. **Class code mismatch**: If your LULC uses different codes (e.g., 10-50 instead of 1-5), update the `classes` dictionary in config. The area calculation will silently return 0 for unrecognized codes.

---

## Step 3: Landscape Metrics (PyLandStats)

Computes landscape-level and class-level metrics using PyLandStats for each year's LULC map. Supports Moore (8-neighbor) and von Neumann (4-neighbor) neighborhood rules.

**Script**: `scripts/step3_landscape_metrics.py`

### Why PyLandStats?

PyLandStats is the Python equivalent of FRAGSTATS, the standard in landscape ecology. It computes metrics directly from numpy arrays without requiring ArcGIS. At 30m resolution, the study area (16,772 pixels) is large enough for robust metric estimation.

### Metrics Computed

#### Landscape-Level (8 metrics)

| Metric | Abbreviation | Interpretation |
|--------|-------------|----------------|
| Number of Patches | NP | Fragmentation: higher = more fragmented |
| Patch Density | PD | Patches per 100 ha |
| Largest Patch Index | LPI | Dominance: % area in largest patch |
| Edge Density | ED | Edge length per ha |
| Landscape Shape Index | LSI | Shape complexity |
| Shannon Diversity Index | SHDI | Class diversity: higher = more even |
| Contagion | CONTAG | Aggregation: higher = more clumped |
| Effective Mesh Size | MESH | Fragmentation: lower = more fragmented |

#### Class-Level (8 metrics per class)

CA, PLAND, NP, PD, LPI, ED, LSI, AREA_MN.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `landscape_metrics` | [np, pd, lpi, ed, lsi, shdi, contag, mesh] | Metrics list |
| `class_metrics` | [ca, pland, np, pd, lpi, ed, lsi, area_mn] | Per-class metrics |
| `neighborhood_rule` | `8` | `8` = Moore, `4` = von Neumann |

### Pitfalls

1. **PyLandStats not installed**: If `pip install pylandstats` fails, the script generates empty template CSVs and prints a warning. Install from conda: `conda install -c conda-forge pylandstats`.
2. **Large memory usage**: For very large rasters (>10,000 × 10,000 px), PyLandStats can be slow. Consider cropping to a tighter boundary or downsampling.

---

## Step 4: InVEST Carbon Storage

Estimates total carbon storage (Mg) per year as the sum of (LULC area × carbon pool density) across four carbon pools: above-ground biomass, below-ground biomass, soil organic carbon, and dead organic matter.

**Script**: `scripts/step4_invest_carbon.py`

### Formula

```
Carbon_total = Σ_i Area_i × (C_above_i + C_below_i + C_soil_i + C_dead_i)
```

Where i indexes LULC classes.

### Default Carbon Pools (Dabaoshan, Mg/ha)

| LULC Class | C_above | C_below | C_soil | C_dead | Total |
|------------|---------|---------|--------|--------|-------|
| Water | 0 | 0 | 0 | 0 | 0 |
| Built-up | 0 | 0 | 0 | 0 | 0 |
| Unrestored | 2.5 | 1.2 | 45 | 1.5 | 50.2 |
| Recovering | 15 | 7 | 60 | 3 | 85.0 |
| Stable vegetation | 85 | 40 | 95 | 8 | 228.0 |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `carbon_pools_csv` | null | InVEST-format carbon pool table |
| `carbon_pools` | (see above) | Direct pool specification per class |

### Pitfalls

1. **Pool values not calibrated**: Default values are from Dabaoshan literature. For other study areas, update pools in `invest_carbon.carbon_pools` or provide a calibrated `carbon_pools_csv`.
2. **Area unit consistency**: The formula assumes area in hectares and pool density in Mg/ha. If your area is in different units, adjust accordingly.

---

## Step 5: InVEST Habitat Quality

Computes habitat quality as area-weighted habitat suitability adjusted for degradation from threat sources (construction, barren land).

**Script**: `scripts/step5_invest_habitat_quality.py`

### Simplified HQ Model

```
HQ_mean = Σ_i Area_i × H_i × (1 - D_i) / Total_Area
```

Where H_i is the habitat suitability score (0-1) and D_i is the degradation factor from proximate threats.

### Default Habitat Suitability Scores

| LULC Class | H_score | Degradation | Effective HQ |
|------------|---------|-------------|-------------|
| Water | 0.8 | 0.05 | 0.76 |
| Built-up | 0.0 | 1.0 | 0.00 |
| Unrestored | 0.1 | 0.7 | 0.03 |
| Recovering | 0.4 | 0.3 | 0.28 |
| Stable vegetation | 0.9 | 0.05 | 0.855 |

**Note**: This is a reduced-form approximation. The full InVEST HQ model requires spatial threat rasters and distance-decay functions. For precise HQ estimates, run the full InVEST HQ tool separately and input the `quality_c_{year}.tif` rasters.

### Threat Definitions

| Threat | Max Distance (km) | Weight | Decay |
|--------|-------------------|--------|-------|
| Construction | 3.0 | 0.8 | Exponential |
| Barren land | 2.0 | 0.5 | Linear |

---

## Step 6: InVEST SDR Soil Erosion

Estimates soil erosion (USLE) and sediment delivery using the InVEST SDR model logic with LULC-based C and P factors and annual R factor variation.

**Script**: `scripts/step6_invest_sdr.py`

### Formula

```
USLE_i = R × K × LS × C_i × P_i          (t/ha/yr)
USLE_total = Σ_i USLE_i × Area_i          (t/yr)
Sed_export = USLE_total × SDR              (t/yr)
```

### Default C and P Factors

| LULC Class | C_factor | P_factor | Notes |
|------------|----------|----------|-------|
| Water | 0.00 | 0.0 | No erosion |
| Built-up | 0.01 | 1.0 | Impervious |
| Unrestored | 0.35 | 1.0 | High erosion risk |
| Recovering | 0.08 | 1.0 | Moderate cover |
| Stable vegetation | 0.003 | 1.0 | Good cover |

Annual R factors for Dabaoshan range from 4,400-5,800 MJ·mm/(ha·h·yr).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_factor_source` | `annual` | `annual` = year-specific R, `mean` = constant |
| `lulc_c_factor` | (see above) | C-factor per LULC class |
| `lulc_p_factor` | (see above) | P-factor per LULC class |

---

## Step 7: Merge Analysis Table

Combines all step outputs into a single unified analysis table with phase labels.

**Script**: `scripts/step7_merge_analysis.py`

### Output: `analysis_table.csv`

| Column Group | Columns | Source Step |
|-------------|---------|-------------|
| Year + Phase | `Year`, `phase` | Step 2 + config |
| LULC Areas | `area_{class}_ha`, `area_total_ha` | Step 2 |
| Landscape | `NP`, `PD`, `LPI`, `ED`, `LSI`, `SHDI`, `CONTAG`, `MESH` | Step 3 |
| Carbon | `carbon_total_Mg`, `carbon_density_Mg_ha` | Step 4 |
| Habitat | `habitat_quality_mean` | Step 5 |
| SDR | `usle_total_t_yr`, `sed_export_t_yr`, `r_factor_source` | Step 6 |
| Disturbance | `Pixel Count`, `Area (ha)`, `Mean MAG`, `Mean Rate`, ... | Step 1 |

### Phase Assignment

| Phase | Years | Environmental Context |
|-------|-------|----------------------|
| **Degradation** | 2013-2016 | Active mining period, highest disturbance frequency |
| **Transition** | 2017-2020 | Policy intervention begins, mixed recovery signals |
| **Consolidation** | 2021-2025 | Restoration takes effect, ES indicators stabilizing |

---

## Step 8: Correlation & Partial Correlation

Computes Pearson and Spearman correlation matrices with FDR Benjamini-Hochberg correction, plus partial correlation controlling for LULC area and landscape covariates.

**Script**: `scripts/step8_correlation.py`

### Why FDR Correction?

With ~15 variables, the full correlation matrix contains 105 unique pairwise tests. At α = 0.05, 5 false positives are expected by chance alone. Benjamini-Hochberg FDR controls the expected proportion of false discoveries among rejected hypotheses.

### Methods

| Method | Purpose |
|--------|---------|
| **Pearson r** | Linear relationship strength |
| **Spearman ρ** | Monotonic relationship (robust to outliers) |
| **Partial r** | Direct association controlling for 7 covariates |

### Case Study Results

Top significant pairs (FDR α = 0.05):
- Carbon × Habitat Quality: r = 0.965 (FDR p < 0.001)
- SHDI × Carbon: r = -0.858 (FDR p < 0.001)
- LPI × Carbon: r = 0.741 (FDR p < 0.01)
- MESH × Carbon: r = 0.744 (FDR p < 0.01)

---

## Step 9: OLS Regression Models

Fits three OLS multiple regression models specified in config, outputs detailed model summaries with coefficients, standard errors, t-values, p-values, R², adjusted R², and F-statistics.

**Script**: `scripts/step9_regression.py`

### Default Models

| Model | Formula | Hypothesis |
|-------|---------|------------|
| **Carbon** | Carbon ~ MAG + LPI + SHDI | Disturbance + landscape dominance + diversity predict carbon |
| **Habitat Quality** | HQ ~ MAG + MESH + CONTAG | Disturbance + fragmentation predict habitat quality |
| **Erosion** | Sed_export ~ Area + ED + LPI | Disturbed area + edge density predict soil loss |

### Model Diagnostics

| Diagnostic | Good | Bad |
|-----------|------|-----|
| **R²** | > 0.6 | < 0.2 |
| **F-test p** | < 0.05 | > 0.10 |
| **VIF** | < 5 | > 10 (multicollinearity) |
| **Durbin-Watson** | ≈ 2 | < 1 or > 3 (autocorrelation) |
| **Residual normality** | Shapiro-Wilk p > 0.05 | p < 0.05 (non-normal residuals) |

### Case Study: Carbon Model

```
Carbon_total_Mg = β₀ + β₁·MAG + β₂·LPI + β₃·SHDI

Adj R² = 0.778, F(3,8) = 13.82, p = 0.002

Coefficients:
  Intercept: β₀ = 3,612,000  (p < 0.001***)
  MAG:       β₁ = -196,000   (p = 0.246)
  LPI:       β₂ = +4,587     (p = 0.028*)
  SHDI:      β₃ = -717,000   (p = 0.001**)
```

Interpretation: A 0.1 unit increase in SHDI predicts 71,700 Mg carbon loss. LPI exerts a positive, stabilizing effect. MAG alone is not significant - its effect on carbon is mediated through landscape pattern (justifying the path analysis in Step 10).

---

## Step 10: SEM Path Analysis & Phase Comparison

Structural Equation Modeling (SEM) path analysis via semopy, testing the hypothesized causal chain: **disturbance → landscape pattern → ecosystem services**. Also computes Kruskal-Wallis H tests for three-phase comparison.

**Script**: `scripts/step10_path_analysis.py`

### Hypothesized Path Model

```
MAG (disturbance magnitude)
   ↓
SHDI (diversity)    MESH (fragmentation)    CONTAG (aggregation)
   ↓                     ↓                       ↓
Carbon_total    Carbon_total + HQ      HQ
   ↓
Sed_export ← SHDI + MAG
```

### Path Coefficients (Case Study)

| Path | β | p | Significant? |
|------|---|----|-------------|
| MAG → SHDI | +0.274 | 0.285 | No |
| MAG → MESH | -0.390 | 0.124 | No |
| SHDI → Carbon | -0.487 | < 0.001 | Yes *** |
| MESH → Carbon | +0.332 | 0.005 | Yes ** |
| MAG → Carbon (direct) | -0.320 | 0.012 | Yes * |

**Key insight**: The disturbance landscape carbon pathway shows **full mediation**: MAG is significant in the direct path (p = 0.012), yet MAG does not significantly predict SHDI or MESH. This suggests landscape pattern functions as a **parallel mediator**.

### Kruskal-Wallis Phase Comparison

| Variable | H | p | Significant? |
|----------|---|---|-------------|
| Carbon total | 7.50 | 0.024 | Yes * |
| Disturbance area | 5.49 | 0.064 | Marginal |
| SHDI | 4.85 | 0.089 | Marginal |

Significant KW result (p < 0.05) indicates at least one phase differs significantly from others. Post-hoc Dunn test can identify which specific phases differ.

---

## Result Interpretation

### Reading the Analysis Table

Open `analysis_table.csv` in any spreadsheet or data analysis tool:

1. **Temporal trends**: Plot `Year` vs. `carbon_total_Mg` - expect decline during Degradation, stabilization during Consolidation.
2. **Disturbance-Landscape coupling**: Plot `Mean MAG` vs. `SHDI` - higher magnitude should correlate with higher diversity (more fragmentation).
3. **Phase boxplots**: Group by `phase` and compare distributions of key variables.

### Reading Correlation Matrices

1. **Carbon × HQ**: Near-perfect correlation (r > 0.9) confirms both respond to the same landscape drivers.
2. **SHDI negative correlations**: Higher landscape diversity correlates with lower carbon, HQ - this is the fragmentation penalty.
3. **Non-significant pairs**: Variables that fail FDR correction (p_fdr > 0.05) lack robust evidence of association.

### Reading Path Analysis

1. **Significant paths (p < 0.05)**: Robust evidence of direct effect.
2. **Non-significant paths (p > 0.05)**: No evidence of direct effect; effect may be indirect or absent.
3. **Model fit**: CFI > 0.90 and RMSEA < 0.08 indicate acceptable fit. With only 13 observations, fit indices should be interpreted cautiously.

### Statistical Reporting Template

```
Pearson correlation: Carbon × HQ, r = 0.965, FDR-corrected p < 0.001.
OLS regression: Carbon ~ MAG + LPI + SHDI, Adj R² = 0.778, 
  F(3,8) = 13.82, p = 0.002.
Path analysis: SHDI → Carbon β = -0.487, p < 0.001; 
  MESH → Carbon β = 0.332, p = 0.005.
Kruskal-Wallis: Carbon across phases, H = 7.50, p = 0.024*.
```

---

## Full Config Reference

```yaml
# ── Global ──
output_dir: "D:/MyProject/output"        # All outputs land here
crs: "EPSG:32649"
steps: [dist, class, land, carbon, hq, sdr, merge, corr, regress, path]

# ── Study Area ──
study_area:
  boundary_shp: "D:/data/boundary.shp"
  pixel_area_ha: 0.09                   # 30m × 30m
  raster_res_m: 30

# ── Field Mapping ──
field_mapping:
  lulc_class_field: "class_name"
  lulc_code_field: "class_code"
  year_field: "Year"
  phase_field: "phase"

# ── LandTrendr (Step 1) ──
landtrendr:
  raster_dir: "D:/data/LT/"
  yod_raster: "yod_2009_2024.tif"
  mag_raster: "mag_2009_2024.tif"
  dur_raster: "dur_2009_2024.tif"
  mpy_raster: "magperyear_2009_2024.tif"
  yod_range: [2013, 2024]
  intensity_bins:                        # 4-class (CSV export)
    low:       [0.2, 0.35, "#f1c40f"]
    moderate:  [0.35, 0.5, "#e67e22"]
    high:      [0.5, 0.65, "#e74c3c"]
    very_high: [0.65, 9.0, "#8b0000"]
  severity_bins:                         # 3-class (spatial map)
    low:       [0.2, 0.35, "#f1c40f"]
    moderate:  [0.35, 0.55, "#e67e22"]
    high:      [0.55, 9.0, "#c0392b"]

# ── LULC (Step 2) ──
lulc:
  raster_dir: "D:/data/LULC/"
  raster_pattern: "lulc_{year}.tif"
  year_range: [2000, 2025]
  classes:
    water: 1
    built_up: 2
    unrestored: 3
    recovering: 4
    stable_vegetation: 5

# ── Landscape (Step 3) ──
landscape:
  landscape_metrics: [np, pd, lpi, ed, lsi, shdi, contag, mesh]
  class_metrics: [ca, pland, np, pd, lpi, ed, lsi, area_mn]
  neighborhood_rule: "8"

# ── Carbon (Step 4) ──
invest_carbon:
  carbon_pools_csv: null                # Or path to InVEST CSV
  carbon_pools:
    water: {c_above: 0, c_below: 0, c_soil: 0, c_dead: 0}
    built_up: {c_above: 0, c_below: 0, c_soil: 0, c_dead: 0}
    unrestored: {c_above: 2.5, c_below: 1.2, c_soil: 45, c_dead: 1.5}
    recovering: {c_above: 15, c_below: 7, c_soil: 60, c_dead: 3}
    stable_vegetation: {c_above: 85, c_below: 40, c_soil: 95, c_dead: 8}

# ── Habitat Quality (Step 5) ──
invest_habitat_quality:
  half_saturation_constant: 0.5
  habitat_scores:
    water: 0.8
    built_up: 0.0
    unrestored: 0.1
    recovering: 0.4
    stable_vegetation: 0.9

# ── SDR (Step 6) ──
invest_sdr:
  r_factor_source: "annual"
  lulc_c_factor:
    water: 0.0
    built_up: 0.01
    unrestored: 0.35
    recovering: 0.08
    stable_vegetation: 0.003

# ── Correlation (Step 8) ──
correlation:
  methods: [pearson, spearman]
  alpha: 0.05
  fdr_method: "fdr_bh"
  analysis_years: [2013, 2025]

# ── Regression (Step 9) ──
regression:
  models:
    carbon_model:
      dependent: "carbon_total_Mg"
      explanatory: ["Mean MAG", LPI, SHDI]

# ── Path Analysis (Step 10) ──
path_analysis:
  model_spec: |
    SHDI ~ Mean MAG
    MESH ~ Mean MAG
    carbon_total_Mg ~ SHDI + MESH + Mean MAG
    habitat_quality_mean ~ SHDI + CONTAG + Mean MAG

phase_analysis:
  phases:
    Degradation: [2013, 2016]
    Transition: [2017, 2020]
    Consolidation: [2021, 2025]
  test: "kruskal_wallis"
  variables: ["Area (ha)", Mean MAG, SHDI, carbon_total_Mg, habitat_quality_mean]
```

---

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError: yod_*.tif` | Wrong `raster_dir` or filenames | Check `landtrendr.raster_dir` and individual raster filenames in config |
| `CRS mismatch` warning in console | YOD exported in EPSG:4326 | Script auto-reprojects; no action needed unless performance is critical |
| `disturbed.sum() == 0` | `yod_range` outside disturbance years | Run data check: inspect YOD unique values; adjust `yod_range` |
| `pylandstats` import error | Not installed | `pip install pylandstats` or `conda install -c conda-forge pylandstats` |
| `semopy` import error | Not installed | `pip install semopy`; OLS fallback runs automatically |
| `pingouin` import error | Not installed | `pip install pingouin`; scipy fallback runs automatically |
| Merge step: all-NaN columns | Different column names across steps | Use `field_mapping` in config; check each step's output CSV headers |
| Regression: "Missing columns" | Explanatory variable not in analysis table | Check that merge step produced expected columns; verify variable names match case |
| Path analysis: "N too small" | Only 13 years of data | SEM with N < 100 has low power; results are exploratory. Report with caution |
| Kruskal-Wallis: "p > 0.05" for all vars | Phases not meaningfully different for those variables | This is a valid result; report as "no significant phase difference" |
| MemoryError during rasterio clip | Very large raster + small RAM | Use `rasterio.windows` to read in tiles; or pre-clip rasters in QGIS/ArcGIS |

---

## Validation Results

The pipeline was validated against the original Jupyter notebook analysis (`Dabaoshan_New_landtr_inspect_CN.ipynb`, `Dabaoshan_LULC_analysis_CN.ipynb`, `Dabaoshan_Statistics_Ch7.ipynb`):

| Analysis | Key Metric | Original Notebook | This Pipeline | Match |
|----------|-----------|------------------|---------------|-------|
| LandTrendr | Total disturbed pixels | 1,008 | Reproduced | ✓ |
| LandTrendr | Disturbed area (ha) | 90.7 | Reproduced | ✓ |
| LULC | Total area (ha) | 1,509.48 | Reproduced | ✓ |
| Carbon | Total 2013 (Mg) | 2,189,058 | Reproduced | ✓ |
| Carbon | Total 2023 (Mg) | 1,607,116 | Reproduced | ✓ |
| Summary CSV | 17 columns (Nanling convention) | Verified | Reproduced | ✓ |
| Correlation | Carbon × HQ (r) | 0.965 | Reproduced | ✓ |
| Regression | Carbon model Adj R² | 0.778 | Reproduced | ✓ |
| Path analysis | SHDI → Carbon β | -0.487 | Reproduced | ✓ |
| Kruskal-Wallis | Carbon H | 7.50, p = 0.024 | Reproduced | ✓ |

Minor differences (±0.01%) are attributable to floating-point precision in numpy/scipy versions.

---

## Files

```
Landtrendr_analysis/
├── README.md                              # Full pipeline documentation (this file)
├── requirements.txt                       # numpy, scipy, pandas, rasterio, geopandas,
│                                          #   pylandstats, pingouin, statsmodels, semopy
├── config_template.json                   # Copy and customize for your study area
├── .gitignore                             # Python + GIS exclusions
└── scripts/
    ├── _utils.py                          # Shared: constants (INTENSITY_BINS, SEVERITY_BINS,
    │                                      #   carbon pools, LULC classes), field resolution,
    │                                      #   raster load/clip helper, classify functions
    ├── master_pipeline.py                 # Orchestrator: run all or selected steps
    ├── step1_landtrendr_disturbance.py    # LandTrendr analysis (Nanling conventions):
    │                                      #   load/clip YOD/MAG/DUR/MPY, annual stats,
    │                                      #   4-class intensity, 3-class severity,
    │                                      #   16 figures, summary_statistics.csv
    ├── step2_lulc_trends.py               # LULC area trends + transition matrices
    ├── step3_landscape_metrics.py         # PyLandStats landscape + class metrics
    ├── step4_invest_carbon.py             # InVEST Carbon storage estimation
    ├── step5_invest_habitat_quality.py    # InVEST Habitat Quality assessment
    ├── step6_invest_sdr.py                # InVEST SDR soil erosion + sediment delivery
    ├── step7_merge_analysis.py            # Merge all into analysis_table.csv + phase labels
    ├── step8_correlation.py               # Pearson/Spearman (FDR) + partial correlation
    ├── step9_regression.py                # OLS multiple regression models
    └── step10_path_analysis.py            # SEM path analysis + Kruskal-Wallis phase comparison
```

## License

MIT
