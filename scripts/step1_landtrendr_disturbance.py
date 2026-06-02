"""
Step 1: LandTrendr Disturbance Raster Analysis
===============================================
Loads GEE-exported LandTrendr rasters (YOD, MAG, DUR, MPY), clips to
study boundary, generates disturbance masks, computes annual statistics,
intensity/severity classification, and produces 16 diagnostic figures.

Follows the Nanling analysis conventions:
  - Intensity bins (4-class): Low [0.2,0.35), Moderate [0.35,0.5),
    High [0.5,0.65), Very High [0.65, inf)
  - Severity bins (3-class, for map): Low [0.2,0.35), Moderate [0.35,0.55),
    High [0.55, inf)
  - Output: summary_statistics.csv with standard columns

Input:  GEE-exported LandTrendr GeoTIFF rasters, study boundary shapefile.
Output: summary_statistics.csv, 16 figures (01-16), severity data.

Usage:
  python step1_landtrendr_disturbance.py --raster-dir D:/data/LT/ \
      --boundary D:/data/boundary.shp --output-dir D:/output/
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import mapping
from scipy import ndimage

from _utils import ensure_dir

# ── Constants: Intensity bins (4-class, used for CSV + stacked bar) ──
INTENSITY_BINS = [
    ("Low (0.2-0.35)",      0.2, 0.35, "#f1c40f"),
    ("Moderate (0.35-0.5)", 0.35, 0.5,  "#e67e22"),
    ("High (0.5-0.65)",     0.5, 0.65,  "#e74c3c"),
    ("Very High (>0.65)",   0.65, 9.0,  "#8b0000"),
]

# ── Constants: Severity bins (3-class, for spatial map) ──
SEVERITY_BINS = [
    ("Low (0.2-0.35)",      0.2, 0.35, "#f1c40f"),
    ("Moderate (0.35-0.55)", 0.35, 0.55, "#e67e22"),
    ("High (>0.55)",        0.55, 9.0,  "#c0392b"),
]

# ── Duration labels ──
DUR_LABELS = ["1 year", "2 years", "≥3 years"]
DUR_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]


def analyze_landtrendr(
    raster_dir,
    boundary_shp,
    output_dir,
    yod_file="yod_2009_2024.tif",
    mag_file="mag_2009_2024.tif",
    dur_file="dur_2009_2024.tif",
    mpy_file="magperyear_2009_2024.tif",
    nodata_val=0,
    yod_range=None,
    pixel_area_ha=0.09,
    intensity_bins=None,
    severity_bins=None,
    figure_prefix=None,
    generate_figures=True,
):
    """
    Analyze LandTrendr disturbance detection rasters using Nanling conventions.

    Parameters
    ----------
    raster_dir : str        Directory containing YOD, MAG, DUR, MPY rasters.
    boundary_shp : str      Path to study boundary shapefile.
    output_dir : str        Output directory.
    yod_file, mag_file, dur_file, mpy_file : str  Raster filenames.
    nodata_val : int/float  Nodata fill value.
    yod_range : list        [start_year, end_year] for disturbance window.
    pixel_area_ha : float   Pixel area in hectares (30m × 30m = 0.09).
    intensity_bins : list   [(label, lo, hi, color), ...] for 4-class.
    severity_bins : list    [(label, lo, hi, color), ...] for 3-class map.
    figure_prefix : str     Output figure path prefix (None = auto).
    generate_figures : bool Whether to generate 16 diagnostic figures.

    Returns
    -------
    str  Path to summary_statistics.csv.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.features import rasterize
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import warnings
    warnings.filterwarnings('ignore')

    if yod_range is None:
        yod_range = [2010, 2024]
    if intensity_bins is None:
        intensity_bins = INTENSITY_BINS
    if severity_bins is None:
        severity_bins = SEVERITY_BINS
    if figure_prefix is None:
        figure_prefix = str(Path(output_dir) / "fig_")

    T0 = time.time()

    # ── Setup matplotlib style ──
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    data_dir = Path(raster_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load boundary ──
    print(f"  [{time.time()-T0:.0f}s] Loading boundary: {boundary_shp}")
    boundary_gdf = gpd.read_file(boundary_shp)
    print(f"    CRS: {boundary_gdf.crs}, features: {len(boundary_gdf)}")

    # ── Per-raster CRS handling: YOD defines the reference grid; every  ──
    # ── other raster is clipped in ITS OWN native CRS, then warped onto  ──
    # ── the YOD grid so all four arrays share CRS / transform / shape.   ──
    from rasterio.warp import reproject, Resampling

    yod_path = data_dir / yod_file
    mag_path = data_dir / mag_file
    dur_path = data_dir / dur_file
    mpy_path = data_dir / mpy_file

    def _clip_native(path, nodata=0):
        """Clip a raster to the boundary in the raster's OWN native CRS."""
        with rasterio.open(path) as src:
            src_crs = src.crs
            if src_crs is None:
                raise ValueError(f"{Path(path).name} has no CRS; cannot clip safely.")
            bnd = boundary_gdf.to_crs(src_crs)            # reproject boundary per-raster
            shapes = [mapping(g) for g in bnd.geometry if g is not None]
            arr, tf = rio_mask(src, shapes, crop=True, nodata=nodata, filled=True)
            meta = src.meta.copy()
        return arr[0], tf, src_crs, meta

    def _align_to_ref(path, ref_crs, ref_tf, ref_shape, resampling, nodata=0):
        """Clip in native CRS, then warp onto the reference grid only if needed."""
        arr, tf, src_crs, _ = _clip_native(path, nodata=nodata)
        if src_crs == ref_crs and arr.shape == ref_shape:
            return arr                                    # already aligned; no warp
        print(f"      reprojecting {Path(path).name}: {src_crs} -> {ref_crs} "
              f"({arr.shape} -> {ref_shape})")
        dst = np.full(ref_shape, nodata, dtype=arr.dtype)
        reproject(
            source=arr, destination=dst,
            src_transform=tf, src_crs=src_crs,
            dst_transform=ref_tf, dst_crs=ref_crs,
            src_nodata=nodata, dst_nodata=nodata,
            resampling=resampling,
        )
        return dst

    print(f"  [{time.time()-T0:.0f}s] Loading & clipping rasters (per-raster CRS handling)...")

    # YOD = reference grid (CRS / transform / shape). Never warped → integer YOD preserved.
    yod, yod_tf, ref_crs, yod_meta = _clip_native(yod_path)
    ref_shape = yod.shape
    yod_meta.update({'height': ref_shape[0], 'width': ref_shape[1], 'transform': yod_tf})
    yod_bnds = rasterio.transform.array_bounds(ref_shape[0], ref_shape[1], yod_tf)
    print(f"    Reference CRS (from YOD): {ref_crs}; grid {ref_shape}")

    boundary_proj = boundary_gdf.to_crs(ref_crs)          # used by roi_mask + plots below

    # integer / categorical -> nearest;  continuous magnitude -> bilinear
    mag = _align_to_ref(mag_path, ref_crs, yod_tf, ref_shape, Resampling.bilinear)
    dur = _align_to_ref(dur_path, ref_crs, yod_tf, ref_shape, Resampling.nearest)
    mpy = _align_to_ref(mpy_path, ref_crs, yod_tf, ref_shape, Resampling.bilinear)

    # Extent for imshow [left, right, bottom, top]
    extent = [yod_bnds[0], yod_bnds[2], yod_bnds[1], yod_bnds[3]]

    # ── Rasterize boundary → roi_mask ──
    roi_mask = rasterize(
        [(geom, 1) for geom in boundary_proj.geometry],
        out_shape=yod.shape,
        transform=yod_tf,
        fill=0,
        dtype=np.uint8
    ) == 1

    # ── Core masks ──
    y1, y2 = yod_range
    disturbed = (yod >= y1) & (yod <= y2) & roi_mask
    pixel_ha = pixel_area_ha
    years = np.arange(y1, y2 + 1)

    print(f"    Clipped raster shape: {yod.shape}")
    print(f"    Pixels inside boundary: {roi_mask.sum():,}")
    print(f"    Disturbed pixels: {disturbed.sum():,}")
    print(f"    Disturbed area: {disturbed.sum() * pixel_ha:,.1f} ha")
    print(f"    Disturbance ratio: {disturbed.sum() / roi_mask.sum() * 100:.2f}%")

    # ── Boundary overlay helper ──
    def plot_boundary(ax, **kwargs):
        style = dict(facecolor='none', edgecolor='white', linewidth=0.8, alpha=0.9)
        style.update(kwargs)
        boundary_proj.boundary.plot(ax=ax, **style)

    # ═══════════════════════════════════════════════════
    # PART 1: Summary Statistics & Area by Year
    # ═══════════════════════════════════════════════════
    print(f"\n  [{time.time()-T0:.0f}s] Part 1: Summary Statistics & Area by Year")

    # Cell 2: YOD - Disturbed Area by Year
    pixel_counts = np.array([((yod == y) & roi_mask).sum() for y in years])
    area_ha = pixel_counts * pixel_ha

    yod_df = pd.DataFrame({
        'Year': years,
        'Pixel Count': pixel_counts,
        'Area (ha)': area_ha,
        'Area (km²)': area_ha / 100
    })
    yod_df['Pct of Total Disturbed'] = (
        yod_df['Pixel Count'] / pixel_counts.sum() * 100).round(2)
    print(f"    Total disturbed area: {area_ha.sum():,.1f} ha ({area_ha.sum()/100:,.2f} km²)")

    # Cell 3: MAG & MAG-per-year distributions
    mag_valid = mag[disturbed]
    mpy_valid = mpy[disturbed]
    print(f"    MAG: median={np.median(mag_valid):.4f}, mean={mag_valid.mean():.4f}")
    print(f"    MPY: median={np.median(mpy_valid):.4f}, mean={mpy_valid.mean():.4f}")

    # Cell 4: DUR - Duration Class Distribution
    dur_valid = dur[disturbed]
    dur_counts = [(dur_valid == 1).sum(), (dur_valid == 2).sum(), (dur_valid == 3).sum()]
    dur_area = [c * pixel_ha for c in dur_counts]
    print(f"    Duration: 1yr={dur_area[0]:.0f}ha, 2yr={dur_area[1]:.0f}ha, 3yr={dur_area[2]:.0f}ha")

    # Cell 5: Mean Magnitude by Year (trend)
    yearly_stats = []
    for y in years:
        m = (yod == y) & roi_mask
        if m.sum() > 0:
            yearly_stats.append({
                'Year': y,
                'Mean MAG': mag[m].mean(),
                'Median MAG': np.median(mag[m]),
                'Std MAG': mag[m].std(),
                'Mean Rate': mpy[m].mean(),
                'Pixel Count': int(m.sum())
            })
    ys_df = pd.DataFrame(yearly_stats)

    # ═══════════════════════════════════════════════════
    # PART 3: Cross-tabulation & Intensity Analysis
    # ═══════════════════════════════════════════════════
    print(f"  [{time.time()-T0:.0f}s] Part 3: Cross-tabulation & Intensity Analysis")

    # Duration × Year cross-tab (ha)
    cross_tab = np.zeros((3, len(years)))
    for i, d_val in enumerate([1, 2, 3]):
        for j, y in enumerate(years):
            cross_tab[i, j] = ((yod == y) & (dur == d_val) & roi_mask).sum() * pixel_ha

    # Intensity × Year table (ha)
    n_intensity = len(intensity_bins)
    int_year = np.zeros((n_intensity, len(years)))
    for i, (label, lo, hi, _) in enumerate(intensity_bins):
        for j, y in enumerate(years):
            int_year[i, j] = ((yod == y) & roi_mask & (mag >= lo) & (mag < hi)).sum() * pixel_ha

    int_labels = [b[0] for b in intensity_bins]
    int_colors = [b[3] for b in intensity_bins]

    # Severe proportion per year (High + Very High / total)
    severe = int_year[2, :] + int_year[3, :]  # High + Very High
    total_by_year = int_year.sum(axis=0)
    total_by_year[total_by_year == 0] = 1
    severe_pct = severe / total_by_year * 100

    # Severity spatial map (3-class)
    severity = np.full(mag.shape, np.nan, dtype=float)
    for i, (label, lo, hi, _) in enumerate(severity_bins):
        severity[disturbed & (mag >= lo) & (mag < hi)] = i + 1

    # Peak year analysis
    overall_peak = years[np.argmax(total_by_year)]
    severe_peak_year = years[np.argmax(severe_pct)]
    print(f"    Overall peak year: {overall_peak} ({total_by_year.max():,.1f} ha)")
    print(f"    Most severe year: {severe_peak_year} ({severe_pct.max():.1f}%)")

    peak_data = []
    for i, (label, lo, hi, color) in enumerate(intensity_bins):
        row = int_year[i, :]
        peak_idx = np.argmax(row)
        peak_year = years[peak_idx]
        peak_area = row[peak_idx]
        row_copy = row.copy()
        row_copy[peak_idx] = 0
        second_idx = np.argmax(row_copy)
        peak_data.append({
            'Intensity': label,
            'Peak Year': peak_year,
            'Peak Area (ha)': round(peak_area, 1),
            '2nd Peak Year': years[second_idx],
            '2nd Peak Area (ha)': round(row_copy[second_idx], 1),
            'Total Area (ha)': round(row.sum(), 1)
        })

    # ═══════════════════════════════════════════════════
    # EXPORT: Build summary_statistics.csv (Cell 19 style)
    # ═══════════════════════════════════════════════════
    print(f"  [{time.time()-T0:.0f}s] Building summary_statistics.csv...")

    summary = yod_df[['Year', 'Pixel Count', 'Area (ha)', 'Area (km²)']].copy()
    summary = summary.merge(
        ys_df[['Year', 'Mean MAG', 'Median MAG', 'Std MAG', 'Mean Rate']],
        on='Year', how='left'
    )
    # Duration columns
    for i, d_label in enumerate(['Dur_1yr_ha', 'Dur_2yr_ha', 'Dur_3yr_ha']):
        summary[d_label] = cross_tab[i, :]
    # Intensity columns
    for i, label in enumerate(int_labels):
        col_name = label.split('(')[0].strip().replace(' ', '_') + '_ha'
        summary[col_name] = int_year[i, :]
    summary['Severe_pct'] = severe_pct

    csv_path = out_dir / 'summary_statistics.csv'
    summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    Saved: {csv_path}")
    print(f"    {len(summary)} years, {len(summary.columns)} columns")

    # ═══════════════════════════════════════════════════
    # FIGURES: 16 diagnostic figures (Nanling style)
    # ═══════════════════════════════════════════════════
    if generate_figures:
        _generate_all_figures(
            out_dir, years, area_ha, pixel_counts, yod_df, ys_df,
            mag_valid, mpy_valid, dur_counts, dur_area, cross_tab,
            int_year, int_labels, int_colors, intensity_bins,
            severity, severity_bins, severe_pct,
            yod, mag, dur, mpy, disturbed, roi_mask,
            extent, yod_tf, yod_bnds, pixel_ha,
            plot_boundary, boundary_proj,
            peak_data, overall_peak, severe_peak_year,
            T0
        )

    print(f"\n  DONE [Step 1, {time.time()-T0:.0f}s]")
    return str(csv_path)


def _generate_all_figures(out_dir, years, area_ha, pixel_counts, yod_df, ys_df,
                           mag_valid, mpy_valid, dur_counts, dur_area, cross_tab,
                           int_year, int_labels, int_colors, intensity_bins,
                           severity, severity_bins, severe_pct,
                           yod, mag, dur, mpy, disturbed, roi_mask,
                           extent, yod_tf, yod_bnds, pixel_ha,
                           plot_boundary, boundary_proj,
                           peak_data, overall_peak, severe_peak_year,
                           T0):
    """Generate all 16 figures following the Nanling notebook conventions."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    print(f"\n  [{time.time()-T0:.0f}s] Generating 16 figures...")

    # ── Figure 01: Annual Disturbance Area ──
    fig, ax1 = plt.subplots(figsize=(12, 5))
    bars = ax1.bar(years, area_ha, color='#2ecc71', edgecolor='white',
                    linewidth=0.5, alpha=0.85, label='Disturbed Area')
    for bar, val in zip(bars, area_ha):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + area_ha.max()*0.01,
                     f'{val:,.0f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xlabel('Year of Detection', fontsize=11)
    ax1.set_ylabel('Disturbed Area (ha)', fontsize=11)
    ax1.set_title('Annual Forest Disturbance Area (2010-2024)', fontsize=13, fontweight='bold')
    ax1.set_xticks(years); ax1.set_xticklabels(years, rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(years, pixel_counts, color='#e74c3c', linewidth=2, marker='o', markersize=5, label='Disturbance Pixels')
    ax2.set_ylabel('Number of Disturbed Pixels', fontsize=11, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.legend(loc='upper right'); ax1.legend(loc='upper left')
    plt.tight_layout(); fig.savefig(out_dir / '01_annual_disturbance_area.png'); plt.close(fig)

    # ── Figure 02: MAG & MAG-per-year Distributions ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(mag_valid, bins=80, color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax1.axvline(np.median(mag_valid), color='black', linestyle='--', linewidth=1,
                label=f'Median={np.median(mag_valid):.3f}')
    ax1.set_xlabel('Magnitude (NBR)'); ax1.set_ylabel('Pixel Count')
    ax1.set_title('Distribution of Disturbance Magnitude', fontweight='bold')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    ax2.hist(mpy_valid, bins=80, color='#8e44ad', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax2.axvline(np.median(mpy_valid), color='black', linestyle='--', linewidth=1,
                label=f'Median={np.median(mpy_valid):.3f}')
    ax2.set_xlabel('Magnitude per Year (NBR/yr)'); ax2.set_ylabel('Pixel Count')
    ax2.set_title('Distribution of Disturbance Rate', fontweight='bold')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '02_mag_distributions.png'); plt.close(fig)

    # ── Figure 03: Duration Distribution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pie_colors = DUR_COLORS
    ax1.pie(dur_counts, labels=DUR_LABELS, colors=pie_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Duration Class Proportions', fontweight='bold')
    ax2.bar(DUR_LABELS, dur_area, color=pie_colors, edgecolor='white')
    for i, val in enumerate(dur_area):
        ax2.text(i, val + max(dur_area)*0.02, f'{val:,.0f} ha', ha='center', fontsize=9)
    ax2.set_ylabel('Area (ha)'); ax2.set_title('Disturbed Area by Duration Class', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '03_duration_distribution.png'); plt.close(fig)

    # ── Figure 04: Annual Magnitude Trend ──
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(ys_df['Year'], ys_df['Pixel Count'], color='#bdc3c7', alpha=0.6, label='Pixel Count')
    ax1.set_ylabel('Pixel Count', color='gray'); ax1.tick_params(axis='y', labelcolor='gray')
    ax2 = ax1.twinx()
    ax2.plot(ys_df['Year'], ys_df['Mean MAG'], color='#e74c3c', linewidth=2, marker='s', markersize=5, label='Mean MAG')
    ax2.plot(ys_df['Year'], ys_df['Median MAG'], color='#2980b9', linewidth=2, marker='^', markersize=5, label='Median MAG')
    ax2.fill_between(ys_df['Year'],
                     ys_df['Mean MAG']-ys_df['Std MAG'],
                     ys_df['Mean MAG']+ys_df['Std MAG'],
                     color='#e74c3c', alpha=0.1)
    ax2.set_ylabel('Magnitude (NBR)'); ax2.legend(loc='upper right')
    ax1.set_xlabel('Year'); ax1.set_title('Annual Disturbance Magnitude Trends', fontsize=13, fontweight='bold')
    ax1.set_xticks(years); ax1.set_xticklabels(years, rotation=45); ax1.legend(loc='upper left')
    plt.tight_layout(); fig.savefig(out_dir / '04_annual_magnitude_trend.png'); plt.close(fig)

    # ── Figure 05: Spatial Map - YOD ──
    yod_plot = np.where(disturbed, yod.astype(float), np.nan)
    n_years = len(years)
    base_colors = ['#1f77b4','#2ca02c','#d62728','#ff7f0e','#9467bd','#17becf',
                   '#e377c2','#8c564b','#bcbd22','#7f7f7f','#00cc96','#636efa',
                   '#ffa15a','#ef553b','#ab63fa']
    discrete_cmap = mcolors.ListedColormap(base_colors[:n_years])
    bounds = np.arange(years[0] - 0.5, years[-1] + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#1a1a2e')
    im = ax.imshow(yod_plot, cmap=discrete_cmap, norm=norm, extent=extent, interpolation='nearest')
    plot_boundary(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=years)
    cbar.set_label('Year of Detection', fontsize=11)
    cbar.ax.set_yticklabels([str(y) for y in years], fontsize=8)
    ax.set_title('Spatial Distribution of Forest Disturbance (YOD)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    plt.tight_layout(); fig.savefig(out_dir / '05_spatial_yod.png'); plt.close(fig)

    # ── Figure 06: Spatial Map - Magnitude ──
    mag_plot = np.where(disturbed, mag.astype(float), np.nan)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap_mag = plt.cm.hot_r.copy(); cmap_mag.set_bad(color='#1a1a2e')
    im = ax.imshow(mag_plot, cmap=cmap_mag, vmin=0.2, vmax=0.8, extent=extent, interpolation='nearest')
    plot_boundary(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02); cbar.set_label('Magnitude (NBR)')
    ax.set_title('Spatial Distribution of Disturbance Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    plt.tight_layout(); fig.savefig(out_dir / '06_spatial_magnitude.png'); plt.close(fig)

    # ── Figure 07: Hotspot Density Map ──
    binary = disturbed.astype(float)
    density = ndimage.gaussian_filter(binary, sigma=15)
    density_km2 = density * (1000 / 30) ** 2
    density_plot = np.where(roi_mask, density_km2, np.nan)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap_hot = plt.cm.inferno.copy(); cmap_hot.set_bad(color='#1a1a2e')
    im = ax.imshow(density_plot, cmap=cmap_hot, vmin=0,
                   vmax=np.nanpercentile(density_plot, 98), extent=extent, interpolation='nearest')
    plot_boundary(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02); cbar.set_label('Disturbance Density (per km²)')
    ax.set_title('Disturbance Hotspot Density Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    plt.tight_layout(); fig.savefig(out_dir / '07_hotspot_density.png'); plt.close(fig)

    # ── Figure 08: Temporal Hotspot - 3-period comparison ──
    periods = {'Early (2010-2014)': (2010, 2014), 'Middle (2015-2019)': (2015, 2019),
               'Recent (2020-2024)': (2020, 2024)}
    global_vmax = np.nanpercentile(density_plot, 95)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (label, (y_s, y_e)) in zip(axes, periods.items()):
        pm = (yod >= y_s) & (yod <= y_e) & roi_mask
        pd_km2 = ndimage.gaussian_filter(pm.astype(float), sigma=15) * (1000/30)**2
        pp = np.where(roi_mask, pd_km2, np.nan)
        im = ax.imshow(pp, cmap='inferno', vmin=0, vmax=global_vmax, extent=extent, interpolation='nearest')
        plot_boundary(ax, edgecolor='cyan', linewidth=0.6)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    fig.suptitle('Disturbance Hotspot Shift Across Periods', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(); fig.savefig(out_dir / '08_temporal_hotspot_periods.png'); plt.close(fig)

    # ── Figure 09: MAG by Duration Class - Boxplot per Year ──
    records = []
    for y in years:
        m = (yod == y) & roi_mask
        if m.sum() > 0:
            for d_val, d_label in zip([1, 2, 3], DUR_LABELS):
                sub = m & (dur == d_val)
                if sub.sum() > 0:
                    vals = mag[sub]
                    for v in vals:
                        records.append({'Year': y, 'Duration': d_label, 'MAG': v})
    cross_df = pd.DataFrame(records)
    color_map = dict(zip(DUR_LABELS, DUR_COLORS))
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.25
    for i, y in enumerate(years):
        for j, d in enumerate(DUR_LABELS):
            subset = cross_df[(cross_df['Year'] == y) & (cross_df['Duration'] == d)]['MAG']
            if len(subset) > 5:
                pos = i + (j - 1) * width
                bp = ax.boxplot([subset.values], positions=[pos], widths=width * 0.8,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='black', linewidth=1.5))
                bp['boxes'][0].set_facecolor(color_map[d]); bp['boxes'][0].set_alpha(0.7)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel('Year of Detection'); ax.set_ylabel('Magnitude (NBR)')
    ax.set_title('Disturbance Magnitude by Year and Duration Class', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(handles=[Patch(facecolor=color_map[d], alpha=0.7, label=d) for d in DUR_LABELS], loc='upper right')
    plt.tight_layout(); fig.savefig(out_dir / '09_mag_by_year_duration.png'); plt.close(fig)

    # ── Figure 10: Cross-tab Heatmap - Duration × Year ──
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(cross_tab, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(DUR_LABELS)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=45)
    for i in range(3):
        for j in range(len(years)):
            val = cross_tab[i, j]
            if val > 0:
                c = 'white' if val > cross_tab.max() * 0.5 else 'black'
                ax.text(j, i, f'{val:,.0f}', ha='center', va='center', fontsize=7, color=c)
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Area (ha)')
    ax.set_title('Duration × Year Cross-tabulation (ha)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year of Detection'); ax.set_ylabel('Duration Class')
    plt.tight_layout(); fig.savefig(out_dir / '10_crosstab_dur_year.png'); plt.close(fig)

    # ── Figure 11: Stacked Area Chart - Duration Composition ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(years, cross_tab[0, :], cross_tab[1, :], cross_tab[2, :],
                 labels=DUR_LABELS, colors=DUR_COLORS, alpha=0.8)
    ax.set_xlabel('Year of Detection'); ax.set_ylabel('Disturbed Area (ha)')
    ax.set_title('Duration Composition of Annual Disturbances', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left'); ax.set_xticks(years); ax.set_xticklabels(years, rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '11_stacked_duration_composition.png'); plt.close(fig)

    # ── Figure 12: MAG vs Rate Scatter ──
    np.random.seed(42)
    n_sample = min(50000, int(disturbed.sum()))
    idx = np.where(disturbed.ravel())[0]
    sample_idx = np.random.choice(idx, size=n_sample, replace=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(mag.ravel()[sample_idx], mpy.ravel()[sample_idx],
                    c=yod.ravel()[sample_idx], cmap='RdYlGn_r', vmin=years[0], vmax=years[-1],
                    s=3, alpha=0.4, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7); cbar.set_label('Year of Detection')
    cbar.set_ticks(np.arange(years[0], years[-1]+1, 2))
    ax.set_xlabel('Magnitude (NBR)'); ax.set_ylabel('Magnitude per Year (NBR/yr)')
    ax.set_title('Magnitude vs. Rate of Change (colored by YOD)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '12_mag_vs_rate_scatter.png'); plt.close(fig)

    # ── Figure 13: Severity Classification Map (3-class) ──
    fig, ax = plt.subplots(figsize=(12, 10))
    sev_cmap = mcolors.ListedColormap([b[3] for b in severity_bins])
    sev_cmap.set_bad(color='#1a1a2e')
    im = ax.imshow(severity, cmap=sev_cmap, vmin=0.5, vmax=len(severity_bins)+0.5,
                   extent=extent, interpolation='nearest')
    plot_boundary(ax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=list(range(1, len(severity_bins)+1)))
    cbar.set_ticklabels([b[0].split('(')[0].strip() for b in severity_bins])
    cbar.set_label('Severity Class')
    ax.set_title('Disturbance Severity Classification', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    plt.tight_layout(); fig.savefig(out_dir / '13_severity_classification.png'); plt.close(fig)

    # ── Figure 14: Intensity × Year Stacked Bar ──
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(years))
    for i, (label, _, _, color) in enumerate(intensity_bins):
        vals = int_year[i, :]
        ax.bar(years, vals, bottom=bottom, color=color, label=label,
               edgecolor='white', linewidth=0.3)
        bottom += vals
    for j, y in enumerate(years):
        total = int_year[:, j].sum()
        if total > 0:
            ax.text(y, total + bottom.max()*0.01, f'{total:,.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xlabel('Year of Detection', fontsize=11)
    ax.set_ylabel('Disturbed Area (ha)', fontsize=11)
    ax.set_title('Annual Disturbance by Intensity Class - Peak Identification', fontsize=13, fontweight='bold')
    ax.set_xticks(years); ax.set_xticklabels(years, rotation=45)
    ax.legend(loc='upper right', fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '14_intensity_by_year_stacked.png'); plt.close(fig)

    # ── Figure 15: Intensity × Year Heatmap ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={'height_ratios': [1, 1]})
    im1 = ax1.imshow(int_year, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax1.set_yticks(range(len(int_labels))); ax1.set_yticklabels(int_labels)
    ax1.set_xticks(range(len(years))); ax1.set_xticklabels(years, rotation=45)
    for i in range(len(int_labels)):
        for j in range(len(years)):
            v = int_year[i, j]
            if v > 0:
                c = 'white' if v > int_year.max() * 0.5 else 'black'
                ax1.text(j, i, f'{v:,.0f}', ha='center', va='center', fontsize=6, color=c)
    plt.colorbar(im1, ax=ax1, shrink=0.8).set_label('Area (ha)')
    ax1.set_title('Intensity × Year - Absolute Area (ha)', fontsize=12, fontweight='bold')
    col_totals = int_year.sum(axis=0, keepdims=True)
    col_totals[col_totals == 0] = 1
    int_pct = int_year / col_totals * 100
    im2 = ax2.imshow(int_pct, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=100)
    ax2.set_yticks(range(len(int_labels))); ax2.set_yticklabels(int_labels)
    ax2.set_xticks(range(len(years))); ax2.set_xticklabels(years, rotation=45)
    for i in range(len(int_labels)):
        for j in range(len(years)):
            v = int_pct[i, j]
            if v > 0:
                c = 'white' if v > 50 else 'black'
                ax2.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=6, color=c)
    plt.colorbar(im2, ax=ax2, shrink=0.8).set_label('% of Year Total')
    ax2.set_title('Intensity × Year - Composition (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year of Detection')
    plt.tight_layout(); fig.savefig(out_dir / '15_intensity_year_heatmap.png'); plt.close(fig)

    # ── Figure 16: Intensity Line Trend + Severity Ratio ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for i, (label, _, _, color) in enumerate(intensity_bins):
        ax1.plot(years, int_year[i, :], color=color, linewidth=2, marker='o', markersize=5, label=label)
    ax1.set_ylabel('Disturbed Area (ha)', fontsize=11)
    ax1.set_title('Disturbance Area by Intensity Class Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9); ax1.grid(alpha=0.3)
    ax2.bar(years, severe_pct, color='#c0392b', alpha=0.7, edgecolor='white')
    ax2.axhline(severe_pct.mean(), color='black', linestyle='--', linewidth=1,
                label=f'Mean={severe_pct.mean():.1f}%')
    for j, y in enumerate(years):
        ax2.text(y, severe_pct[j] + 1, f'{severe_pct[j]:.0f}%', ha='center', fontsize=7)
    ax2.set_xlabel('Year of Detection', fontsize=11)
    ax2.set_ylabel('Severe Disturbance %', fontsize=11)
    ax2.set_title('Annual Proportion of High-Severity Disturbance (High + Very High)',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(years); ax2.set_xticklabels(years, rotation=45)
    ax2.legend(loc='upper right'); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / '16_intensity_trend_severity_ratio.png'); plt.close(fig)

    print(f"    Generated 16 figures in: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LandTrendr Disturbance Raster Analysis (Nanling conventions)")
    p.add_argument("--raster-dir", required=True, help="Directory with YOD/MAG/DUR/MPY TIFFs")
    p.add_argument("--boundary", required=True, help="Study boundary shapefile")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--yod", default="yod_2009_2024.tif")
    p.add_argument("--mag", default="mag_2009_2024.tif")
    p.add_argument("--dur", default="dur_2009_2024.tif")
    p.add_argument("--mpy", default="magperyear_2009_2024.tif")
    p.add_argument("--yod-start", type=int, default=2010)
    p.add_argument("--yod-end", type=int, default=2024)
    p.add_argument("--pixel-ha", type=float, default=0.09)
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = p.parse_args()

    analyze_landtrendr(
        raster_dir=args.raster_dir,
        boundary_shp=args.boundary,
        output_dir=args.output_dir,
        yod_file=args.yod, mag_file=args.mag,
        dur_file=args.dur, mpy_file=args.mpy,
        yod_range=[args.yod_start, args.yod_end],
        pixel_area_ha=args.pixel_ha,
        generate_figures=not args.no_figures,
    )
