# -*- coding: utf-8 -*-
"""
Compare two gridded CMIP7 emissions NetCDF files.

Computes and compares:
  - Total global sum (Mt/yr) over time
  - Total global sum by sector over time
  - Min/max gridpoint and total values
  - Saves comparison and difference figures

Usage:
    python compare_two_cmip7_files.py <file_a> <file_b> [--output-dir <dir>]

Example:
    python compare_two_cmip7_files.py ^
        "results/vl_1-1-0-alpha/CO-em-anthro_..._gn_202201-210012.nc" ^
        "results/vl_1-1-0-alpha-old/CO-em-anthro_..._gn_202201-210012.nc"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Helper: compute global annual emissions (Mt/yr), optionally by sector
# ---------------------------------------------------------------------------
def global_annual_emissions(ds, cell_area, keep_sectors=False):
    """
    Convert a gridded emissions dataset (kg s⁻¹ m⁻²) to global annual
    totals in Mt/yr.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded emissions dataset with one data variable.
    cell_area : xr.DataArray
        areacella (m²), shape (lat, lon).
    keep_sectors : bool
        If True, return totals per sector; otherwise sum over sectors.

    Returns
    -------
    xr.DataArray
        Global annual emissions in Mt/yr.
    """
    # Get the single data variable
    (var_name,) = [v for v in ds.data_vars if not v.endswith("_bnds") and v != "areacella"]
    da = ds[var_name]

    # seconds per month
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60

    # kg m⁻² s⁻¹  →  kg / cell / month
    monthly_kg = seconds_per_month * cell_area * da

    # sum over spatial dims
    sum_dims = ["lat", "lon"]
    if "level" in monthly_kg.dims:
        sum_dims.append("level")
    global_monthly = monthly_kg.sum(dim=sum_dims)

    # monthly → annual
    annual_kg = global_monthly.groupby("time.year").sum()

    # kg → Mt (= Tg)
    annual_Mt = annual_kg * 1e-9

    if "sector" in annual_Mt.dims and not keep_sectors:
        annual_Mt = annual_Mt.sum(dim="sector")

    return annual_Mt, var_name


# ---------------------------------------------------------------------------
# Helper: resolve sector names from dataset attributes
# ---------------------------------------------------------------------------
def get_sector_names(ds, var_name):
    """Try to extract sector names from the sector_ordering attribute."""
    if "sector" not in ds[var_name].dims:
        return None

    # Try different attribute names where sector labels may be stored
    for attr in ["sector_ordering", "sector_names"]:
        ordering = ds.attrs.get(attr, None) or ds[var_name].attrs.get(attr, None)
        if ordering is not None:
            if isinstance(ordering, str):
                return [s.strip() for s in ordering.split(",")]
            return list(ordering)

    # Fallback: integer labels
    n_sectors = ds.sizes.get("sector", 0)
    return [f"Sector {i}" for i in range(n_sectors)]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------
def compare_files(file_a, file_b, areacella_path, output_dir):
    """Run the full comparison between two NetCDF files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load areacella
    area_ds = xr.open_dataset(areacella_path)
    cell_area = area_ds["areacella"]

    # Load the two files
    ds_a = xr.open_dataset(file_a)
    ds_b = xr.open_dataset(file_b)

    label_a = Path(file_a).parent.name  # e.g. folder name as label
    label_b = Path(file_b).parent.name
    # If both come from same folder, use filename
    if label_a == label_b:
        label_a = Path(file_a).stem[:40] + " (A)"
        label_b = Path(file_b).stem[:40] + " (B)"

    # ------------------------------------------------------------------
    # 1. Total global sum
    # ------------------------------------------------------------------
    total_a, var_a = global_annual_emissions(ds_a, cell_area, keep_sectors=False)
    total_b, var_b = global_annual_emissions(ds_b, cell_area, keep_sectors=False)

    gas_name = var_a.split("_")[0]

    print(f"\n{'='*80}")
    print(f"COMPARISON: {gas_name} gridded emissions")
    print(f"  File A: {file_a}")
    print(f"  File B: {file_b}")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # 2. Total global sum by sector
    # ------------------------------------------------------------------
    sector_a, _ = global_annual_emissions(ds_a, cell_area, keep_sectors=True)
    sector_b, _ = global_annual_emissions(ds_b, cell_area, keep_sectors=True)

    sector_names = get_sector_names(ds_a, var_a)

    # ------------------------------------------------------------------
    # 3. Min / max gridpoint and total values
    # ------------------------------------------------------------------
    print("─" * 80)
    print("GLOBAL TOTAL EMISSIONS (Mt/yr)")
    print("─" * 80)
    print(f"{'':>8} {'File A':>16} {'File B':>16} {'Diff (A-B)':>16} {'Diff %':>12}")
    print("-" * 72)
    for year_val in [total_a.year.values[0], total_a.year.values[-1]]:
        va = float(total_a.sel(year=year_val))
        vb = float(total_b.sel(year=year_val))
        diff = va - vb
        pct = (diff / vb * 100) if abs(vb) > 1e-15 else float("inf")
        print(f"{year_val:>8} {va:>16.6f} {vb:>16.6f} {diff:>16.6f} {pct:>11.4f}%")

    print(f"\n{'─'*80}")
    print("GRIDPOINT STATISTICS (raw values, kg s⁻¹ m⁻²)")
    print("─" * 80)
    da_a = ds_a[var_a]
    da_b = ds_b[var_b]

    stats = {
        "File A min": float(da_a.min()),
        "File A max": float(da_a.max()),
        "File B min": float(da_b.min()),
        "File B max": float(da_b.max()),
    }
    for k, v in stats.items():
        print(f"  {k:<20}: {v:>14.6e}")

    # Difference stats (aligned on common coords)
    diff_da = da_a - da_b
    print(f"\n  Diff min           : {float(diff_da.min()):>14.6e}")
    print(f"  Diff max           : {float(diff_da.max()):>14.6e}")
    print(f"  Diff mean          : {float(diff_da.mean()):>14.6e}")
    print(f"  Diff abs max       : {float(np.abs(diff_da).max()):>14.6e}")

    # ------------------------------------------------------------------
    # 4. Figure: total comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 4a. Total global sum
    ax = axes[0]
    total_a.plot(ax=ax, label=label_a, linewidth=2)
    total_b.plot(ax=ax, label=label_b, linewidth=2, linestyle="--")
    ax.set_title(f"{gas_name} – Global Total Emissions")
    ax.set_ylabel("Mt / yr")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4b. By sector
    ax = axes[1]
    if "sector" in sector_a.dims:
        n_sectors = sector_a.sizes["sector"]
        for i in range(n_sectors):
            sname = sector_names[i] if sector_names and i < len(sector_names) else f"Sector {i}"
            sector_a.isel(sector=i).plot(ax=ax, label=f"{sname} (A)", linewidth=1.5)
            sector_b.isel(sector=i).plot(ax=ax, label=f"{sname} (B)", linewidth=1.5, linestyle="--")
        ax.set_title(f"{gas_name} – Global Emissions by Sector")
    else:
        total_a.plot(ax=ax, label=label_a, linewidth=2)
        total_b.plot(ax=ax, label=label_b, linewidth=2, linestyle="--")
        ax.set_title(f"{gas_name} – Global Total (no sector dim)")
    ax.set_ylabel("Mt / yr")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_total = output_dir / f"compare_totals_{gas_name}.png"
    fig.savefig(fig_total, dpi=150)
    print(f"\n📊 Totals figure saved to: {fig_total}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Figure: differences
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 5a. Total difference
    ax = axes[0]
    diff_total = total_a - total_b
    diff_total.plot(ax=ax, color="crimson", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"{gas_name} – Difference in Global Total (A − B)")
    ax.set_ylabel("Mt / yr")
    ax.grid(True, alpha=0.3)

    # 5b. By sector difference
    ax = axes[1]
    if "sector" in sector_a.dims:
        diff_sector = sector_a - sector_b
        n_sectors = diff_sector.sizes["sector"]
        for i in range(n_sectors):
            sname = sector_names[i] if sector_names and i < len(sector_names) else f"Sector {i}"
            diff_sector.isel(sector=i).plot(ax=ax, label=sname, linewidth=1.5)
        ax.set_title(f"{gas_name} – Difference by Sector (A − B)")
    else:
        diff_total.plot(ax=ax, color="crimson", linewidth=2)
        ax.set_title(f"{gas_name} – Difference (no sector dim)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Mt / yr")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_diff = output_dir / f"compare_differences_{gas_name}.png"
    fig.savefig(fig_diff, dpi=150)
    print(f"📊 Differences figure saved to: {fig_diff}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Print sector-level summary table
    # ------------------------------------------------------------------
    if "sector" in sector_a.dims:
        print(f"\n{'─'*80}")
        print(f"SECTOR-LEVEL TOTALS — first year ({int(sector_a.year.values[0])}) "
              f"and last year ({int(sector_a.year.values[-1])})")
        print("─" * 80)
        first_year = int(sector_a.year.values[0])
        last_year = int(sector_a.year.values[-1])
        
        print(f"{'Sector':<35} {'Year':>6} {'A (Mt/yr)':>12} {'B (Mt/yr)':>12} {'Diff':>12} {'Diff %':>10}")
        print("-" * 90)
        for i in range(sector_a.sizes["sector"]):
            sname = sector_names[i] if sector_names and i < len(sector_names) else f"Sector {i}"
            for yr in [first_year, last_year]:
                va = float(sector_a.isel(sector=i).sel(year=yr))
                vb = float(sector_b.isel(sector=i).sel(year=yr))
                diff = va - vb
                pct = (diff / vb * 100) if abs(vb) > 1e-15 else (0.0 if abs(va) < 1e-15 else float("inf"))
                pct_str = f"{pct:.4f}%" if abs(pct) != float("inf") else "inf"
                print(f"{sname:<35} {yr:>6} {va:>12.6f} {vb:>12.6f} {diff:>12.6f} {pct_str:>10}")

    # Cleanup
    ds_a.close()
    ds_b.close()
    area_ds.close()

    print(f"\n✅ Comparison complete. Outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare two CMIP7 gridded emissions NetCDF files."
    )
    parser.add_argument("file_a", type=str, help="Path to first NetCDF file")
    parser.add_argument("file_b", type=str, help="Path to second NetCDF file")
    parser.add_argument(
        "--areacella", type=str, default=None,
        help="Path to areacella NetCDF file. If not given, tries to find it "
             "from the concordia settings."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save figures. Defaults to parent of file_a."
    )

    args = parser.parse_args()

    # Resolve areacella
    areacella_path = args.areacella
    if areacella_path is None:
        # Try to find it via concordia settings
        try:
            from concordia.settings import Settings
            config_candidates = [
                Path(__file__).parent.parent / "notebooks" / "cmip7" / "config_cmip7_v0-4-0.yaml",
                Path.cwd() / "notebooks" / "cmip7" / "config_cmip7_v0-4-0.yaml",
            ]
            for cfg in config_candidates:
                if cfg.exists():
                    settings = Settings.from_config(version="compare", local_config_path=cfg)
                    areacella_path = settings.gridding_path / "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"
                    if Path(areacella_path).exists():
                        break
        except Exception:
            pass

    if areacella_path is None or not Path(areacella_path).exists():
        parser.error(
            "Could not find areacella file. Please provide --areacella path."
        )

    output_dir = args.output_dir or str(Path(args.file_a).parent / "comparison")

    compare_files(args.file_a, args.file_b, areacella_path, output_dir)


if __name__ == "__main__":
    main()
