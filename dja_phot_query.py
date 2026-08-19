#!/usr/bin/env python
"""
Query the DAWN JWST Archive (DJA) for NIRCam photometry near a given sky
position, and download matching FITS and RGB image stamps.

Uses two parts of the DJA API:
    https://dawn-cph.github.io/dja/general/api_summary/

1. Photometry
   There is no dedicated cone-search endpoint for the DJA/grizli
   photometric catalogs, so photometry is obtained by cone-matching
   against the large merged NIRSpec table (which carries the nearest
   photometric-catalog counterpart for every spectroscopic source):
       https://s3.amazonaws.com/msaexp-nirspec/extractions/
           dja_msaexp_emission_lines_v4.4.csv.gz
   This means photometry is only available for positions that have a
   nearby NIRSpec source in the DJA (it is *not* a general imaging
   cone search). The table is ~130 MB and is cached locally after the
   first download. Fluxes (`phot_{filter}_tot_1` / `_etot_1`) are in
   microjansky (AB zeropoint 23.9).

2. Image stamps
   The `grizli-cutout` "thumb" endpoint:
       https://grizli-cutout.herokuapp.com/thumb
           ?coords={ra},{dec}&size={size_arcsec}&filters=...&output=fits
   returns a multi-extension FITS cutout (one image HDU per filter).
   Leaving off `output` (or `output=png`) instead returns an RGB PNG
   thumbnail built from three filters.

Examples
--------
Command line:
    python dja_phot_query.py 34.2775 -5.2282 --radius 1.0
    python dja_phot_query.py 34.2775 -5.2282 --sizes 2 4 8 --outdir stamps

As a module:
    from dja_phot_query import query_dja_photometry, download_fits_stamps, download_rgb_thumbnails
    phot = query_dja_photometry(ra=34.2775, dec=-5.2282, radius_arcsec=1.0)
    download_fits_stamps(34.2775, -5.2282, sizes_arcsec=[2, 4, 8], outdir="stamps")
    download_rgb_thumbnails(34.2775, -5.2282, sizes_arcsec=[2, 4, 8], outdir="stamps")
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

THUMB_URL = "https://grizli-cutout.herokuapp.com/thumb"
PHOT_TABLE_URL = (
    "https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    "dja_msaexp_emission_lines_v4.4.csv.gz"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dja_query"

PHOT_FILTERS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f410m", "f444w"]
DEFAULT_FITS_FILTERS = [f"{f}-clear" for f in PHOT_FILTERS]
DEFAULT_RGB_FILTERS = ["f115w-clear", "f277w-clear", "f444w-clear"]

_phot_table_cache = {}


# --------------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------------

def _download_with_progress(url, outpath, timeout=600):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {outpath.name}"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.rename(outpath)
    return outpath


def _load_phot_table(cache_dir=None, refresh=False):
    """Load (downloading + caching to disk if needed) the merged DJA table."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    local_path = cache_dir / Path(PHOT_TABLE_URL).name

    if not refresh and local_path in _phot_table_cache:
        return _phot_table_cache[local_path]

    if refresh or not local_path.exists():
        _download_with_progress(PHOT_TABLE_URL, local_path)

    df = pd.read_csv(local_path, compression="gzip", low_memory=False)
    _phot_table_cache[local_path] = df
    return df


def query_dja_photometry(ra, dec, radius_arcsec=1.0, cache_dir=None, refresh=False):
    """
    Cone-search the merged DJA NIRSpec table for NIRCam photometry.

    Parameters
    ----------
    ra, dec : float
        Search center in decimal degrees.
    radius_arcsec : float
        Search radius in arcseconds.
    cache_dir : str or Path, optional
        Where to cache the ~130 MB merged table (default: ~/.cache/dja_query).
    refresh : bool
        Re-download the merged table even if a cached copy exists.

    Returns
    -------
    pandas.DataFrame
        Matching rows (including `phot_{filter}_tot_1` / `_etot_1` flux
        columns in microjansky, and a `sep_arcsec` separation column),
        sorted by separation. Empty DataFrame if nothing is found.
    """
    df = _load_phot_table(cache_dir=cache_dir, refresh=refresh)

    dra = (df["ra"] - ra) * np.cos(np.radians(dec))
    ddec = df["dec"] - dec
    sep_arcsec = np.hypot(dra, ddec) * 3600.0

    matches = df.loc[sep_arcsec <= radius_arcsec].copy()
    matches["sep_arcsec"] = sep_arcsec[sep_arcsec <= radius_arcsec]
    matches = matches.sort_values("sep_arcsec").reset_index(drop=True)
    return matches


def get_nircam_fluxes(row, filters=PHOT_FILTERS):
    """
    Extract a tidy (filter, flux_ujy, eflux_ujy) table from one row of
    query_dja_photometry() output.
    """
    records = []
    for filt in filters:
        col, ecol = f"phot_{filt}_tot_1", f"phot_{filt}_etot_1"
        if col in row and pd.notna(row[col]):
            records.append({"filter": filt, "flux_ujy": row[col], "eflux_ujy": row.get(ecol)})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Image stamps
# --------------------------------------------------------------------------

def _coord_str(ra, dec):
    return f"{ra},{dec}"


def download_fits_stamp(ra, dec, size_arcsec=4.0, filters=DEFAULT_FITS_FILTERS,
                         weight=False, outfile=None, outdir=".", timeout=180):
    """
    Download a multi-extension FITS cutout (one image HDU per filter).

    Parameters
    ----------
    ra, dec : float
        Center in decimal degrees.
    size_arcsec : float
        Cutout side length in arcseconds.
    filters : list[str]
        Filter/pupil combinations, e.g. "f200w-clear".
    weight : bool
        If True, also fetch inverse-variance weight extensions (`output=fits_weight`).
    outfile : str or Path, optional
        Explicit output path. If not given, a name is built from position and size.
    outdir : str or Path
        Directory to save into when `outfile` is not given.

    Returns
    -------
    Path
        Path to the saved FITS file.
    """
    params = {
        "coords": _coord_str(ra, dec),
        "size": size_arcsec,
        "filters": ",".join(filters),
        "output": "fits_weight" if weight else "fits",
    }
    resp = requests.get(THUMB_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    if outfile is None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"dja_stamp_{ra:.6f}_{dec:.6f}_{size_arcsec:g}arcsec.fits"
    outfile = Path(outfile)
    outfile.write_bytes(resp.content)
    return outfile


def download_fits_stamps(ra, dec, sizes_arcsec=(2, 4, 8), **kwargs):
    """Download FITS cutouts at multiple sizes. See download_fits_stamp() for kwargs."""
    return [download_fits_stamp(ra, dec, size_arcsec=size, **kwargs) for size in sizes_arcsec]


def download_rgb_thumbnail(ra, dec, size_arcsec=4.0, filters=DEFAULT_RGB_FILTERS,
                            rgb_scl=None, asinh=True, scl=None, invert=False,
                            outfile=None, outdir=".", timeout=180):
    """
    Download an RGB PNG thumbnail built from three filters.

    Parameters
    ----------
    ra, dec : float
        Center in decimal degrees.
    size_arcsec : float
        Cutout side length in arcseconds.
    filters : list[str]
        Exactly three filter/pupil combinations, mapped to R, G, B.
    rgb_scl : list[float], optional
        Per-channel scaling, e.g. [1.0, 2.0, 1.01].
    asinh : bool
        Use asinh scaling instead of the default Lupton (2004) scaling.
    scl : float, optional
        Overall contrast scaling.
    invert : bool
        Invert the image (light background).
    outfile : str or Path, optional
        Explicit output path. If not given, a name is built from position and size.
    outdir : str or Path
        Directory to save into when `outfile` is not given.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    params = {
        "coords": _coord_str(ra, dec),
        "size": size_arcsec,
        "filters": ",".join(filters),
        "asinh": asinh,
    }
    if rgb_scl is not None:
        params["rgb_scl"] = ",".join(str(x) for x in rgb_scl)
    if scl is not None:
        params["scl"] = scl
    if invert:
        params["invert"] = True

    resp = requests.get(THUMB_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    if outfile is None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"dja_rgb_{ra:.6f}_{dec:.6f}_{size_arcsec:g}arcsec.png"
    outfile = Path(outfile)
    outfile.write_bytes(resp.content)
    return outfile


def download_rgb_thumbnails(ra, dec, sizes_arcsec=(2, 4, 8), **kwargs):
    """Download RGB PNG thumbnails at multiple sizes. See download_rgb_thumbnail() for kwargs."""
    return [download_rgb_thumbnail(ra, dec, size_arcsec=size, **kwargs) for size in sizes_arcsec]


def main():
    parser = argparse.ArgumentParser(
        description="Query DJA NIRCam photometry and download FITS/RGB image stamps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ra", type=float, help="RA in decimal degrees")
    parser.add_argument("dec", type=float, help="Dec in decimal degrees")
    parser.add_argument(
        "--radius", type=float, default=1.0,
        help="Photometry search radius in arcsec (default: 1.0)",
    )
    parser.add_argument(
        "--sizes", type=float, nargs="+", default=[2.0, 4.0, 8.0],
        help="Cutout sizes in arcsec for FITS/RGB stamps (default: 2 4 8)",
    )
    parser.add_argument(
        "--fits-filters", nargs="+", default=DEFAULT_FITS_FILTERS,
        help="Filters to include in the FITS cutouts",
    )
    parser.add_argument(
        "--rgb-filters", nargs="+", default=DEFAULT_RGB_FILTERS,
        help="Three filters (R,G,B) for the RGB thumbnails",
    )
    parser.add_argument(
        "--weight", action="store_true",
        help="Also fetch inverse-variance weight extensions in the FITS cutouts",
    )
    parser.add_argument(
        "--outdir", default="dja_stamps",
        help="Output directory for downloaded stamps (default: dja_stamps)",
    )
    parser.add_argument("--no-phot", action="store_true", help="Skip the photometry query")
    parser.add_argument("--no-fits", action="store_true", help="Skip FITS cutout downloads")
    parser.add_argument("--no-rgb", action="store_true", help="Skip RGB thumbnail downloads")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Where to cache the merged photometry table (default: ~/.cache/dja_query)",
    )
    parser.add_argument(
        "--refresh-table", action="store_true",
        help="Re-download the merged photometry table even if cached",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only print the photometry match, do not download any stamps",
    )
    args = parser.parse_args()

    if not args.no_phot:
        print(f"Querying DJA photometry at RA={args.ra}, Dec={args.dec}, "
              f"radius={args.radius}\" ...")
        phot = query_dja_photometry(args.ra, args.dec, args.radius,
                                     cache_dir=args.cache_dir, refresh=args.refresh_table)
        if len(phot) == 0:
            print("No photometry match found (only positions with a nearby "
                  "NIRSpec source in DJA have photometry via this method).")
        else:
            show_cols = [c for c in ["srcid", "root", "ra", "dec", "sep_arcsec", "z", "grade"]
                         if c in phot.columns]
            print(f"\nFound {len(phot)} matching source(s):\n")
            print(phot[show_cols].to_string(index=False))

            fluxes = get_nircam_fluxes(phot.iloc[0])
            if len(fluxes):
                print(f"\nNIRCam photometry for nearest match "
                      f"(sep={phot.iloc[0]['sep_arcsec']:.3f}\", flux in uJy):\n")
                print(fluxes.to_string(index=False))

    if args.list_only:
        return

    if not args.no_fits:
        print(f"\nDownloading FITS stamps to '{args.outdir}/' ...")
        paths = download_fits_stamps(
            args.ra, args.dec, sizes_arcsec=args.sizes, filters=args.fits_filters,
            weight=args.weight, outdir=args.outdir,
        )
        for p in paths:
            print(f"  saved {p}")

    if not args.no_rgb:
        print(f"\nDownloading RGB thumbnails to '{args.outdir}/' ...")
        paths = download_rgb_thumbnails(
            args.ra, args.dec, sizes_arcsec=args.sizes, filters=args.rgb_filters,
            outdir=args.outdir,
        )
        for p in paths:
            print(f"  saved {p}")


if __name__ == "__main__":
    main()
