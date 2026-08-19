#!/usr/bin/env python
"""
Query the DAWN JWST Archive (DJA) for reduced NIRSpec MSA spectra near a
given sky position, and download the matching spectrum FITS files.

Uses the DJA "grizli-cutout" API:
    https://dawn-cph.github.io/dja/general/api_summary/

Endpoint used:
    https://grizli-cutout.herokuapp.com/nirspec_extractions
        ?coords={ra},{dec}&size={radius_arcsec}&output=csv

Each matching row gives a `root` and `file`; the FITS spectrum itself
lives at:
    https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}

Examples
--------
Command line:
    python dja_nirspec_query.py 34.2775 -5.2282 --radius 1.0
    python dja_nirspec_query.py 34.2775 -5.2282 --radius 2.0 --grade-min 2 --outdir spectra

As a module:
    from dja_nirspec_query import query_dja_nirspec, download_spectra
    df = query_dja_nirspec(ra=34.2775, dec=-5.2282, radius_arcsec=1.0)
    download_spectra(df, outdir="spectra")
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_URL = "https://grizli-cutout.herokuapp.com/nirspec_extractions"
S3_BASE = "https://s3.amazonaws.com/msaexp-nirspec/extractions"


def query_dja_nirspec(ra, dec, radius_arcsec=1.0, grade_min=None, timeout=60):
    """
    Cone-search the DJA NIRSpec extraction database.

    Parameters
    ----------
    ra, dec : float
        Search center in decimal degrees.
    radius_arcsec : float
        Search radius in arcseconds (the API's `size` parameter).
    grade_min : float, optional
        If given, keep only rows with `grade` >= this value. Grades are
        visual-inspection redshift-quality flags, 3 = robust, 0 = no
        useful redshift. Most spectra (grade < 3) still exist and are
        downloadable even without a reliable redshift.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        One row per matched spectrum (a given source can appear multiple
        times: once per grating/filter, and again if observed in more
        than one mask/root). Empty DataFrame if nothing is found.
    """
    params = {"coords": f"{ra},{dec}", "size": radius_arcsec, "output": "csv"}
    resp = requests.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(text))

    if len(df) and grade_min is not None and "grade" in df.columns:
        df = df[df["grade"] >= grade_min].reset_index(drop=True)

    return df


def download_spectra(df, outdir="dja_spectra", overwrite=False, timeout=120):
    """
    Download the `.spec.fits` files referenced in a query_dja_nirspec() result.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of query_dja_nirspec(); must have `root` and `file` columns.
    outdir : str or Path
        Local directory to save files into (created if needed).
    overwrite : bool
        Re-download files that already exist locally.

    Returns
    -------
    list[Path]
        Paths to the successfully downloaded (or already-present) files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading spectra"):
        root, fname = row["root"], row["file"]
        url = f"{S3_BASE}/{root}/{fname}"
        outpath = outdir / fname

        if outpath.exists() and not overwrite:
            saved.append(outpath)
            continue

        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            print(f"  ! failed ({r.status_code}): {url}", file=sys.stderr)
            continue

        outpath.write_bytes(r.content)
        saved.append(outpath)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Cone-search the DJA and download matching NIRSpec spectra.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ra", type=float, help="RA in decimal degrees")
    parser.add_argument("dec", type=float, help="Dec in decimal degrees")
    parser.add_argument(
        "--radius", type=float, default=1.0,
        help="Search radius in arcsec (default: 1.0)",
    )
    parser.add_argument(
        "--grade-min", type=float, default=None,
        help="Keep only spectra with redshift grade >= this value (0-3)",
    )
    parser.add_argument(
        "--outdir", default="dja_spectra",
        help="Output directory for downloaded FITS files (default: dja_spectra)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download files that already exist locally",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only print the matches, do not download anything",
    )
    args = parser.parse_args()

    print(f"Querying DJA at RA={args.ra}, Dec={args.dec}, radius={args.radius}\" ...")
    df = query_dja_nirspec(args.ra, args.dec, args.radius, args.grade_min)

    if len(df) == 0:
        print("No NIRSpec spectra found at this position "
              "(try a larger --radius, or double-check the coordinates).")
        sys.exit(0)

    show_cols = [c for c in ["root", "file", "grating", "filter", "z", "grade", "sn50"]
                 if c in df.columns]
    print(f"\nFound {len(df)} matching spectra:\n")
    print(df[show_cols].to_string(index=False))

    if args.list_only:
        return

    print(f"\nDownloading to '{args.outdir}/' ...")
    files = download_spectra(df, args.outdir, args.overwrite)
    print(f"\nDone: {len(files)}/{len(df)} files saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()
