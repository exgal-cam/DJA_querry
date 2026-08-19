# DJA_querry

Python tools for querying the [DAWN JWST Archive (DJA)](https://dawn-cph.github.io/dja/) for reduced NIRSpec MSA spectra and NIRCam photometry/imaging near a given sky position, and downloading the matching data.

Built on top of the DJA [`grizli-cutout` API](https://dawn-cph.github.io/dja/general/api_summary/).

## Contents

- [`dja_nirspec_query.py`](dja_nirspec_query.py) — cone-search the DJA NIRSpec extraction database and download matching `.spec.fits` spectra.
- [`dja_phot_query.py`](dja_phot_query.py) — cone-search NIRCam photometry (via the merged NIRSpec+photometry table) and download FITS cutouts / RGB PNG thumbnails.
- [`DJA_querry_show.ipynb`](DJA_querry_show.ipynb) — example notebook demonstrating both modules on a sample catalog.

## Requirements

- Python 3
- `pandas`, `numpy`, `requests`, `tqdm`

```bash
pip install pandas numpy requests tqdm
```

## Usage

### NIRSpec spectra

```bash
python dja_nirspec_query.py <ra> <dec> --radius 1.0
python dja_nirspec_query.py 34.2775 -5.2282 --radius 2.0 --grade-min 2 --outdir spectra
```

As a module:

```python
from dja_nirspec_query import query_dja_nirspec, download_spectra

df = query_dja_nirspec(ra=34.2775, dec=-5.2282, radius_arcsec=1.0)
download_spectra(df, outdir="spectra")
```

`grade` is a visual-inspection redshift-quality flag (3 = robust, 0 = no useful redshift); spectra with lower grades still exist and are downloadable.

### NIRCam photometry & image stamps

```bash
python dja_phot_query.py <ra> <dec> --radius 1.0
python dja_phot_query.py 34.2775 -5.2282 --sizes 2 4 8 --outdir stamps
```

As a module:

```python
from dja_phot_query import query_dja_photometry, download_fits_stamps, download_rgb_thumbnails

phot = query_dja_photometry(ra=34.2775, dec=-5.2282, radius_arcsec=1.0)
download_fits_stamps(34.2775, -5.2282, sizes_arcsec=[2, 4, 8], outdir="stamps")
download_rgb_thumbnails(34.2775, -5.2282, sizes_arcsec=[2, 4, 8], outdir="stamps")
```

Notes:

- There is no dedicated cone-search endpoint for the DJA photometric catalogs, so photometry is obtained by matching against the merged NIRSpec table (`dja_msaexp_emission_lines_v4.4.csv.gz`), which carries the nearest photometric counterpart for every spectroscopic source. **This means photometry is only available near existing NIRSpec sources — it is not a general imaging cone search.**
- That table is ~130 MB and is cached locally after the first download (default: `~/.cache/dja_query`).
- Fluxes (`phot_{filter}_tot_1` / `_etot_1`) are in microjansky (AB zeropoint 23.9).
- FITS cutouts are multi-extension (one image HDU per filter); RGB thumbnails are built from three filters as a PNG.

## Data sources

- Spectra: `https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}`
- Cone-search / cutout API: `https://grizli-cutout.herokuapp.com/`
- DJA API reference: https://dawn-cph.github.io/dja/general/api_summary/
