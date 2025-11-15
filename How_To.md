# Synthetic Photometry in Johnson-Kron-Cousins Standard Colors from GAIA High Resolution Spectra

This guide explains how to generate synthetic photometry in Johnson-Kron-Cousins (JKC) standard colors using GAIA high resolution spectra.

---

## Requirements

Make sure you have the following Python packages installed:

- `astroquery`
- `astropy`
- `gaiaxpy`

You can install them using pip:

```bash
pip install astroquery astropy gaiaxpy
```

Starting with a known RA and Dec, follow a two step process

First, find the GAIA Source ID.  Notice that the source_id must be a list:
```python
# Landolt Star 114 548 RA 22 41 36.833 Dec +00 59 05.80
RA = 15*(22 + 41/60 + 36.833/3600) * u.deg
Dec = (59/60 + 5.8/3600) * u.deg
coord = SkyCoord(ra=RA, dec=Dec, frame='icrs')
width = u.Quantity(1, u.arcsec)
height = u.Quantity(1, u.arcsec)
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
source_id = [ r['source_id'][0] ]
```
The source_id may be a list of many objects.

# Step 2: Generate the photometry from the BP and RP spectra
```python
synthetic_photometry = generate(source_id, photometric_system=PhotometricSystem.JKC_Std)
```

The columns returned are:
```text
Index(['source_id', 'JkcStd_mag_U', 'JkcStd_mag_B', 'JkcStd_mag_V',
       'JkcStd_mag_R', 'JkcStd_mag_I', 'JkcStd_flux_U', 'JkcStd_flux_B',
       'JkcStd_flux_V', 'JkcStd_flux_R', 'JkcStd_flux_I',
       'JkcStd_flux_error_U', 'JkcStd_flux_error_B', 'JkcStd_flux_error_V',
       'JkcStd_flux_error_R', 'JkcStd_flux_error_I'],
      dtype='object')
```

From this object, we are interested in:
- 'JkcStd_mag_U'
- 'JkcStd_mag_B'
- 'JkcStd_mag_V'
- 'JkcStd_mag_R'
- 'JkcStd_mag_I'

The magnitude errors can also be calculated from the flux and flux error for each color.
```python
flux_V = r['Jkc_Std_flux_V'][0]
flux_error_V = r['Jkc_Std_flux_error_V'][0]
SNR_V = flux_V / flux_error_V
mag_error_V = 2.5 / np.log(10) / SNR_V
```
