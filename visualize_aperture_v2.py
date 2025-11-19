# visualize_aperture.py 2025-09-17
# John Phillips, john.d.phillips@comcast.net

"""
visualize_aperture.py:
A standalone application to visualize a target star field with GAIA DR3 sources,
simulating their PSF brightness profiles using 2D Gaussians. It displays
two side-by-side plots: one with the target star and all neighboring sources,
and a second with the target star removed to clearly highlight "third lights"
that may fall within the photometry aperture or sky annulus.

Users input target RA/Dec, image scale, seeing (FWHM), and aperture/annulus
radii. The field of view for the plots can be custom-defined or
automatically calculated based on the aperture/annulus sizes. It also calculates
and reports key flux contamination ratios.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.modeling.models import Gaussian2D
from astropy.nddata import block_reduce
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.error

# Oversampling factor ensures each star's Gaussian PSF is rendered on a finer grid
# before being rebinned to the working resolution, preserving profile fidelity.
OVERSAMPLING_FACTOR = 5
DEBUG_PRINT_GAIA_ROW = False
MAMAJEK_TABLE_URL = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
MAMAJEK_CACHE_FILENAME = "EEM_dwarf_UBVIJHK_colors_Teff.txt"
MAMAJEK_CACHE_DIR = ".cache"
MAMAJEK_REFRESH_DAYS = 14
_MAMAJEK_ROWS = None


def _as_float(value):
    """Return the given column value as a float, or None if invalid/masked."""
    if value is None:
        return None
    if np.ma.is_masked(value):
        return None
    # Allow astropy Quantity values (e.g., distances) to be converted to plain floats
    if hasattr(value, "to_value"):
        try:
            value = value.to_value()
        except Exception:
            try:
                value = value.value
            except Exception:
                return None
    elif hasattr(value, "value") and not isinstance(value, (float, int, np.floating, np.integer)):
        try:
            value = value.value
        except Exception:
            return None
    try:
        float_val = float(value)
    except Exception:
        return None
    if not np.isfinite(float_val):
        return None
    return float_val


def _download_mamajek_table(target_path: Path):
    try:
        with urllib.request.urlopen(MAMAJEK_TABLE_URL, timeout=30) as response, open(target_path, "wb") as out_f:
            out_f.write(response.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download Mamajek dwarf table: {e}") from e


def _load_mamajek_table(cache_dir=MAMAJEK_CACHE_DIR, refresh_days=MAMAJEK_REFRESH_DAYS) -> Path:
    cache_dir_path = Path(cache_dir)
    if not cache_dir_path.is_absolute():
        cache_dir_path = Path(__file__).resolve().parent / cache_dir_path
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir_path / MAMAJEK_CACHE_FILENAME

    needs_refresh = True
    if cache_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age < timedelta(days=refresh_days):
            needs_refresh = False
    if needs_refresh:
        try:
            _download_mamajek_table(cache_file)
        except Exception as ex:
            if cache_file.exists():
                print(f"Warning: Could not refresh Mamajek table ({ex}). Using cached copy.")
            else:
                raise
    return cache_file


def _parse_mamajek_table(table_path: Path):
    header_map = {}
    rows = []
    try:
        with open(table_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if line.startswith("#SpT"):
                        header_tokens = line.lstrip("#").split()
                        header_map = {token: idx for idx, token in enumerate(header_tokens)}
                    continue
                if not header_map:
                    continue
                parts = line.split()
                sp_idx = header_map.get("SpT")
                if sp_idx is None or sp_idx >= len(parts):
                    continue
                spectral_type = parts[sp_idx]
                if "V" not in spectral_type:
                    continue

                def extract_float(column_name):
                    idx = header_map.get(column_name)
                    if idx is None or idx >= len(parts):
                        return None
                    value = parts[idx]
                    if value in {"...", "....."}:
                        return None
                    try:
                        return float(value)
                    except ValueError:
                        return None

                bp_rp = extract_float("Bp-Rp")
                m_g = extract_float("M_G")
                teff = extract_float("Teff")
                if bp_rp is None or m_g is None:
                    continue
                rows.append(
                    {
                        "spectral_type": spectral_type,
                        "bp_rp": bp_rp,
                        "M_G": m_g,
                        "Teff": teff,
                    }
                )
    except FileNotFoundError as e:
        raise RuntimeError(f"Mamajek table not found at {table_path}") from e
    rows.sort(key=lambda r: r["bp_rp"])
    return rows


def _get_mamajek_rows():
    global _MAMAJEK_ROWS
    if _MAMAJEK_ROWS is None:
        try:
            table_path = _load_mamajek_table()
            _MAMAJEK_ROWS = _parse_mamajek_table(table_path)
        except Exception as ex:
            print(f"Warning: Unable to load Mamajek spectral table ({ex}).")
            _MAMAJEK_ROWS = []
    return _MAMAJEK_ROWS


def _classify_star(abs_mag, adj_color):
    if abs_mag is None or adj_color is None:
        return None
    rows = _get_mamajek_rows()
    if not rows:
        return None

    color_sigma = 0.03
    mag_sigma = 0.5
    best_entry = None
    best_score = None
    best_color_diff = None
    best_mag_diff = None

    for row in rows:
        color_diff = adj_color - row["bp_rp"]
        mag_diff = abs_mag - row["M_G"]
        score = (color_diff / color_sigma) ** 2 + (mag_diff / mag_sigma) ** 2
        if best_score is None or score < best_score:
            best_score = score
            best_entry = row
            best_color_diff = color_diff
            best_mag_diff = mag_diff

    if best_entry is None:
        return None

    return {
        "label": best_entry["spectral_type"],
        "bp_rp": best_entry["bp_rp"],
        "M_G": best_entry["M_G"],
        "Teff": best_entry["Teff"],
        "color_diff": best_color_diff,
        "mag_diff": best_mag_diff,
        "score": best_score,
    }


def _column_value(row, *names):
    """Return the first available column value for the provided names."""
    for name in names:
        if name in row.colnames:
            return _as_float(row[name])
    return None


def _column_text(row, *names):
    """Return the first available column value as text, or None."""
    for name in names:
        if name in row.colnames:
            value = row[name]
            if np.ma.is_masked(value):
                continue
            return str(value)
    return None


def _parse_ra_input(value: str) -> float:
    """Parse RA input in decimal degrees or sexagesimal (hh:mm:ss.s) into degrees."""
    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass

    separators = [":", " "]
    for sep in separators:
        if sep in stripped:
            parts = stripped.split(sep)
            break
    else:
        raise ValueError(f"Invalid RA input: '{value}'")

    if len(parts) != 3:
        raise ValueError(f"Invalid RA sexagesimal format: '{value}'")

    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Invalid RA sexagesimal components: '{value}'") from exc

    if not (0 <= minutes < 60) or not (0 <= seconds < 60):
        raise ValueError(f"Invalid RA sexagesimal range: '{value}'")

    total_hours = hours + minutes / 60.0 + seconds / 3600.0
    return total_hours * 15.0


def _parse_dec_input(value: str) -> float:
    """Parse Dec input in decimal degrees or sexagesimal (±dd:mm:ss.s) into degrees."""
    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass

    sign = 1.0
    if stripped.startswith("+"):
        stripped = stripped[1:]
    elif stripped.startswith("-"):
        stripped = stripped[1:]
        sign = -1.0

    separators = [":", " "]
    for sep in separators:
        if sep in stripped:
            parts = stripped.split(sep)
            break
    else:
        raise ValueError(f"Invalid Dec input: '{value}'")

    if len(parts) != 3:
        raise ValueError(f"Invalid Dec sexagesimal format: '{value}'")

    try:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Invalid Dec sexagesimal components: '{value}'") from exc

    if not (0 <= minutes < 60) or not (0 <= seconds < 60):
        raise ValueError(f"Invalid Dec sexagesimal range: '{value}'")

    total_degrees = degrees + minutes / 60.0 + seconds / 3600.0
    return sign * total_degrees


def _fetch_gaiaxpy_photometry(source_id_value):
    """Fetch GaiaXPy synthetic photometry (JKC) for a Gaia source_id."""
    if source_id_value is None:
        return None
    try:
        from gaiaxpy import generate  # type: ignore
    except ImportError:
        print("GaiaXPy package is not installed; skipping synthetic photometry.")
        return None

    try:
        source_id_int = int(str(source_id_value).strip())
    except Exception:
        print(f"Invalid Gaia source ID '{source_id_value}' for GaiaXPy; skipping synthetic photometry.")
        return None

    try:
        result = generate(
            source_id=[source_id_int],
            photometric_system="JKC_Std",
            save_file=False,
        )
    except Exception as exc:
        print(f"GaiaXPy photometry request failed: {exc}")
        return None

    if result is None or len(result) == 0:
        print("GaiaXPy returned no photometry for this source.")
        return None

    try:
        if hasattr(result, "iloc"):
            row = result.iloc[0]
        elif isinstance(result, (list, tuple)):
            row = result[0]
        else:
            row = result
    except Exception:
        print("Unable to parse GaiaXPy photometry output.")
        return None

    bands = ['U', 'B', 'V', 'R', 'I']
    photometry = {}
    for band in bands:
        mag_key = f"JkcStd_mag_{band}"
        flux_key = f"JkcStd_flux_{band}"
        flux_err_key = f"JkcStd_flux_error_{band}"

        mag_value = row.get(mag_key) if hasattr(row, "get") else row[mag_key]
        flux_value = row.get(flux_key) if hasattr(row, "get") else row[flux_key]
        flux_err_value = row.get(flux_err_key) if hasattr(row, "get") else row[flux_err_key]

        mag_float = _as_float(mag_value)
        flux_float = _as_float(flux_value)
        flux_err_float = _as_float(flux_err_value)

        photometry[mag_key] = mag_float

        if flux_float is not None and flux_err_float is not None and flux_err_float > 0:
            snr = flux_float / flux_err_float
            if snr > 0:
                mag_err = 2.5 / np.log(10) / snr
                photometry[f"{mag_key}_error"] = mag_err
            else:
                photometry[f"{mag_key}_error"] = None
        else:
            photometry[f"{mag_key}_error"] = None

    if all(photometry[key] is None for key in photometry if key.startswith("JkcStd_mag_")):
        print("GaiaXPy photometry columns were unavailable in the response.")
        return None

    return photometry

if __name__ == '__main__':
    print("--- GAIA Third Light PSF Visualizer ---")
    print("Please enter the following parameters:")

    try:
        # --- 1. Input Parameters from User ---
        ra_input = input("Target RA (degrees or hh:mm:ss): ")
        dec_input = input("Target Dec (degrees or ±dd:mm:ss): ")
        ra_deg = _parse_ra_input(ra_input)
        dec_deg = _parse_dec_input(dec_input)
        image_scale = float(input("Image Scale (arcseconds/pixel): "))
        fwhm_arcsec = float(input("Seeing FWHM (arcseconds): "))
        aperture_radius_pixels = float(input("Aperture Radius (pixels): "))
        annulus_inner_radius_pixels = float(input("Annulus Inner Radius (pixels): "))
        annulus_width_pixels = float(input("Annulus Width (pixels): "))
        
        custom_fov_str = input("Enter custom Field of View (arcseconds) or press Enter for calculated default: ")
        field_of_view_arcsec = float(custom_fov_str) if custom_fov_str else None 
        
        # --- 2. Calculate default field_of_view_arcsec if not provided ---
        if field_of_view_arcsec is None:
            default_fov_pixels = 4 * (annulus_inner_radius_pixels + annulus_width_pixels)
            field_of_view_arcsec = default_fov_pixels * image_scale
            print(f"Using default field of view: {field_of_view_arcsec:.2f} arcsec "
                  f"({default_fov_pixels:.1f} pixels across).")

        # --- 3. Query GAIA DR3 Catalog ---
        target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
        # Change 3: Increase search radius to cover corners
        search_radius_arcsec = (field_of_view_arcsec / 2.0) * np.sqrt(2)
        search_radius = search_radius_arcsec * u.arcsec

        print(f"Querying GAIA DR3 around {target_coord.to_string('hmsdms')} "
              f"with a radius of {search_radius.to(u.arcmin):.2f}")

        if DEBUG_PRINT_GAIA_ROW:
            vizier_columns = ['**']
        else:
            vizier_columns = ['**']

        vizier_query = Vizier(
            columns=vizier_columns,
            row_limit=-1
        )
        gaia_catalog_id = 'I/355/gaiadr3'

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result_tables = vizier_query.query_region(target_coord, radius=search_radius, catalog=gaia_catalog_id)
        except Exception as e:
            print(f"Failed to query Vizier: {e}")
            exit()

        if not result_tables:
            print("No GAIA DR3 sources found for the provided coordinates. Cannot determine target magnitude.")
            exit()
        else:
            gaia_stars = result_tables[0]
            if len(gaia_stars) == 0:
                print("GAIA DR3 query returned zero sources. Cannot determine target magnitude.")
                exit()
            print(f"Found {len(gaia_stars)} GAIA DR3 sources.")
            if DEBUG_PRINT_GAIA_ROW:
                print("\n=== DEBUG: Raw VizieR result table ===")
                try:
                    gaia_stars.pprint(max_lines=-1, max_width=500)
                except Exception:
                    print(gaia_stars)
                print("=== DEBUG: End VizieR result table ===\n")

        # --- 4. Identify the target in the GAIA list, or add it if not present ---
        closest_gaia_idx = -1
        x_pixel_offset_raw = np.array([])
        y_pixel_offset_raw = np.array([])

        gaia_coords = SkyCoord(ra=gaia_stars['RA_ICRS'], dec=gaia_stars['DE_ICRS'], unit='deg', frame='icrs')
        separations = target_coord.separation(gaia_coords)

        delta_ra_arcsec = (gaia_coords.ra - target_coord.ra).to(u.arcsec).value
        delta_dec_arcsec = (gaia_coords.dec - target_coord.dec).to(u.arcsec).value
        x_pixel_offset_raw = (delta_ra_arcsec * np.cos(target_coord.dec.to(u.rad).value)) / image_scale
        y_pixel_offset_raw = delta_dec_arcsec / image_scale

        closest_gaia_idx_candidate = np.argmin(separations)
        if separations[closest_gaia_idx_candidate] < (image_scale * u.arcsec * 2.0):
            target_gmag = gaia_stars['Gmag'][closest_gaia_idx_candidate]
            closest_gaia_idx = closest_gaia_idx_candidate
            # Change 2: Display Gmag with .3f format
            print(f"Target found in GAIA DR3 (Gmag={target_gmag:.3f}).")

            target_row = gaia_stars[closest_gaia_idx]

            if DEBUG_PRINT_GAIA_ROW:
                print("\n=== DEBUG: GAIA target row (full contents) ===")
                print(target_row)
                print("=== DEBUG: End GAIA target row ===\n")

            gaia_ra = _column_value(target_row, 'RA_ICRS')
            gaia_dec = _column_value(target_row, 'DE_ICRS')
            gaia_bp = _column_value(target_row, 'BPmag', 'phot_bp_mean_mag')
            gaia_rp = _column_value(target_row, 'RPmag', 'phot_rp_mean_mag')
            gaia_bp_rp = _column_value(target_row, 'BP-RP', 'BP_RP', 'bp_rp')
            gaia_ebp_rp = _column_value(target_row, 'E(BP-RP)', 'E_BP-RP', 'ebpminrp_gspphot')

            dist_gspphot_pc = _column_value(target_row, 'Dist', 'distance_gspphot')
            parallax_mas = _column_value(target_row, 'Plx', 'parallax')
            parallax_distance_pc = None
            if parallax_mas is not None and parallax_mas > 0:
                parallax_distance_pc = 1000.0 / parallax_mas

            gaia_dist_pc = None
            distance_source = None
            if dist_gspphot_pc is not None:
                gaia_dist_pc = dist_gspphot_pc
                distance_source = "distance_gspphot"
            elif parallax_distance_pc is not None:
                gaia_dist_pc = parallax_distance_pc
                distance_source = "1/parallax"
            gaia_source_id = _column_text(target_row, 'Source', 'source_id')
            gaia_teff = _column_value(target_row, 'Teff_gspphot', 'Teff')

            adjusted_bp_rp = None
            if gaia_bp_rp is not None and gaia_ebp_rp is not None:
                adjusted_bp_rp = gaia_bp_rp - gaia_ebp_rp

            absolute_mag = None
            if gaia_dist_pc is not None and gaia_dist_pc > 0:
                absolute_mag = target_gmag - 5 * np.log10(gaia_dist_pc / 10.0)

            classification = _classify_star(absolute_mag, adjusted_bp_rp)
            gaiaxpy_photometry = _fetch_gaiaxpy_photometry(gaia_source_id)

            print("\n--- Target Photometric Properties ---")
            if gaia_source_id is not None:
                print(f"GAIA Source ID: {gaia_source_id}")
            else:
                print("GAIA Source ID: unavailable")
            if gaia_ra is not None and gaia_dec is not None:
                coord_icrs = SkyCoord(ra=gaia_ra * u.deg, dec=gaia_dec * u.deg, frame='icrs')
                print(f"GAIA coordinate: RA={gaia_ra:.6f} deg, Dec={gaia_dec:.6f} deg ({coord_icrs.to_string('hmsdms')})")
            else:
                print("GAIA coordinate: unavailable")

            print("Magnitudes:")
            print(f"  Gmag = {target_gmag:.3f}")
            print(f"  BPmag = {gaia_bp:.3f}" if gaia_bp is not None else "  BPmag = unavailable")
            print(f"  RPmag = {gaia_rp:.3f}" if gaia_rp is not None else "  RPmag = unavailable")
            print(f"  BP-RP colour = {gaia_bp_rp:.3f}" if gaia_bp_rp is not None else "  BP-RP colour = unavailable")

            if gaia_ebp_rp is not None:
                print(f"Colour excess E(BP-RP) = {gaia_ebp_rp:.3f}")
            else:
                print("Colour excess E(BP-RP) = unavailable")

            if adjusted_bp_rp is not None:
                print(f"Adjusted BP-RP colour = {adjusted_bp_rp:.3f}")
            else:
                print("Adjusted BP-RP colour = unavailable")

            if gaia_teff is not None:
                print(f"Effective temperature (teff_gspphot) = {gaia_teff:.0f} K")

            if gaia_dist_pc is not None:
                if distance_source == "distance_gspphot":
                    print(f"Photometric distance (distance_gspphot) = {gaia_dist_pc:.1f} pc ({gaia_dist_pc * 3.26156:.1f} ly)")
                elif distance_source == "1/parallax":
                    print(f"Distance (1/parallax) = {gaia_dist_pc:.1f} pc ({gaia_dist_pc * 3.26156:.1f} ly)")
                else:
                    print(f"Distance = {gaia_dist_pc:.1f} pc ({gaia_dist_pc * 3.26156:.1f} ly)")
            else:
                print("Photometric distance = unavailable")

            if absolute_mag is not None:
                print(f"Absolute magnitude (G) = {absolute_mag:.3f}")
            else:
                print("Absolute magnitude (G) = unavailable")

            if classification is not None:
                label = classification["label"]
                bp_color = classification["bp_rp"]
                m_g_value = classification["M_G"]
                derived_teff = classification["Teff"]
                display_teff = gaia_teff if gaia_teff is not None else derived_teff
                details = f"Stellar type estimate: {label} (Bp-Rp≈{bp_color:.3f}, M_G≈{m_g_value:.2f}"
                if display_teff is not None:
                    if gaia_teff is not None:
                        details += f", Teff≈{display_teff:.0f} K (teff_gspphot)"
                    else:
                        details += f", Teff≈{display_teff:.0f} K (table)"
                details += ")"
                print(details)
            else:
                print("Stellar type estimate: unavailable (insufficient data)")

            if gaiaxpy_photometry:
                print("\n--- GaiaXPy Synthetic Photometry (JKC) ---")
                for band in ['U', 'B', 'V', 'R', 'I']:
                    mag_key = f"JkcStd_mag_{band}"
                    err_key = f"{mag_key}_error"
                    mag_value = gaiaxpy_photometry.get(mag_key)
                    err_value = gaiaxpy_photometry.get(err_key)
                    if mag_value is not None:
                        if err_value is not None:
                            print(f"  {band}-band magnitude = {mag_value:.3f} ± {err_value:.3f}")
                        else:
                            print(f"  {band}-band magnitude = {mag_value:.3f}")
                    else:
                        print(f"  {band}-band magnitude = unavailable")
            else:
                print("\nNo GaiaXPy synthetic photometry available for this source (spectra unavailable or query failed).")
            print("-------------------------------\n")
        else:
            print("Target not found in GAIA DR3 near the provided coordinates. Please verify the inputs and try again.")
            exit()

        x_pixel_offset = x_pixel_offset_raw
        y_pixel_offset = y_pixel_offset_raw
        star_magnitudes = np.array(gaia_stars['Gmag'])
        star_ids = np.array(gaia_stars['Source'])

        # --- 5. Image Grid Setup for PSF Simulation ---
        half_fov_pixels = (field_of_view_arcsec / 2.0) / image_scale
        image_width_pixels = int(np.ceil(2 * half_fov_pixels))
        if image_width_pixels % 2 == 0: image_width_pixels += 1

        base_x = np.arange(image_width_pixels) - (image_width_pixels - 1) / 2.0
        base_y = np.arange(image_width_pixels) - (image_width_pixels - 1) / 2.0

        oversampled_width = image_width_pixels * OVERSAMPLING_FACTOR
        oversampled_x = np.linspace(base_x.min(), base_x.max(), oversampled_width)
        oversampled_y = np.linspace(base_y.min(), base_y.max(), oversampled_width)
        oversampled_X, oversampled_Y = np.meshgrid(oversampled_x, oversampled_y)

        fwhm_pixels = fwhm_arcsec / image_scale
        stddev_pixels = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))

        reference_gmag = 10.0
        reference_peak_flux = 1000.0
        normalization_factor = reference_peak_flux / (10**(-0.4 * reference_gmag))

        # --- 6. Generate Synthetic Images ---
        def create_image_from_indices(indices):
            synthetic_image_oversampled = np.zeros((oversampled_width, oversampled_width), dtype=np.float32)
            if len(indices) == 0:
                return block_reduce(synthetic_image_oversampled, OVERSAMPLING_FACTOR, func=np.sum)

            for i in indices:
                star_mag = star_magnitudes[i]
                amplitude = normalization_factor * (10**(-0.4 * star_mag))
                star_psf = Gaussian2D(
                    amplitude=amplitude, x_mean=x_pixel_offset[i], y_mean=y_pixel_offset[i],
                    x_stddev=stddev_pixels, y_stddev=stddev_pixels, theta=0
                )
                synthetic_image_oversampled += star_psf(oversampled_X, oversampled_Y)
                
            return block_reduce(synthetic_image_oversampled, OVERSAMPLING_FACTOR, func=np.sum)

        all_indices = np.arange(len(star_magnitudes))
        third_light_indices = np.delete(all_indices, closest_gaia_idx)
        
        synthetic_image_all = create_image_from_indices(all_indices)
        synthetic_image_no_target = create_image_from_indices(third_light_indices)
        target_only_image = create_image_from_indices([closest_gaia_idx])

        # --- 7. Flux Ratio Calculations ---
        # The center of the numpy array for photometry
        phot_center_xy = ((image_width_pixels - 1) / 2.0, (image_width_pixels - 1) / 2.0)
        aperture = CircularAperture(phot_center_xy, r=aperture_radius_pixels)
        
        # Calculate target's flux in the aperture
        target_phot_table = aperture_photometry(target_only_image, aperture)
        target_flux_in_aperture = target_phot_table['aperture_sum'][0]
        
        # Calculate third light flux in the aperture
        third_light_phot_table = aperture_photometry(synthetic_image_no_target, aperture)
        third_light_flux_in_aperture = third_light_phot_table['aperture_sum'][0]
        
        # Calculate total flux of just the target star over the whole image
        total_target_flux = np.sum(target_only_image)
        
        # Calculate fluxes in the sky annulus
        annulus_outer_radius_pixels = annulus_inner_radius_pixels + annulus_width_pixels
        annulus = CircularAnnulus(phot_center_xy, r_in=annulus_inner_radius_pixels, r_out=annulus_outer_radius_pixels)
        
        # Calculate total flux in the annulus (all sources including target)
        all_annulus_phot_table = aperture_photometry(synthetic_image_all, annulus)
        total_flux_in_annulus = all_annulus_phot_table['aperture_sum'][0]
        
        # Calculate number of pixels in annulus and aperture
        annulus_area = annulus.area
        aperture_area = aperture.area
        
        # Calculate the third light contamination as done in aperture photometry:
        # flux per pixel in annulus * aperture area / target flux in aperture
        flux_per_pixel_in_annulus = total_flux_in_annulus / annulus_area
        background_estimate_for_aperture = flux_per_pixel_in_annulus * aperture_area
        
        print("\n--- Aperture Analysis ---")
        # Change 4: Report ratio of target flux in aperture to total target flux
        if total_target_flux > 0:
            target_flux_ratio = (target_flux_in_aperture / total_target_flux) * 100
            print(f"Target flux in aperture is {target_flux_ratio:.2f}% of the target's total simulated flux.")
        else:
            print("Target has no flux; cannot calculate flux ratio.")
            
        # Change 5: Report ratio of third light flux to target flux
        if target_flux_in_aperture > 0:
            third_light_ratio = (third_light_flux_in_aperture / target_flux_in_aperture) * 100
            print(f"Third light flux in aperture is {third_light_ratio:.2f}% of the target's flux in aperture (contamination).")
        else:
            print("Target has no flux in aperture; cannot calculate contamination ratio.")
        
        print("\n--- Sky Annulus Analysis ---")
        # Report third light contamination as calculated in aperture photometry
        if target_flux_in_aperture > 0:
            sky_contamination_ratio = (background_estimate_for_aperture / target_flux_in_aperture) * 100
            print(f"Sky background contamination (all sources in annulus scaled to aperture): {sky_contamination_ratio:.2f}% of target flux.")
            print(f"  (Total annulus flux: {total_flux_in_annulus:.2f}, Flux per pixel: {flux_per_pixel_in_annulus:.4f})")
            print(f"  (Aperture area: {aperture_area:.2f} pixels, Annulus area: {annulus_area:.2f} pixels)")
        else:
            print("Target has no flux in aperture; cannot calculate sky contamination ratio.")
        print("-------------------------\n")


        # --- 8. Plotting ---
        # Change 1: Display RA/Dec with .6f format
        fig_title = f"Target Field Simulation (RA={ra_deg:.6f}, Dec={dec_deg:.6f})"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharex=True, sharey=True)
        fig.suptitle(fig_title, fontsize=16)

        vmax_all = np.percentile(synthetic_image_all, 99.9) if synthetic_image_all.max() > 0 else 1
        vmin_all = np.percentile(synthetic_image_all, 1) if synthetic_image_all.max() > 0 else 0

        def plot_field_simulated(ax, image_data, title, is_third_light_plot=False):
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("Relative X (pixels)")
            ax.set_ylabel("Relative Y (pixels)")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            extent = [base_x.min() - 0.5, base_x.max() + 0.5, base_y.min() - 0.5, base_y.max() + 0.5]
            im = ax.imshow(image_data, cmap='gray_r', origin='lower', extent=extent,
                           vmin=vmin_all, vmax=vmax_all)
            fig.colorbar(im, ax=ax, shrink=0.8, label='Simulated Pixel Flux (ADU)')

            plot_center_xy = (0, 0) # For plotting, the center is (0,0)
            aperture_plot = CircularAperture(plot_center_xy, r=aperture_radius_pixels)
            annulus_outer_radius_pixels = annulus_inner_radius_pixels + annulus_width_pixels
            annulus_plot = CircularAnnulus(plot_center_xy, r_in=annulus_inner_radius_pixels, r_out=annulus_outer_radius_pixels)

            aperture_plot.plot(ax=ax, color='cyan', lw=1.5, ls='-', label=f'Aperture (r={aperture_radius_pixels:.1f} pix)')
            annulus_plot.plot(ax=ax, color='orange', lw=1.5, ls='-', label=f'Annulus (r_in={annulus_inner_radius_pixels:.1f} pix, r_out={annulus_outer_radius_pixels:.1f} pix)')
            
            ax.set_title(title + " (Target Removed)" if is_third_light_plot else title)
            ax.legend()

        plot_field_simulated(ax1, synthetic_image_all, "Simulated Field")
        plot_field_simulated(ax2, synthetic_image_no_target, "Simulated Field", is_third_light_plot=True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except ValueError:
        print("Invalid input. Please ensure all numerical inputs are valid numbers and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
