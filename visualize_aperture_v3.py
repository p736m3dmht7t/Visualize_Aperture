# visualize_aperture_v3.py
# Based on visualize_aperture_v2.py by John Phillips
# Extended for Multi-Target Processing and PDF Output

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
import matplotlib.backends.backend_pdf

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
                # It's optional, so just proceed without it if it fails
                pass
    return cache_file


def _parse_mamajek_table(table_path: Path):
    header_map = {}
    rows = []
    if not table_path.exists():
        return []
        
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
    except FileNotFoundError:
        pass 
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
    stripped = str(value).strip()
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
    stripped = str(value).strip()
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
        return None

    try:
        source_id_int = int(str(source_id_value).strip())
    except Exception:
        return None

    try:
        result = generate(
            source_id=[source_id_int],
            photometric_system="JKC_Std",
            save_file=False,
        )
    except Exception:
        return None

    if result is None or len(result) == 0:
        return None

    try:
        if hasattr(result, "iloc"):
            row = result.iloc[0]
        elif isinstance(result, (list, tuple)):
            row = result[0]
        else:
            row = result
    except Exception:
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
        return None

    return photometry


def simulate_star_field(ra_deg, dec_deg, image_scale, fwhm_arcsec, aperture_radius_pixels, 
                       annulus_inner_radius_pixels, annulus_width_pixels, field_of_view_arcsec=None):
    """
    Simulate the star field and return images and metrics.
    """
    
    if field_of_view_arcsec is None:
        default_fov_pixels = 4 * (annulus_inner_radius_pixels + annulus_width_pixels)
        field_of_view_arcsec = default_fov_pixels * image_scale

    # --- Query GAIA DR3 Catalog ---
    target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    search_radius_arcsec = (field_of_view_arcsec / 2.0) * np.sqrt(2)
    search_radius = search_radius_arcsec * u.arcsec

    # print(f"Querying GAIA DR3 around {target_coord.to_string('hmsdms')}...")

    vizier_query = Vizier(columns=['**'], row_limit=-1)
    gaia_catalog_id = 'I/355/gaiadr3'

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result_tables = vizier_query.query_region(target_coord, radius=search_radius, catalog=gaia_catalog_id)
    except Exception as e:
        return None, f"Failed to query Vizier: {e}"

    if not result_tables or len(result_tables) == 0:
        return None, "No GAIA DR3 sources found."

    gaia_stars = result_tables[0]
    if len(gaia_stars) == 0:
        return None, "GAIA DR3 query returned zero sources."

    # --- Identify Target ---
    gaia_coords = SkyCoord(ra=gaia_stars['RA_ICRS'], dec=gaia_stars['DE_ICRS'], unit='deg', frame='icrs')
    separations = target_coord.separation(gaia_coords)
    
    delta_ra_arcsec = (gaia_coords.ra - target_coord.ra).to(u.arcsec).value
    delta_dec_arcsec = (gaia_coords.dec - target_coord.dec).to(u.arcsec).value
    
    # Projection to pixels
    x_pixel_offset = (delta_ra_arcsec * np.cos(target_coord.dec.to(u.rad).value)) / image_scale
    y_pixel_offset = delta_dec_arcsec / image_scale

    closest_gaia_idx = np.argmin(separations)
    if separations[closest_gaia_idx] >= (image_scale * u.arcsec * 2.0):
        return None, "Target not found in GAIA DR3 within 2 pixels tolerance."

    target_gmag = gaia_stars['Gmag'][closest_gaia_idx]
    target_row = gaia_stars[closest_gaia_idx]
    
    # Extract detailed target information
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
    
    star_magnitudes = np.array(gaia_stars['Gmag'])
    
    # --- Image Grid Setup ---
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

    # --- Generate Synthetic Images ---
    def create_image_from_indices(indices):
        synthetic_image_oversampled = np.zeros((oversampled_width, oversampled_width), dtype=np.float32)
        if len(indices) == 0:
            return block_reduce(synthetic_image_oversampled, OVERSAMPLING_FACTOR, func=np.sum)

        for i in indices:
            star_mag = star_magnitudes[i]
            if np.isnan(star_mag): continue # Skip stars with no mag
            
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

    # --- Flux Ratio Calculations ---
    phot_center_xy = ((image_width_pixels - 1) / 2.0, (image_width_pixels - 1) / 2.0)
    aperture = CircularAperture(phot_center_xy, r=aperture_radius_pixels)
    
    target_phot_table = aperture_photometry(target_only_image, aperture)
    target_flux_in_aperture = target_phot_table['aperture_sum'][0]
    
    third_light_phot_table = aperture_photometry(synthetic_image_no_target, aperture)
    third_light_flux_in_aperture = third_light_phot_table['aperture_sum'][0]
    
    annulus_outer_radius_pixels = annulus_inner_radius_pixels + annulus_width_pixels
    annulus = CircularAnnulus(phot_center_xy, r_in=annulus_inner_radius_pixels, r_out=annulus_outer_radius_pixels)
    
    all_annulus_phot_table = aperture_photometry(synthetic_image_all, annulus)
    total_flux_in_annulus = all_annulus_phot_table['aperture_sum'][0]
    
    annulus_area = annulus.area
    aperture_area = aperture.area
    
    flux_per_pixel_in_annulus = total_flux_in_annulus / annulus_area
    background_estimate_for_aperture = flux_per_pixel_in_annulus * aperture_area

    metrics = {
        "target_gmag": target_gmag,
        "target_flux_in_aperture": target_flux_in_aperture,
        "third_light_flux_in_aperture": third_light_flux_in_aperture,
        "third_light_ratio": (third_light_flux_in_aperture / target_flux_in_aperture) * 100 if target_flux_in_aperture > 0 else 0,
        "sky_contamination_ratio": (background_estimate_for_aperture / target_flux_in_aperture) * 100 if target_flux_in_aperture > 0 else 0,
        "aperture_radius": aperture_radius_pixels,
        "annulus_inner": annulus_inner_radius_pixels,
        "annulus_outer": annulus_outer_radius_pixels,
        "extent": [base_x.min() - 0.5, base_x.max() + 0.5, base_y.min() - 0.5, base_y.max() + 0.5]
    }
    
    target_info = {
        "gaia_source_id": gaia_source_id,
        "gaia_ra": gaia_ra,
        "gaia_dec": gaia_dec,
        "gaia_bp": gaia_bp,
        "gaia_rp": gaia_rp,
        "gaia_bp_rp": gaia_bp_rp,
        "gaia_ebp_rp": gaia_ebp_rp,
        "adjusted_bp_rp": adjusted_bp_rp,
        "gaia_teff": gaia_teff,
        "gaia_dist_pc": gaia_dist_pc,
        "distance_source": distance_source,
        "absolute_mag": absolute_mag,
        "classification": classification,
        "gaiaxpy_photometry": gaiaxpy_photometry
    }

    return {
        "image_all": synthetic_image_all,
        "image_no_target": synthetic_image_no_target,
        "metrics": metrics,
        "target_info": target_info,
        "base_x": base_x,
        "base_y": base_y
    }, None


def format_target_info_row(target_num, ra_str, dec_str, file_mag, target_info, metrics):
    """
    Format target information as a single table row.
    Column widths must exactly match the header.
    """
    info = target_info
    m = metrics
    
    # Format values - widths: 3, 15, 15, 7, 15, 6, 6, 6, 6, 8, 9, 6, 8, 6, 8, 7, 6
    source_id = str(info.get("gaia_source_id", "N/A"))[:15] if info.get("gaia_source_id") else "N/A"
    
    gmag = f"{m.get('target_gmag', 0):.3f}" if m.get('target_gmag') is not None else "  N/A"
    bpmag = f"{info.get('gaia_bp', 0):.3f}" if info.get('gaia_bp') is not None else "  N/A"
    rpmag = f"{info.get('gaia_rp', 0):.3f}" if info.get('gaia_rp') is not None else "  N/A"
    bp_rp = f"{info.get('gaia_bp_rp', 0):.3f}" if info.get('gaia_bp_rp') is not None else "  N/A"
    ebp_rp = f"{info.get('gaia_ebp_rp', 0):.3f}" if info.get('gaia_ebp_rp') is not None else "  N/A"
    adj_bp_rp = f"{info.get('adjusted_bp_rp', 0):.3f}" if info.get('adjusted_bp_rp') is not None else "  N/A"
    
    teff = f"{info.get('gaia_teff', 0):.0f}" if info.get('gaia_teff') is not None else "  N/A"
    
    dist_pc = info.get('gaia_dist_pc')
    if dist_pc is not None:
        dist_str = f"{dist_pc:8.1f}"
    else:
        dist_str = "    N/A"
    
    abs_mag = f"{info.get('absolute_mag', 0):.3f}" if info.get('absolute_mag') is not None else "  N/A"
    
    classification = info.get('classification')
    if classification:
        spec_type = classification.get('label', 'N/A')
        teff_table = classification.get('Teff')
        if teff_table is not None and info.get('gaia_teff') is None:
            teff_display = f"{teff_table:.0f}"
        else:
            teff_display = teff
    else:
        spec_type = "N/A"
        teff_display = teff
    
    if file_mag is not None:
        try:
            file_mag_str = f"{file_mag:6.3f}"
        except (ValueError, TypeError):
            file_mag_str = "  N/A"
    else:
        file_mag_str = "  N/A"
    
    third_light = f"{m.get('third_light_ratio', 0):6.2f}" if m.get('third_light_ratio') is not None else "  N/A"
    sky_contam = f"{m.get('sky_contamination_ratio', 0):6.2f}" if m.get('sky_contamination_ratio') is not None else "  N/A"
    
    # Format row - exact widths to match header: 3|15|15|7|15|6|6|6|6|8|9|6|8|6|8|7|6
    row = (
        f"{target_num:3d} | {ra_str:15s} | {dec_str:15s} | {file_mag_str:>7s} | "
        f"{source_id:>15s} | {gmag:>6s} | {bpmag:>6s} | {rpmag:>6s} | "
        f"{bp_rp:>6s} | {ebp_rp:>8s} | {adj_bp_rp:>9s} | {teff_display:>6s} | "
        f"{dist_str:>8s} | {abs_mag:>6s} | {spec_type:>8s} | "
        f"{third_light:>7s}% | {sky_contam:>6s}%"
    )
    return row


def get_target_info_header():
    """Get the header row for the target information table."""
    # Column widths: 3|15|15|7|15|6|6|6|6|8|9|6|8|6|8|7|6
    header = (
        " #  | RA              | Dec             | FileMag | "
        "Source ID       |  Gmag  |  BPmag |  RPmag | "
        " BP-RP | E(BP-RP) | Adj BP-RP |   Teff | Distance |   M_G  | SpecType | "
        " 3rdLt % |   Sky%  "
    )
    separator = "-" * len(header)
    return header, separator


def print_target_info_header():
    """Print the header row for the target information table."""
    header, separator = get_target_info_header()
    print(header)
    print(separator)


def parse_radec_file(filepath):
    targets = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
                
            ra_str = parts[0]
            dec_str = parts[1]
            
            try:
                ra = _parse_ra_input(ra_str)
                dec = _parse_dec_input(dec_str)
            except ValueError:
                continue
                
            ref_star = 0
            if len(parts) > 2 and parts[2]:
                try:
                    ref_star = int(parts[2])
                except ValueError:
                    pass
            
            mag = None
            if len(parts) > 4 and parts[4]:
                try:
                    mag = float(parts[4])
                    if mag > 99: mag = None
                except ValueError:
                    pass
            
            targets.append({
                "ra": ra,
                "dec": dec,
                "ra_str": ra_str,
                "dec_str": dec_str,
                "ref_star": ref_star,
                "mag": mag
            })
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
        
    return targets


def process_targets_to_pdf(targets, output_filename, params):
    """
    Process list of targets and save to multi-page PDF.
    Layout: 4 targets per page (Rows), 2 plots per target (Cols).
    """
    print(f"\nStarting batch processing of {len(targets)} targets...")
    print(f"Output will be saved to: {output_filename}\n")
    
    # Check if file is already open or locked
    output_path = Path(output_filename)
    if output_path.exists():
        try:
            # Try to open the file in append mode to check if it's locked
            with open(output_path, 'ab') as test_file:
                pass
        except PermissionError:
            print(f"\nERROR: Cannot write to '{output_filename}'")
            print("The file may be open in another program (PDF viewer, etc.).")
            print("Please close the file and try again.\n")
            return
    
    # Prepare text output file
    text_output_file = output_path.with_suffix('.txt')
    table_rows = []
    
    # Print table header
    header, separator = get_target_info_header()
    print_target_info_header()
    table_rows.append(header)
    table_rows.append(separator)
    
    try:
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_filename)
    except PermissionError as e:
        print(f"\nERROR: Cannot create PDF file '{output_filename}'")
        print("The file may be open in another program (PDF viewer, etc.).")
        print("Please close the file and try again.")
        print(f"Details: {e}\n")
        return
    
    # Process in chunks of 4
    chunk_size = 4
    for i in range(0, len(targets), chunk_size):
        chunk = targets[i:i + chunk_size]
        
        # A4 size approx (8.27, 11.69), let's use a tall figure
        fig, axes = plt.subplots(4, 2, figsize=(10, 14), squeeze=False)
        fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # If chunk is smaller than 4, hide extra axes
        for j in range(len(chunk), 4):
            axes[j, 0].axis('off')
            axes[j, 1].axis('off')
            
        for idx, target in enumerate(chunk):
            target_num = i + idx + 1
            print(f"Processing target {target_num}/{len(targets)}: RA={target['ra_str']}, Dec={target['dec_str']}...", end=" ")
            
            res, error_msg = simulate_star_field(
                target['ra'], target['dec'], 
                params['image_scale'], params['fwhm'], 
                params['aperture'], params['annulus_inner'], params['annulus_width']
            )
            
            ax1 = axes[idx, 0]
            ax2 = axes[idx, 1]
            
            if error_msg:
                print(f"ERROR: {error_msg}")
                # Print error row
                error_row = (
                    f"{target_num:3d} | {target['ra_str']:15s} | {target['dec_str']:15s} | "
                    f"{target.get('mag', 'N/A'):>6s} | ERROR: {error_msg[:50]:50s}"
                )
                print(error_row)
                table_rows.append(error_row)
                ax1.text(0.5, 0.5, f"Error: {error_msg}", ha='center', va='center', wrap=True)
                ax1.axis('off')
                ax2.axis('off')
                continue
            
            # Print formatted information row
            row = format_target_info_row(
                target_num, target['ra_str'], target['dec_str'], 
                target.get('mag'), res.get('target_info', {}), res.get('metrics', {})
            )
            print("OK")
            print(row)
            table_rows.append(row)
            
            # Plotting logic
            metrics = res['metrics']
            image_all = res['image_all']
            image_no = res['image_no_target']
            extent = metrics['extent']
            
            # Determine vmin/vmax from 'all' image
            vmax = np.percentile(image_all, 99.9) if image_all.max() > 0 else 1
            vmin = np.percentile(image_all, 1) if image_all.max() > 0 else 0
            
            def plot_sub(ax, img, title, is_third_light=False):
                ax.imshow(img, cmap='gray_r', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
                
                plot_center_xy = (0, 0)
                ap_plot = CircularAperture(plot_center_xy, r=metrics['aperture_radius'])
                an_plot = CircularAnnulus(plot_center_xy, r_in=metrics['annulus_inner'], r_out=metrics['annulus_outer'])
                
                ap_plot.plot(ax=ax, color='cyan', lw=1, alpha=0.7)
                an_plot.plot(ax=ax, color='orange', lw=1, alpha=0.7)
                
                ax.set_title(title, fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=7)
                
                # Remove axis labels to save space, maybe just simple grids
                # ax.axis('off') # Keep axis for scale reference?
                ax.grid(True, linestyle=':', alpha=0.3)
            
            # Title info
            mag_info = f"Gmag={metrics['target_gmag']:.2f}"
            if target.get('mag'):
                mag_info += f" (File V={target['mag']:.2f})"
                
            title_main = f"Tgt {i+idx+1}: {target['ra_str']} {target['dec_str']}\n{mag_info}"
            
            plot_sub(ax1, image_all, title_main)
            
            # Add contamination text to the second plot or overlay
            contam_text = (
                f"3rd Light: {metrics['third_light_ratio']:.2f}%\n"
                f"Sky Contam: {metrics['sky_contamination_ratio']:.2f}%"
            )
            plot_sub(ax2, image_no, "Target Removed", is_third_light=True)
            
            # Add text box for stats
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax2.text(0.05, 0.95, contam_text, transform=ax2.transAxes, fontsize=7,
                    verticalalignment='top', bbox=props)

        try:
            pdf.savefig(fig)
        except PermissionError as e:
            plt.close(fig)
            pdf.close()
            print(f"\nERROR: Cannot save figure to PDF file '{output_filename}'")
            print("The file may have been opened by another program during processing.")
            print("Please close the file and try again.")
            print(f"Details: {e}\n")
            return
        plt.close(fig)
        
    try:
        pdf.close()
        print(f"\nDone! PDF saved to {output_filename}")
    except Exception as e:
        print(f"\nWARNING: Error closing PDF file: {e}")
        print(f"PDF may have been partially saved to {output_filename}")
    
    # Write table to text file
    try:
        with open(text_output_file, 'w', encoding='utf-8') as f:
            f.write(f"GAIA Third Light PSF Visualizer v3 - Target Analysis Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {params.get('input_file', 'N/A')}\n")
            f.write(f"Image Scale: {params['image_scale']:.3f} arcsec/pixel\n")
            f.write(f"Seeing FWHM: {params['fwhm']:.2f} arcsec\n")
            f.write(f"Aperture Radius: {params['aperture']:.1f} pixels\n")
            f.write(f"Annulus Inner: {params['annulus_inner']:.1f} pixels\n")
            f.write(f"Annulus Width: {params['annulus_width']:.1f} pixels\n")
            f.write(f"\nTotal targets processed: {len(targets)}\n")
            f.write("=" * 80 + "\n\n")
            for row in table_rows:
                f.write(row + "\n")
        print(f"Table saved to: {text_output_file}")
    except Exception as e:
        print(f"\nWARNING: Could not write text file: {e}")


def main():
    print("--- GAIA Third Light PSF Visualizer v3 ---")
    print("1. Process a single target (Manual Entry)")
    print("2. Process a list of targets (.radec file)")
    
    mode = input("Select mode (1 or 2): ").strip()
    
    if mode == '1':
        # Single target mode (similar to v2)
        try:
            ra_input = input("Target RA (degrees or hh:mm:ss): ")
            dec_input = input("Target Dec (degrees or ±dd:mm:ss): ")
            ra_deg = _parse_ra_input(ra_input)
            dec_deg = _parse_dec_input(dec_input)
            
            image_scale = float(input("Image Scale (arcseconds/pixel): "))
            fwhm_arcsec = float(input("Seeing FWHM (arcseconds): "))
            aperture_radius_pixels = float(input("Aperture Radius (pixels): "))
            annulus_inner_radius_pixels = float(input("Annulus Inner Radius (pixels): "))
            annulus_width_pixels = float(input("Annulus Width (pixels): "))
            
            res, error = simulate_star_field(
                ra_deg, dec_deg, image_scale, fwhm_arcsec, 
                aperture_radius_pixels, annulus_inner_radius_pixels, annulus_width_pixels
            )
            
            if error:
                print(f"Error: {error}")
                return

            # Print detailed target information
            target_info = res.get('target_info', {})
            metrics = res['metrics']
            
            print("\n--- Target Photometric Properties ---")
            if target_info.get('gaia_source_id'):
                print(f"GAIA Source ID: {target_info['gaia_source_id']}")
            else:
                print("GAIA Source ID: unavailable")
                
            if target_info.get('gaia_ra') is not None and target_info.get('gaia_dec') is not None:
                coord_icrs = SkyCoord(ra=target_info['gaia_ra'] * u.deg, dec=target_info['gaia_dec'] * u.deg, frame='icrs')
                print(f"GAIA coordinate: RA={target_info['gaia_ra']:.6f} deg, Dec={target_info['gaia_dec']:.6f} deg ({coord_icrs.to_string('hmsdms')})")
            else:
                print("GAIA coordinate: unavailable")

            print("Magnitudes:")
            print(f"  Gmag = {metrics['target_gmag']:.3f}")
            print(f"  BPmag = {target_info.get('gaia_bp', 0):.3f}" if target_info.get('gaia_bp') is not None else "  BPmag = unavailable")
            print(f"  RPmag = {target_info.get('gaia_rp', 0):.3f}" if target_info.get('gaia_rp') is not None else "  RPmag = unavailable")
            print(f"  BP-RP colour = {target_info.get('gaia_bp_rp', 0):.3f}" if target_info.get('gaia_bp_rp') is not None else "  BP-RP colour = unavailable")

            if target_info.get('gaia_ebp_rp') is not None:
                print(f"Colour excess E(BP-RP) = {target_info['gaia_ebp_rp']:.3f}")
            else:
                print("Colour excess E(BP-RP) = unavailable")

            if target_info.get('adjusted_bp_rp') is not None:
                print(f"Adjusted BP-RP colour = {target_info['adjusted_bp_rp']:.3f}")
            else:
                print("Adjusted BP-RP colour = unavailable")

            if target_info.get('gaia_teff') is not None:
                print(f"Effective temperature (teff_gspphot) = {target_info['gaia_teff']:.0f} K")

            if target_info.get('gaia_dist_pc') is not None:
                dist_source = target_info.get('distance_source', '')
                if dist_source == "distance_gspphot":
                    print(f"Photometric distance (distance_gspphot) = {target_info['gaia_dist_pc']:.1f} pc ({target_info['gaia_dist_pc'] * 3.26156:.1f} ly)")
                elif dist_source == "1/parallax":
                    print(f"Distance (1/parallax) = {target_info['gaia_dist_pc']:.1f} pc ({target_info['gaia_dist_pc'] * 3.26156:.1f} ly)")
                else:
                    print(f"Distance = {target_info['gaia_dist_pc']:.1f} pc ({target_info['gaia_dist_pc'] * 3.26156:.1f} ly)")
            else:
                print("Photometric distance = unavailable")

            if target_info.get('absolute_mag') is not None:
                print(f"Absolute magnitude (G) = {target_info['absolute_mag']:.3f}")
            else:
                print("Absolute magnitude (G) = unavailable")

            classification = target_info.get('classification')
            if classification:
                label = classification.get("label", "N/A")
                bp_color = classification.get("bp_rp", 0)
                m_g_value = classification.get("M_G", 0)
                derived_teff = classification.get("Teff")
                display_teff = target_info.get('gaia_teff') if target_info.get('gaia_teff') is not None else derived_teff
                details = f"Stellar type estimate: {label} (Bp-Rp≈{bp_color:.3f}, M_G≈{m_g_value:.2f}"
                if display_teff is not None:
                    if target_info.get('gaia_teff') is not None:
                        details += f", Teff≈{display_teff:.0f} K (teff_gspphot)"
                    else:
                        details += f", Teff≈{display_teff:.0f} K (table)"
                details += ")"
                print(details)
            else:
                print("Stellar type estimate: unavailable (insufficient data)")

            gaiaxpy_photometry = target_info.get('gaiaxpy_photometry')
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
            
            print("\n--- Aperture Analysis ---")
            print(f"Target Gmag: {metrics['target_gmag']:.3f}")
            print(f"Third Light Contamination: {metrics['third_light_ratio']:.2f}%")
            print(f"Sky Background Contamination: {metrics['sky_contamination_ratio']:.2f}%")
            print("-------------------------------\n")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            image_all = res['image_all']
            extent = metrics['extent']
            vmax = np.percentile(image_all, 99.9) if image_all.max() > 0 else 1
            vmin = np.percentile(image_all, 1) if image_all.max() > 0 else 0
            
            ax1.imshow(res['image_all'], cmap='gray_r', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
            ax1.set_title("Simulated Field")
            
            ax2.imshow(res['image_no_target'], cmap='gray_r', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
            ax2.set_title("Target Removed")
            
            # Plot apertures
            for ax in [ax1, ax2]:
                ap = CircularAperture((0,0), r=metrics['aperture_radius'])
                an = CircularAnnulus((0,0), r_in=metrics['annulus_inner'], r_out=metrics['annulus_outer'])
                ap.plot(ax=ax, color='cyan')
                an.plot(ax=ax, color='orange')
            
            plt.show()
            
        except ValueError as e:
            print(f"Invalid input: {e}")
            
    elif mode == '2':
        # File mode
        default_glob = list(Path("assets").glob("*.radec"))
        default_file = default_glob[0] if default_glob else "targets.radec"
        
        file_input = input(f"Enter path to .radec file [{default_file}]: ").strip()
        if not file_input:
            file_input = str(default_file)
            
        if not Path(file_input).exists():
            print(f"File not found: {file_input}")
            return
            
        targets = parse_radec_file(file_input)
        if not targets:
            print("No valid targets found in file.")
            return
            
        print(f"Found {len(targets)} targets.")
        
        # Global params
        try:
            print("\n--- Global Parameters ---")
            image_scale = float(input("Image Scale (arcseconds/pixel): "))
            fwhm_arcsec = float(input("Seeing FWHM (arcseconds): "))
            aperture_radius_pixels = float(input("Aperture Radius (pixels): "))
            annulus_inner_radius_pixels = float(input("Annulus Inner Radius (pixels): "))
            annulus_width_pixels = float(input("Annulus Width (pixels): "))
            
            params = {
                "image_scale": image_scale,
                "fwhm": fwhm_arcsec,
                "aperture": aperture_radius_pixels,
                "annulus_inner": annulus_inner_radius_pixels,
                "annulus_width": annulus_width_pixels,
                "input_file": file_input
            }
            
            output_file = Path(file_input).with_suffix('.pdf')
            process_targets_to_pdf(targets, str(output_file), params)
            
        except ValueError:
            print("Invalid numeric input.")
    else:
        print("Invalid mode selected.")

if __name__ == '__main__':
    main()
