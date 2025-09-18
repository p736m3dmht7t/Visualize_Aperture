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

if __name__ == '__main__':
    print("--- GAIA Third Light PSF Visualizer ---")
    print("Please enter the following parameters:")

    try:
        # --- 1. Input Parameters from User ---
        ra_deg = float(input("Target RA (degrees): "))
        dec_deg = float(input("Target Dec (degrees): "))
        image_scale = float(input("Image Scale (arcseconds/pixel): "))
        fwhm_arcsec = float(input("Seeing FWHM (arcseconds): "))
        aperture_radius_pixels = float(input("Aperture Radius (pixels): "))
        annulus_inner_radius_pixels = float(input("Annulus Inner Radius (pixels): "))
        annulus_width_pixels = float(input("Annulus Width (pixels): "))
        
        custom_fov_str = input("Enter custom Field of View (arcseconds) or press Enter for calculated default: ")
        field_of_view_arcsec = float(custom_fov_str) if custom_fov_str else None 
        
        oversampling_factor = 5 
        target_gmag = 12.0      


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

        vizier_query = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 'Source'], row_limit=-1)
        gaia_catalog_id = 'I/355/gaiadr3'

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result_tables = vizier_query.query_region(target_coord, radius=search_radius, catalog=gaia_catalog_id)
        except Exception as e:
            print(f"Failed to query Vizier: {e}")
            exit()

        if not result_tables:
            print("No stars found in GAIA DR3 catalog for this region. Using only a synthetic target star.")
            gaia_stars = None
        else:
            gaia_stars = result_tables[0]
            print(f"Found {len(gaia_stars)} GAIA DR3 sources.")

        # --- 4. Identify the target in the GAIA list, or add it if not present ---
        target_is_in_gaia = False
        closest_gaia_idx = -1
        x_pixel_offset_raw = np.array([])
        y_pixel_offset_raw = np.array([])

        if gaia_stars is not None:
            gaia_coords = SkyCoord(ra=gaia_stars['RA_ICRS'], dec=gaia_stars['DE_ICRS'], unit='deg', frame='icrs')
            separations = target_coord.separation(gaia_coords)

            delta_ra_arcsec = (gaia_coords.ra - target_coord.ra).to(u.arcsec).value
            delta_dec_arcsec = (gaia_coords.dec - target_coord.dec).to(u.arcsec).value
            x_pixel_offset_raw = (delta_ra_arcsec * np.cos(target_coord.dec.to(u.rad).value)) / image_scale
            y_pixel_offset_raw = delta_dec_arcsec / image_scale

            closest_gaia_idx_candidate = np.argmin(separations)
            if separations[closest_gaia_idx_candidate] < (image_scale * u.arcsec / 2.0):
                target_gmag = gaia_stars['Gmag'][closest_gaia_idx_candidate]
                target_is_in_gaia = True
                closest_gaia_idx = closest_gaia_idx_candidate
                # Change 2: Display Gmag with .3f format
                print(f"Target found in GAIA DR3 (Gmag={target_gmag:.3f}).")
            else:
                print(f"Target not explicitly found in GAIA DR3 within 0.5 pixel. Using default target_gmag={target_gmag:.3f}.")
        else:
            print(f"Using default target_gmag={target_gmag:.3f}.")

        if not target_is_in_gaia:
            x_pixel_offset = np.insert(x_pixel_offset_raw, 0, 0.0)
            y_pixel_offset = np.insert(y_pixel_offset_raw, 0, 0.0)
            gaia_mags = np.array(gaia_stars['Gmag']) if gaia_stars is not None else np.array([])
            gaia_ids = np.array(gaia_stars['Source']) if gaia_stars is not None else np.array([])
            star_magnitudes = np.insert(gaia_mags, 0, target_gmag)
            star_ids = np.insert(gaia_ids, 0, -1)
            closest_gaia_idx = 0 
        else:
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

        oversampled_width = image_width_pixels * oversampling_factor
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
                return block_reduce(synthetic_image_oversampled, oversampling_factor, func=np.sum)

            for i in indices:
                star_mag = star_magnitudes[i]
                amplitude = normalization_factor * (10**(-0.4 * star_mag))
                star_psf = Gaussian2D(
                    amplitude=amplitude, x_mean=x_pixel_offset[i], y_mean=y_pixel_offset[i],
                    x_stddev=stddev_pixels, y_stddev=stddev_pixels, theta=0
                )
                synthetic_image_oversampled += star_psf(oversampled_X, oversampled_Y)
                
            return block_reduce(synthetic_image_oversampled, oversampling_factor, func=np.sum)

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
