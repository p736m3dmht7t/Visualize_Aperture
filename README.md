# Visualize_Aperture

visualize_aperture.py is a simple Python script that uses `matplotlib` to create a visual representation of GAIA DR3 detected light sources (stars) that are in the vicinity of, and may contaminate, the aperture photometry of a planned image.

This can be useful for pre-planning appropriate comparison stars that are free of contamination.  It can also be useful to determine how much light may be included inside the aperture with a target of interest.

A third use is to identify appropriate sky annulus inner radius and width to avoid including bright objects inside the sky annulus.

visualize_aperture.py was written while studying for the AAVSO (http://aavso.org) CHOICE course "Observational Best Practices CHOICE course (2025)".

## Features

*   **Customizable Aperture:** Easily adjust the radius of the central aperture.
*   **Customizable Sky Annulus:** Easily adjust the inner radius and width of the sky annulus used to find the background level of the observation.
*   **Clear Visualization:** Generates a clean 2D plot of the aperture, sky annulus, target star, and other stars that may contaminate the aperture photometry.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need Python 3 installed on your system.
The script also requires `numpy`, `matplotlib`, `astropy`, `astroquery`, `photutils`, and `warnings`. You can install these using pip:

```bash
pip install numpy matplotlib astropy astroquery photutils
```

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/p736m3dmht7t/Visualize_Aperture.git
    cd Visualize_Aperture
    ```

2.  Run the script:

    ```bash
    python Visualize_Aperture.py
    ```

## Usage

When you run the script you are asked to provide the basic inputs required as shown below:

```text
--- GAIA Third Light PSF Visualizer ---
Please enter the following parameters:
Target RA (degrees): 339.449384
Target Dec (degrees): 1.534390
Image Scale (arcseconds/pixel): 0.676
Seeing FWHM (arcseconds): 1.66
Aperture Radius (pixels): 5
Annulus Inner Radius (pixels): 10
Annulus Width (pixels): 10
Enter custom Field of View (arcseconds) or press Enter for calculated default:
Using default field of view: 54.08 arcsec (80.0 pixels across).
```

Next, there are a few informative status messages and calculations shown:

```text
Using default field of view: 54.08 arcsec (80.0 pixels across).
Querying GAIA DR3 around 22h37m47.85216s +01d32m03.804s with a radius of 0.64 arcmin
Found 3 GAIA DR3 sources.
Target found in GAIA DR3 (Gmag=11.001).
```

You will then see the output plot showing the target star and any other detections on the left plot, and with the target star not shown on the right plot.

The matplotlib controls are active in this side by side plot, so you can click on the magnifying glass icon and `click-drag` over the color bar at the right of the image to bring any dim background stars into view.

![CY Aqr](CY%20Aqr.png)





