"""
Create the zernike modes as a fits file. Skip tip/tilt and piston.

We want to get up to the 4th order modes.
focus/astig: 3 modes
coma and trefoil: 4 modes
4th order other than spherical: 4 modes.
"""
import numpy as np
import math
from astropy.io import fits

# Outer radius of the modes in actuators - truncate
# Zernike modes at this point.
OUTER_RADIUS = 5.5  
N_ACTUATORS = 12  # Number of actuators across the DM
N_MODES = 11  # Number of modes to create

def zernike(n, m, rho, theta):
    """Compute the Zernike polynomial of radial order n and azimuthal frequency m."""
    if m > 0:
        return np.sqrt(2) * zernike_radial(n, m, rho) * np.cos(m * theta)
    elif m < 0:
        return np.sqrt(2) * zernike_radial(n, -m, rho) * np.sin(-m * theta)
    else:
        return zernike_radial(n, 0, rho)

def zernike_radial(n, m, rho):
    """Compute the radial part of the Zernike polynomial."""
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        R += (-1)**k * math.factorial(n - k) / (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)) * rho**(n - 2*k)
    return R

def create_modes():
    # create a grid of actuator points
    X = np.arange(N_ACTUATORS) - (N_ACTUATORS - 1) / 2
    Y = np.arange(N_ACTUATORS) - (N_ACTUATORS - 1) / 2
    X, Y = np.meshgrid(X, Y)
    rho = np.sqrt(X**2 + Y**2) / OUTER_RADIUS
    rho = np.clip(rho, 0, 1)  # Limit to the unit circle
    theta = np.arctan2(Y, X)
    modes = np.zeros((N_MODES, N_ACTUATORS, N_ACTUATORS))
    # Define the modes we want to create (n, m)
    mode_indices = [(2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (3, 3), (3, -3), (4, -2), (4, 2), (4, -4), (4, 4)]
    for i, (n, m) in enumerate(mode_indices):
        modes[i] = zernike(n, m, rho, theta)
    return modes    

if __name__ == "__main__":
    modes = create_modes()
    # Save the modes to a fits file
    hdu = fits.PrimaryHDU(modes.reshape(N_MODES, N_ACTUATORS*N_ACTUATORS))
    hdu.writeto("modes.fits", overwrite=True)