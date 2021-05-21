"""Physical earthquake source parameters.

This file contains the functions that compute physical earthquake source
parameters assuming .

This file can also be imported as a module and contains the following
functions:

    - fc
    - mo_from_mw
    - mw

"""
import numpy as np


def fc(vs: float, sd: float, mo: float, c: float = 0.49) -> float:

    """
    The formula for corner frequnecy assuming constant stress drop 
    (Aki, 1967; Brune, 1970, 1971). The default assumes SI units 
    for all parameters, if using cgs override c=0.49 to c=4.9E6 
    (e.g., Boore, 2003).

    Parameters
    ----------
    vs : float
        The shear-wave velocity at the rupture source (m [default] or km).
    sd : float
        The total stress drop (Δσ) of the rupture (Pa [default] or Bar).
    mo : float
        The total scalar seismic moment (M0) release of the rupture (N m
        [default] or dyne-cm)
    c : float
        A scaling constant (assuming self-similarity) relating stress drop
        and corner frequency (SI [default] or cgs units).

    Returns
    -------
    float
        The theoretical corner frequency of a double-couple point source 
        earthquake.
    """
    return c * vs * (sd / mo)**(1 / 3) 


def sd(fc: float, mo: float, vs: float, k: float = 0.37) -> float:

    """
    The formula for stress drop assuming a circular crack model (Eshelby, 1957,
    Aki, 1967; Brune, 1970, 1971). k depends on wave type (P or S). Default
    value is for S-waves. The default assumes SI units for all parameters.

    Parameters
    ----------
    fc : float
        The radially averaged corner frequency of the source spectrum (Hz).
    mo : float
        The total scalar seismic moment (M0) release of the rupture (N m
        [default] or dyne-cm)
    vs : float
        The shear-wave velocity at the rupture source (m [default] or km).
    k : float
        A scaling constant that depends on wave type (P or S). S is the default
        value.

    Returns
    -------
    float
        The theoretical stress drop in MPa.
    """
    
    return (7 / 16) * mo * (fc / (k * vs))**3 


def mo_from_mw(mw: float, c: float = 6.0333) -> float:
    """
    A formula for converting moment magnitude (Hanks and Kanamori, 1979) to
    seismic moment. The default assumes SI units for seismic moment, if
    using cgs override c=6.0333 to c=10.7 (Boore, 2003).

    Parameters
    ----------
    mw : float
        The moment magnitude.

    c : float
        A constant that maps log10-seismic moment to moment magnitude
        (SI [default] or cgs units).

    Returns
    -------
    float
        The seismic moment for a given Moment Magnitude.
    """

    return 10**((3 / 2) * (mw + c))


def mw(mo: float, c: float = 6.0333) -> float:
    """
    A formula for converting moment magnitude (Hanks and Kanamori, 1979) to
    seismic moment. The default assumes SI units for seismic moment, if
    using cgs override c=6.0333 to c=10.7 (Boore, 2003).

    Parameters
    ----------
    mw : float
        The moment magnitude.

    c : float
        A constant that maps log10-seismic moment to moment magnitude
        (SI [default] or cgs units).

    Returns
    -------
    float
        The moment magnitude for a given seismic moment.
    """

    return (2 / 3) * np.log10(mo) + c


def moment_scaling(vs: float = 3500.0,
                   rho: float = 2600.0,
                   ro: float = 1000.0,
                   fs: float = 2.0,
                   rp: float = 0.55
                   ) -> float:
    """
    The displacement to moment scaling factor (log10 units) at the source.
    The constants for rp are taken

    Parameters
    ----------
    vs : float
        The shear-wave velocity at the rupture source (m [default] or km).

    rho : float
        The shear-wave density at the rupture source (kgm^-3 [default] or
        gcm^-3).

    ro : float
        The reference distance from the source (m[default]=1000 or km).

    fs : float
        Free surface amplification factor (2 for SH waves).

    rp : float
        The radiation partitioning coefficient (Boore and Boatwright)
        https://pubs.geoscienceworld.org/ssa/bssa/article/74/5/1615/118641/Average-body-wave-radiation-coefficients
    
    Returns
    -------
    float
        The scaling parameter from long period displacement to seismic moment.
    """
    return np.log10((4 * np.pi * vs**3 * rho * ro) / (fs * rp))
