"""Time domain filters for random noise.

This file contains the functions of the constituant models that
are combined to compute the stochastic ground motion model for arbitrary
earthquake scenarios.

This file can also be imported as a module and contains the following
functions:

motion_factor
    The scaling of ground motion to displacement, velocity or acceleration.

source_scf
    A generic single corner frequency model for a seismic source.

"""
import numpy as np
from typing import Union


def motion_factor(f: np.ndarray,
                  motion: str = 'disp'
                  ) -> Union[np.ndarray, None]:
    """
    A function to scale the source spectrum to the desired motion parameter.
    Add to differentiate and subtract to integrate.

    Parameters
    ----------
    """

    gms = ['disp', 'vel', 'acc']

    if motion.lower() not in gms:
        raise ValueError(f"Motion must be {gms}")

    if motion.lower() == 'disp':
        return 0

    if motion.lower() == 'vel':
        return np.log10(2 * np.pi * f)

    if motion.lower() == 'acc':
        return np.log10(np.power(2 * np.pi * f, 2))


# DEFAULT PARAMS FOR SOURCE MODE:
# BRUNE_MODEL = (1, 2) # omega squared
# BOATWRIGHT_MODEL = (2, 2) # omega cubed


def source_scf(f: np.ndarray,
               llpsp: float,
               fc: float,
               gam: float,
               n: float
               ) -> np.ndarray:
    """
    Generic single corner frequency model for the far-field displacement
    spectrum of an arbitrary seismic source as a function of frequency (log
    base 10 representation).

    Parameters
    ----------

    llpsp :
        Log10 amplitude of the long period plateau (e.g. log10[M0])
    """
    return llpsp - (1 / gam) * np.log10((1 + (f / fc)**(gam * n)))


def f_idep_attenutation(f: np.ndarray,
                        Q: float,
                        R: float,
                        b: float,
                        ) -> np.ndarray:
    """
    Frequency independent attenuation model (log base 10 representation).

    Parameters
    ----------
    f : np.ndarray
        The frequencies to compute the attenuation for.
    Q : float
        The frequency independent quality factor.
    R : float
        The propagation distance in km or m.
    b : float
        Average seismic velocity along the path in km of m.
    """
    return -(((np.pi * f * R) / (Q * b)) / np.log(10))


def f_dep_attenuation(f: np.ndarray,
                      a: float,
                      Q: float,
                      R: float,
                      b: float,
                      ) -> np.ndarray:
    """
    Frequency dependent attenuation model (log base 10 representation).

    Parameters
    ----------
    f : np.ndarray
        The frequencies to compute attenuation over [Hz].
    a : float
        The frequency dependent factor for attenuation.
    Q : float
        The frequency independent quality factor.
    R : float
        The propagation distance [km or m].
    b : float
        Average seismic velocity along the path [km or m].
    """
    assert 0 <= a < 1, "a must be in range 0 <= a < 1."
    return -(np.pi * (f**(1 - a)) * (R / Q * b) / np.log(10))


def single_geospreading(R: Union[float, np.ndarray], p=1):
    """

    """
    return -p * np.log10(R)
