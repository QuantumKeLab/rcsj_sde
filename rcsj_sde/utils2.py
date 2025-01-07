import jax.numpy as jnp
from scipy.integrate import quad
from scipy.special import iv  # modified Bessel function of the 1st kind
from typing import Union, Optional

# reduced Planck constant / (2*elementary charge) in SI units [weber]
hbar_over_2e = 3.29106e-16
# Boltzmann constant in SI units [joule/kelvin]
kB = 1.380649e-23


def overdamped_zerotemperature(I: Union[float, jnp.ndarray], I_c: float, R: float) -> Union[float, jnp.ndarray]:
    """
    Calculate the junction voltage in the overdamped limit for T=0.

    Parameters
    ----------
    I : Union[float, jnp.ndarray]
        Bias current
    I_c : float
        Critical current of the junction
    R : float
        Parallel resistance

    Returns
    -------
    jnp.ndarray or float
        Junction voltage, same shape as ijnput I
    """
    V = jnp.where(I < 0,
                 -jnp.real(I_c * R * jnp.emath.sqrt((I/I_c)**2 - 1)),
                 jnp.real(I_c * R * jnp.emath.sqrt((I/I_c)**2 - 1)))
    return V


def linear_reference(I: Union[float, jnp.ndarray], R: float) -> Union[float, jnp.ndarray]:
    """
    Calculate the voltage drop over resistor R at current I.

    Parameters
    ----------
    I : Union[float, jnp.ndarray]
        Current
    R : float
        Resistance

    Returns
    -------
    Union[float, jnp.ndarray]
        Voltage on resistor, same shape as ijnput I
    """
    V = I*R
    return V

def ambegaokar_overdamped(I: Union[float, jnp.ndarray],
                          I_c: float,
                          R: float,
                          T: Optional[float] = None,
                          gamma0: Optional[float] = None) -> Union[float, jnp.ndarray]:
    """
    Calculate the junction voltage in the overdamped limit for arbitrary T>=0.
    
    See Eq. 3.4.21 in the textbook Applied Superconductivity by Prof. Dr. Rudolf Gross and  Dr. Achim Marx:

    https://www.wmi.badw.de/fileadmin/WMI/Lecturenotes/Applied_Superconductivity/AS_Chapter3.pdf

    Either provide T or gamma0. 

    Parameters
    ----------
    I : Union[float, jnp.ndarray]
        Current bias
    I_c : float
        Junction critical current
    R : float
        Parallel shunt resistor
    T : float, optional
        Temperature in kelvin, by default None
    gamma0 : float, optional
        Gamma parameter, by default None

    Returns
    -------
    Union[float, jnp.ndarray]
        Junction voltage, same shape as ijnput I
    """

    def integrand(phi, i):
        # iv: modified Bessel function
        return jnp.exp(-i*gamma0*phi/2)*iv(0, gamma0*jnp.sin(phi/2))

    if T is None and gamma0 is None:
        raise ValueError("Either T or gamma0 must be provided")

    if T is not None and gamma0 is not None:
        raise ValueError(
            "Arguments T and gamma0 cannot be provided simultaneously")

    if T is not None:
        if T == 0:
            return overdamped_zerotemperature(I, I_c, R)
        elif T > 0:
            # 3.4.18 (flux quantum = hbar_over_2e*2*jnp.pi)
            gamma0 = hbar_over_2e*2*jnp.pi*I_c/(jnp.pi*kB*T)
        else:
            raise ValueError("Argument T cannot be negative")

    if isinstance(I, jnp.ndarray):
        ispan = I/I_c
        integral = jnp.zeros_like(ispan)
        for i_idx in range(len(ispan)):
            integral[i_idx] = quad(integrand, 0, 2*jnp.pi,
                                   args=(ispan[i_idx],))[0]
        x = ispan*jnp.pi*gamma0

    else:
        i = I/I_c
        integral = quad(integrand, 0, 2*jnp.pi, args=(i,))[0]
        x = i*jnp.pi*gamma0

    return 2*I_c*R/gamma0*(1 - jnp.exp(-x))*1/integral


def v2_to_dbm(psd: Union[float, jnp.ndarray], R: float = 50) -> Union[float, jnp.ndarray]:
    """
    Convert power spectral density from V^2/Hz to dBm/Hz.

    Parameters
    ----------
    psd : Union[float, jnp.ndarray]
        Power spectral density in V^2/Hz
    R : float, optional
        Resistance, by default 50

    Returns
    -------
    Union[float, jnp.ndarray]
        Power spectral density in dBm/Hz
    """
    return 10*jnp.log10(psd/R) + 30


def watt_to_dbm(p: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Convert power from watt to dBm

    Parameters
    ----------
    p : Union[float, jnp.ndarray]
        Power in watt

    Returns
    -------
    Union[float, jnp.ndarray]:
        Power in dBm
    """
    return 10*jnp.log10(p) + 30


def thermal_noise_voltage(T: Union[float, jnp.ndarray], R: float) -> Union[float, jnp.ndarray]:
    """
    Calculate the thermal noise voltage of a resistor per unit bandwidth in <V**2>/Hz

    Parameters
    ----------
    T : float
        Temperature in kelvin
    R : float
        Resistor value

    Returns
    -------
    Union[float, jnp.ndarray]
        Thermal noise voltage in <V**2>/Hz
    """
    return 4*kB*T*R

def thermal_noise_power(T: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Calculate the thermal noise power of a resistor per unit bandwidth in <P>/Hz.

    Parameters
    ----------
    T : float
        Temperature in kelvin

    Returns
    -------
    Union[float, jnp.ndarray]
        Thermal noise power in <P>/Hz
    """
    return 4*kB*T

    
def symmetrize(array: jnp.ndarray) -> jnp.ndarray:
    """
    Generate symmetrized array by mirroring, tailored for Shapiro plots.
    
    For a 1D array (e.g. bias current), it negates and flips the mirrored part.
    For a 2D array (e.g. voltage output of Shapiro simulation), it only flips
    along axis 1 to make the mirrored part.

    Parameters
    ----------
    array : jnp.ndarray
        1D or 2D array to be symmetrized

    Returns
    -------
    jnp.ndarray
        Symmetrized array

    Raises
    ------
    Exception
        If ndim > 2, raises an error.
    """
    if array.ndim == 1:
        array_sym = array[:-1]
        array_sym = jnp.concatenate([-array_sym[::-1], array_sym[1:]])
    elif array.ndim == 2:
        r, c = array.shape
        array_sym = jnp.zeros((r, 2*c-1))
        array_sym[:, c-1:] = array
        array_sym[:, :c] = jnp.fliplr(array)
    else:
        raise Exception("Only works for 1D and 2D arrays.")

    return array_sym
