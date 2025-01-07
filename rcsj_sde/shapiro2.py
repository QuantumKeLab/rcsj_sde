import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple

from rcsj_sde.junction import JosephsonJunction
from rcsj_sde.simu import simulation_voltage


def run_shapiro(jj: JosephsonJunction,
                I_DC_range: jnp.ndarray,
                F_ac: float,
                tau_max: int,
                tau_points: int,
                powers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run a Shapiro simulation.
    
    In a Shapiro experiment, both an AC drive and a DC current bias is applied and varied.

    Parameters
    ----------
    jj : JosephsonJunction
        Josephson junction on which the Shapiro simulation is carried out.
    I_DC_range : jnp.ndarray
        DC current bias values
    F_ac : float
        Frequency of the AC drive in hertz.
    tau_max : int
        Length of the time-domain simulation in normalized units.
    tau_points : int
        Number of time points.
    powers : jnp.ndarray
        Power values of the AC drive in dBm.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        (differential resistance dV/dI, junction voltage V), dVdI has shape (N_power, N_I-1),
        and V has shape (N_power, N_I)
    """
    power_points = len(powers)
    I_points = len(I_DC_range)

    dVdI = jnp.empty((power_points, I_points - 1))
    V = jnp.empty((power_points, I_points))
    
    # variables for the power transfer calculation
    Zc = 1 / (1j * 2 * jnp.pi * F_ac * jj.C)
    Zjj = Zc*jj.R/(Zc + jj.R) 
    Z0 = 50
    Gamma = (Zjj - Z0)/(Zjj + Z0)

    dI = I_DC_range[1] - I_DC_range[0]

    for pi, power in enumerate(tqdm(powers, leave=True)):
        V0 = (10**((power - 30)/10)*Z0)**0.5 * 2**0.5 # voltage amplitude (for perfect match)
        I_ac = abs(V0/Z0*(1 - Gamma)) # taking into account the impedance mismatch 
        
        V_sim = simulation_voltage(jj=jj,
                               I_DC_range=I_DC_range,
                               I_ac=I_ac,
                               F_ac=F_ac,
                               tau_max=tau_max,
                               tau_points=tau_points,
                               progressbar=False)
        V[pi] = V_sim
        dVdI[pi] = jnp.diff(V_sim) / dI

    return dVdI, V