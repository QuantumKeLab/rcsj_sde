import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.special import erf

def rng_wrapper(key, h, N):
    """
    Generate random numbers with Gaussian distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generator
    h : float
        Time step size
    N : int
        Number of random numbers

    Returns
    -------
    jnp.ndarray
        N independent Gaussian-distributed N(0, sqrt(h)) random numbers
    """
    return random.normal(key, (N,)) * jnp.sqrt(h)

@jit
def rcsj_solver(key, epsilon: float,
                beta: float,
                i_dc: float,
                a: float,
                b: float,
                y0: jnp.ndarray,
                tspan: jnp.ndarray,
                i_ac: float=0.0,
                f_ac: float=0.0) -> jnp.ndarray:
    """
    Solve the RCSJ SDE with Heun's method. The RCSJ SDE is formulated
    in normalized units.
    
    For a reference on Heun's method, see:
    Numerical Treatment of Stochastic Differential Equations
    W. Rümelin
    SIAM Journal on Numerical Analysis
    Vol. 19, No. 3 (Jun., 1982), pp. 604-613 
    https://www.jstor.org/stable/2156972?origin=JSTOR-pdf

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generator
    epsilon : float
        Noise term, epsilon=sigma/beta.
    beta : float
        Stewart-McCumber parameter
    i_dc : float
        DC bias current (in normalized units)
    a : float
        Prefactor of trivial term in the current-phase relation, a*sin(phi)
    b : float
        Prefactor of topological term in the current-phase relation, b*sin(phi/2)         
    y0 : jnp.ndarray
        Initial conditions, phi and phidot
    tspan : jnp.ndarray
        Time span of the solution (in normalized units)
    i_ac : float
        Amplitude of the AC bias current (in normalized units)
    f_ac : float
        Frequency of the AC bias current (in normalized units)
        
    Returns
    -------
    jnp.ndarray
        Solution of the SDE, phi and phidot
    """
    d = len(y0)
    N = len(tspan)
    h = tspan[1] - tspan[0]
    key, subkey = random.split(key)
    dW = rng_wrapper(subkey, h, N)
    y = jnp.zeros((N, d), dtype=y0.dtype)
    f1 = jnp.zeros((2,), dtype=y0.dtype)
    f2 = jnp.zeros_like(f1)
    y = y.at[0].set(y0)

    for n in range(0, N-1):
        yn = y[n] 
        itot = i_dc + i_ac * jnp.sin(2 * jnp.pi * f_ac * n * h)
        f1 = jnp.array([yn[1] * h, (1 / beta) * (itot - (a * jnp.sin(yn[0]) + b * jnp.sin(yn[0] / 2)) - yn[1]) * h])
        k1 = yn + f1 
        f2 = jnp.array([k1[1] * h, (1 / beta) * (itot - (a * jnp.sin(k1[0]) + b * jnp.sin(k1[0] / 2)) - k1[1]) * h])
        Yn1 = yn + 0.5 * (f1 + f2)
        Yn1 = Yn1.at[1].add(epsilon * dW[n])  # add noise
        y = y.at[n+1].set(Yn1)

    return y