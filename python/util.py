import numpy as np
from matplotlib import rcParams
from scipy.signal import bilinear_zpk, freqs_zpk, freqz_zpk, zpk2sos, tf2sos,\
                         findfreqs, freqs, freqz, invresz
from scipy.special import spherical_jn, spherical_yn, exp1
from scipy.signal.filter_design import _is_int_type


def spherical_hn1(n, z, derivative=False):
    return spherical_jn(n, z, derivative) + 1j * spherical_yn(n, z, derivative)


def spherical_hn2(n, z, derivative=False):
    return spherical_jn(n, z, derivative) - 1j * spherical_yn(n, z, derivative)


def sos_pw_rigid_sphere(n, r, s2z, fs, c=343):
    if n == 0:
        f0 = 1000
    else:
        f0 = c * n / 2 / np.pi / r
    f0 = 10
    s_zeros, s_poles, s_gain = s_zpk_hn2_poly(n)
#    if s2z is bilinear_zpk:  # pre-warping: scale the pole radii
#        rmean = np.mean(np.abs(s_poles))
#        s_poles *= 2*fs*np.tan(rmean/fs/2) / rmean
#        omega_ref = 2 * np.pi * f0
#        s_poles *= 2 * fs * np.tan(omega_ref / fs / 2) / omega_ref
    s_poles *= c/r
    s_gain *= c/r
    z_zeros, z_poles, z_gain = s2z_zpk(s_zeros, s_poles, s_gain, s2z, fs, f0)
    return zpk2sos(z_zeros, z_poles, z_gain)


def sos_ps_rigid_sphere(n, r, rs, s2z, fs, c=343):
    s_zeros, s_poles, s_gain = s_zpk_ps_rigid_sphere(n)
    s_zeros *= c/rs
    s_poles *= c/r
    s_gain *= c / r / rs
    # f0 = c * n / 2 / np.pi / r
    f0 = 0.5 * fs/2
    z_zeros, z_poles, z_gain = s2z_zpk(s_zeros, s_poles, s_gain, s2z, fs, f0)
    return zpk2sos(z_zeros, z_poles, z_gain)


def hn2_poly(n, derivative=False):
    """Bessel polynomial of n-th order.
    Polynomial that characterizes the spherical Hankel functions.
    The coefficients are computed by using the recurrence relation.
    The returned array has a length of n+1. The first coefficient is always 1.

    Parameters
    ----------

    n : int
        Bessel polynomial order.

    """
    if derivative:
        return derivative_hn2_poly(n)
    else:
        beta = np.zeros(n + 1)
        beta[n] = 1
        for k in range(n-1, -1, -1):
            beta[k] = beta[k+1] * (2*n-k) * (k+1) / (n-k) / 2
        return beta


def decrease_hn2_poly_order_by_one(beta):
    """Bessel polynomial of order decreased by 1.
    """
    n = len(beta)-1
    alpha = np.zeros(n)
    for k in range(n-1):
        alpha[k] = beta[k+1] * (k+1) / (2*n-k-1)
    alpha[-1] = 1
    return alpha


def increase_hn2_poly_order_by_one(beta):
    """Bessel polynomial of order increased by 1.
    """
    n = len(beta)
    alpha = np.zeros(n+1)
    for k in range(n):
        alpha[k+1] = beta[k] * (2*n-k-1) / (k+1)
    alpha[0] = alpha[1]
    return alpha


def derivative_hn2_poly(n):
    """Polynomial characterizing the derivative of the spherical Hankel func.
    """
    gamma = hn2_poly(n+1)
    gamma[:-1] -= n * decrease_hn2_poly_order_by_one(gamma)
    return gamma


def s_zeros_hn2_poly(n):
    return np.roots(hn2_poly(n)[::-1])


def s_zeros_derivative_hn2_poly(n):
    return np.zeros(n)


def s_poles_derivative_hn2_poly(n):
    return np.roots(derivative_hn2_poly(n)[::-1])


def s_zpk_hn2_poly(n):
    s_zeros = s_zeros_derivative_hn2_poly(n)
    s_poles = s_poles_derivative_hn2_poly(n)
    return s_zeros, s_poles, 1


def s_zpk_ps_rigid_sphere(n):
    s_zeros = s_zeros_hn2_poly(n)
    s_poles = s_poles_derivative_hn2_poly(n)
    return s_zeros, s_poles, 1


def matchedz_zpk(s_zeros, s_poles, s_gain, fs):
    """Matched-z transform of poles and zeros.
    """
    z_zeros = np.exp(s_zeros / fs)
    z_poles = np.exp(s_poles / fs)
    z_gain = s_gain
    return z_zeros, z_poles, z_gain


def s2z_zpk(s_zeros, s_poles, s_gain, s2z, fs, f0):
    z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs=fs)
    if s2z is matchedz_zpk:
        z_gain *= np.abs(freqs_zpk(s_zeros, s_poles, s_gain, worN=[2*np.pi*f0])[1])\
                  /np.abs(freqz_zpk(z_zeros, z_poles, z_gain, worN=[f0], fs=fs)[1])
    return z_zeros, z_poles, z_gain


def fd_radial_function(n, kr):
    return -1j / kr**2 / spherical_hn2(n, kr, derivative=True)


def shpak(b, w, fs=1):
    T = 1/fs
    b0, b1, b2 = b
    c1 = np.cos(w*T)
    c2 = np.cos(2*w*T)
    u0, u1, u2 = b0**2, b1**2, b2**2
    v0, v1, v2 = b0*b1, b1*b2, b0*b2
    num = (u1+2*u2) + (v0+3*v1)*c1 + 2*v2*c2
    den = (u0+u1+u2) + 2*(v0+v1)*c1 + 2*v2*c2
    return T * num / den


def shpak_analog(b, w):
    b0, b1, b2 = b
    w2 = w**2
    w4 = w**4
    if b0 == 0:
        if b1 == 0 or b2 == 0:
            return np.zeros_like(w)
        else:
            return -(b1 * b2) / (b1**2 * w2 + b2**2)
    elif b2 == 0:
        if b1 == 0:
            return np.zeros_like(w)
        else:
            return -(b0 * b1) / (b0**2 * w2 + b1**2)
    else:
        return -((b0 * b1 * w2 + b1 * b2)
                 / (b0**2 * w4 + (b1**2 - 2 * b0 * b2) * w2 + b2**2))



def group_delay_iir(b, a, f, fs=1):
    """
    Group delay of a digital filter.
    """
    sos = tf2sos(b, a)
    gd = 0
    for sosi in sos:
        gd += shpak(sosi[:3], 2*np.pi*f, fs=fs)
        gd -= shpak(sosi[3:], 2*np.pi*f, fs=fs)
    return gd


def group_delay_sos(sos, f, fs=1):
    """
    Group delay of a digital second-order section filter.
    """
    gd = 0
    for sosi in sos:
        gd += shpak(sosi[:3], 2*np.pi*f, fs=fs)
        gd -= shpak(sosi[3:], 2*np.pi*f, fs=fs)
    return gd


def phaseshift_timedelay(delay, w):
    return np.exp(-1j * 2 * np.pi * w * delay)


def phaseshift_sampledelay(n, w, fs):
    return phaseshift_timedelay(delay=n/fs, w=w)


def dotproduct_on_unitsphere(theta1, phi1, theta2, phi2):
    return np.sin(theta1) * np.sin(theta2) * np.cos(phi1-phi2)\
           + np.cos(theta1)*np.cos(theta2)


def s2ms(t):
    return t * 1000


def log_frequencies(fmin, fmax, num_f, endpoint=True):
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_f,
                       endpoint=endpoint)

def db(x, *, power=False):
    """Convert *x* to decibel.

    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.

    """
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\alpha \in [-pi, pi], \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates

    Returns
    -------
    alpha : float or `numpy.ndarray`
            Azimuth angle in radiants
    beta : float or `numpy.ndarray`
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / r)
    return alpha, beta, r
