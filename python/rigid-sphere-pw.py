import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join
from scipy.signal import (bilinear_zpk, sosfreqz, sosfilt, unit_impulse,
                          butter, lfilter, kaiser, sos2tf)
from scipy.special import eval_legendre as legendre
from util import (db, sos_pw_rigid_sphere, matchedz_zpk, group_delay_sos,
                  dotproduct_on_unitsphere, log_frequencies)


c = 343
fs = 48000 * 2

# incident plane wave (propagating direction)
theta_pw, phi_pw = np.pi/2, np.pi/2

# rigid sphere
R = 0.042
num_mic = 360
theta = np.pi/2 * np.ones(num_mic)
phi = np.linspace(-np.pi, np.pi, num=num_mic, endpoint=False)
cosTheta = dotproduct_on_unitsphere(theta, phi, theta_pw, phi_pw)

# radial filters
max_order = 20
s2z = matchedz_zpk  # {matchedz_zpk, bilinear_zpk}
# s2z = bilinear_zpk  # {matchedz_zpk, bilinear_zpk}
overall_delay = -R/c
sos = [sos_pw_rigid_sphere(n, R, s2z=s2z, fs=fs, c=343)
       for n in range(max_order+1)]

# evaluation frequencies
fmin, fmax, num_f = 20, 22000, 1000
f = log_frequencies(fmin, fmax, num_f)

# excitation
L = 1024
time = np.arange(L) / fs + overall_delay
# imp = unit_impulse(L)  # unit impulse
f_cutoff = 20000
b, a = butter(2, Wn=f_cutoff, btype='low', fs=fs)
imp = lfilter(b, a, unit_impulse(L))  # low-pass filtered pulse

# spherical harmonics expansion
modal_weight = np.ones(max_order+1)
# modal_weight = kaiser(2*max_order+1, beta=8.6)[max_order:]
h_modal = np.stack([(-1)**n * (2*n+1)
                    * sosfilt(sos[n], imp)
                    * legendre(n, cosTheta[:, np.newaxis])
                    for n in range(max_order+1)])
h_modal *= modal_weight[:, np.newaxis, np.newaxis]
h = np.sum(h_modal, axis=0)


# plots
colors = cm.viridis
dir_savefig = './'
extent = (np.rad2deg(phi[0]), np.rad2deg(phi[-1]), time[0]*1000, time[-1]*1000)
tlim = -0.5, 2.5
kw_savefig = dict(bbox_inches='tight', dpi=300)
kw_lin = dict(origin='lower', cmap='coolwarm', vmin=-0.25, vmax=0.25)
kw_db = dict(origin='lower', cmap='viridis', vmin=-200)

fig_name = 'modal-spectra-pw'
fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
for i, sos_i in enumerate(sos):
    color = colors((i+1)/(max_order+2))
    _, H = sosfreqz(sos_i, worN=f, fs=fs)
    gd = overall_delay + group_delay_sos(sos_i, f=f, fs=fs)
    ax[0].plot(f, db(H), c=color, alpha=0.5, lw=2)
    ax[1].plot(f, gd * 1000, c=color, alpha=0.5, lw=2)
for axi in ax:
    axi.grid(True)
    axi.set_xlim(fmin, fmax)
    axi.set_xlabel('Frequency in Hz')
    axi.set_xscale('log')
ax[0].set_ylim(-85, 15)
ax[0].set_ylabel('Magnitude in dB')
ax[1].set_ylabel('Group delay in ms')
ax[1].set_ylim(-2*R/c*1000, 2*R/c*1000)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)

fig_name = 'impulse-responses-pw-lin'
fig, ax = plt.subplots()
im = ax.imshow(h.T, extent=extent, **kw_lin)
ax.axis('tight')
ax.set_xlabel('Azimuth in rad')
ax.set_ylabel('Time in ms')
ax.set_ylim(tlim)
fig.colorbar(im)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)

fig_name = 'impulse-responses-pw-log'
fig, ax = plt.subplots()
im = ax.imshow(db(h.T), extent=extent, **kw_db)
ax.axis('tight')
ax.set_xlabel('Azimuth in rad')
ax.set_ylabel('Time in ms')
ax.set_ylim(tlim)
fig.colorbar(im)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)
plt.savefig(join(dir_savefig, fig_name), **kw_savefig)
