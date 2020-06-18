#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 12/03/2017
#
# Feel free to contact for any information.
#
# You can cite this code by referencing:
#   D. Laszuk, "Python implementation of Kuramoto systems," 2017-,
#   [Online] Available: http://www.laszukdawid.com/codes
#
# LICENCE:
# This program is free software on GNU General Public Licence version 3.
# For details of the copyright please see: http://www.gnu.org/licenses/.

from __future__ import print_function

import numpy as np
from scipy.integrate import ode

__version__ = '0.3'
__author__ = 'Dawid Laszuk'

class Kuramoto(object):
    """
    Implementation of Kuramoto coupling model [1] with harmonic terms
    and possible perturbation.
    It uses NumPy and Scipy's implementation of Runge-Kutta 4(5)
    for numerical integration.

    Usage example:
    >>> kuramoto = Kuramoto(initial_values)
    >>> phase = kuramoto.solve(X)

    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
        (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """

    _noises = { 'logistic': np.random.logistic,
                'normal': np.random.normal,
                'uniform': np.random.uniform,
                'custom': None
              }

    noise_types = _noises.keys()

    def __init__(self, init_values, noise=None):
        """
        Passed arguments should be a dictionary with NumPy arrays
        for initial phase (Y0), intrisic frequencies (W) and coupling
        matrix (K).
        """
        self.dtype = np.float32

        self.dt = 1.
        self.init_phase = np.array(init_values['Y0'])
        self.W = np.array(init_values['W'])
        self.K = np.array(init_values['K'])

        self.n_osc = len(self.W)
        self.m_order = self.K.shape[0]

        self.noise = noise


    @property
    def noise(self):
        """Sets perturbations added to the system at each timestamp.
        Noise function can be manually defined or selected from
        predefined by assgining corresponding name. List of available
        pertrubations is reachable through `noise_types`. """
        return self._noise

    @noise.setter
    def noise(self, _noise):

        self._noise = None
        self.noise_params = None
        self.noise_type = 'custom'

        # If passed a function
        if callable(_noise):
            self._noise = _noise

        # In case passing string
        elif isinstance(_noise, str):

            if _noise.lower() not in self.noise_types:
                self.noise_type = None
                raise NameError("No such noise method")

            self.noise_type = _noise.lower()
            self.update_noise_params(self.dt)

            noise_function = self._noises[self.noise_type]
            self._noise = lambda: np.array([noise_function(**p) for p in self.noise_params])

    def update_noise_params(self, dt):
        self.scale_func = lambda dt: dt/np.abs(self.W**2)
        scale = self.scale_func(dt)

        if self.noise_type == 'uniform':
            self.noise_params = [{'low':-s, 'high': s} for s in scale]
        elif self.noise_type in self.noise_types:
            self.noise_params = [{'loc':0, 'scale': s} for s in scale]
        else:
            pass

    def kuramoto_ODE(self, t, y, arg):
        """General Kuramoto ODE of m'th harmonic order.
           Argument `arg` = (w, k), with
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
            """

        w, k = arg
        yt = y[:,None]
        dy = y-yt
        phase = w.astype(self.dtype)
        if self.noise != None:
            n = self.noise().astype(self.dtype)
            phase += n
        for m, _k in enumerate(k):
            phase += np.sum(_k*np.sin((m+1)*dy),axis=1)

        return phase

    def kuramoto_ODE_jac(self, t, y, arg):
        """Kuramoto's Jacobian passed for ODE solver."""

        w, k = arg
        yt = y[:,None]
        dy = y-yt

        phase = [m*k[m-1]*np.cos(m*dy) for m in range(1,1+self.m_order)]
        phase = np.sum(phase, axis=0)

        for i in range(self.n_osc):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

    def solve(self, t):
        """Solves Kuramoto ODE for time series `t` with initial
        parameters passed when initiated object.
        """
        dt = t[1]-t[0]
        if self.dt != dt and self.noise_type != 'custom':
            self.dt = dt
            self.update_noise_params(dt)

        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        kODE.set_integrator("dopri5")

        # Set parameters into model
        kODE.set_initial_value(self.init_phase, t[0])
        kODE.set_f_params((self.W, self.K))
        kODE.set_jac_params((self.W, self.K))

        if self._noise != None:
            self.update_noise_params(dt)

        phase = np.empty((self.n_osc, len(t)))

        # Run ODE integrator
        for idx, _t in enumerate(t[1:]):
            phase[:,idx] = kODE.y
            kODE.integrate(_t)

        phase[:,-1] = kODE.y

        return phase


def simulate_kuramoto(num_atoms, num_timesteps=10000, T=None, dt=0.01, undirected=False):
    if T is None:
        # num_timesteps = int((10000 / float(100)) - 1)
        # t0, t1, dt = 0, int((10000 / float(100)) / 10), 0.01
        dt = 0.01
        t0, t1= 0, int(num_timesteps * dt * 10)

        T = np.arange(t0, t1, dt)

    intrinsic_freq = np.random.rand(num_atoms) * 9 + 1.
    initial_phase = np.random.rand(num_atoms) * 2 * np.pi
    edges = np.random.choice(2, size=(num_atoms, num_atoms), p=[0.5, 0.5])
    if undirected:
        edges = np.tril(edges) + np.tril(edges, -1).T    ## created symmetric edges matrix (i.e. undirected edges)
    np.fill_diagonal(edges, 0)

    kuramoto = Kuramoto({'W': intrinsic_freq,
                         'K': np.expand_dims(edges, 0),
                         'Y0': initial_phase})

    # kuramoto.noise = 'logistic'
    odePhi = kuramoto.solve(T)

    # Subsample
    phase_diff = np.diff(odePhi)[:, ::10] / dt
    trajectories = np.sin(odePhi[:, :-1])[:, ::10]

    # Normalize dPhi (individually)
    min_vals = np.expand_dims(phase_diff.min(1), 1)
    max_vals = np.expand_dims(phase_diff.max(1), 1)
    phase_diff = (phase_diff - min_vals) * 2 / (max_vals - min_vals) - 1

    # Get absolute phase and normalize
    phase = odePhi[:, :-1][:, ::10]
    min_vals = np.expand_dims(phase.min(1), 1)
    max_vals = np.expand_dims(phase.max(1), 1)
    phase = (phase - min_vals) * 2 / (max_vals - min_vals) - 1

    # If oscillator is uncoupled, set trajectory to dPhi to 0 for all t
    isolated_idx = np.where(edges.sum(1) == 0)[0]
    phase_diff[isolated_idx] = 0.

    # Normalize frequencies to [-1, 1]
    intrinsic_freq = (intrinsic_freq - 1.) * 2 / (10. - 1.) - 1.

    phase_diff = np.expand_dims(phase_diff, -1)[:, :num_timesteps, :]
    trajectories = np.expand_dims(trajectories, -1)[:, :num_timesteps, :]
    phase = np.expand_dims(phase, -1)[:, :num_timesteps, :]
    intrinsic_freq = np.expand_dims(np.repeat(
        np.expand_dims(intrinsic_freq, -1),
        num_timesteps, axis=1), -1)

    sim_data = np.concatenate(
        (phase_diff, trajectories, phase, intrinsic_freq),
        -1)

    return sim_data, edges


######################################

if __name__ == "__main__":
    import pylab as plt

    ####################################################
    t0, t1, dt = 0, 40, 0.05
    T = np.arange(t0, t1, dt)


    # Y0, W, K are initial phase, intrisic freq and
    # coupling K matrix respectively
    _Y0 = np.array([0, np.pi, 0, 1, 5, 2, 3])
    _W = np.array([28, 19, 11, 9, 2, 4])
    _K = np.array([[ 2.3844,  1.2934,  0.6834,  2.0099,  1.9885],
                   [ -2.3854,  3.6510,  2.0467,  3.6252,  3.2463],
                   [ 10.1939,  4.4156,  1.1423,  0.2509,  4.1527],
                   [ 3.8386,  2.8487,  3.4895,  0.0683,  0.8246],
                   [ 3.9127,  1.2861,  2.9401,  0.1530,  0.6573]])
    _K2 = np.array([[ 0.2628,  0.0043,  0.9399,  0.5107,  0.9857],
                   [ 0.8667,  0.8154,  0.4592,  0.9781,  0.0763],
                   [ 0.3723,  0.3856,  0.8374,  0.8812,  0.9419],
                   [ 0.1869,  0.2678,  0.9704,  0.2823,  0.3404],
                   [ 0.1456,  0.7341,  0.1389,  0.5602,  0.3823]])

    _K = np.dstack((_K, _K2)).T

    # Preparing oscillators with Kuramoto model
    oscN = 3 # num of oscillators

    Y0 = _Y0[:oscN]
    W = _W[:oscN]
    K = _K[:,:oscN,:oscN]

    init_params = {'W':W, 'K':K, 'Y0':Y0}

    kuramoto = Kuramoto(init_params)
    kuramoto.noise = 'logistic'
    odePhi = kuramoto.solve(T)
    odeT = T[:-1]

    ##########################################
    # Plot the phases
    plt.figure()

    for comp in range(len(W)):
        plt.subplot(len(W),1,comp+1)
        plt.plot(odeT, np.diff(odePhi[comp])/dt,'r')
        plt.ylabel('$\dot\phi_%i(t)$'%(comp+1))

    plt.suptitle("Instantaneous frequencies")
    plt.savefig('phases')

    # Display plot
    plt.show()