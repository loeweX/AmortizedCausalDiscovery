"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


class SpringSim(object):
    def __init__(
        self,
        n_balls=5,
        box_size=5.0,
        loc_std=0.5,
        vel_norm=0.5,
        interaction_strength=0.1,
        noise_var=0.0,
    ):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0.0, 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += (
                            0.5
                            * self.interaction_strength
                            * edges[i, j]
                            * (dist ** 2)
                            / 2
                        )
            return U + K

    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hitting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(
        self, T=10000, sample_freq=10, spring_prob=[1 / 2, 0, 1 / 2], edges=None
    ):
        n = self.n_balls
        assert T % sample_freq == 0
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        if edges is None:
            # Sample edges
            edges = np.random.choice(
                self._spring_types, size=(self.n_balls, self.n_balls), p=spring_prob
            )
            # edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            forces_size = -self.interaction_strength * edges
            np.fill_diagonal(
                forces_size, 0
            )  # self forces are zero (fixes division by zero)
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(
                axis=-1
            )  # sum over influence from different particles to get their joint force
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = -self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                                1, n, n
                            ),
                        )
                    )
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


def init():
    # init lines
    for line in lines:
        line.set_data([], [])

    return lines


def update(frame):
    for j, line in enumerate(lines):
        line.set_data(loc[frame, 0, j], loc[frame, 1, j])
    return lines


if __name__ == "__main__":
    sim = SpringSim(n_balls=5)

    t = time.time()

    loc, vel, edges = sim.sample_trajectory(T=30000, sample_freq=100)  # 5000, 100

    print(edges)
    print("Simulation time: {}".format(time.time() - t))

    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5.0, 5.0])
    axes.set_ylim([-5.0, 5.0])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], "d")

    plt.show()

    # create animation
    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
    lines = [
        plt.plot([], [], marker="$" + "{:d}".format(i) + "$", alpha=1, markersize=10)[0]
        for i in range(loc.shape[-1] - 1)
    ] + [
        plt.plot([], [], "*", alpha=1, markersize=10)[0]
    ]  # lines to animate

    ani = FuncAnimation(fig, update, frames=loc.shape[0], interval=5, blit=True)

    ani.save(filename="test.mp4", writer=writer)
