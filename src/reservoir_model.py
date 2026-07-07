"""
reservoir_model.py module : a simple linear lumped rainfall-runoff model
"""

import numpy as np
from matplotlib import pyplot as plt


class ReservoirModel:
    """A class to simulate a lumped rainfall-runoff 'reservoir-like' model
    """

    def __init__(self, alpha=0.85, beta=0.5, gamma=0.9, s_0=0.):
        """Class constructor
        """

        # Initial condition
        if s_0 < 0.:
            raise ValueError(f"Input s_0 value must be >= 0, got {s_0}")
        self._s_0 = s_0             # Initial storage in the reservoir [m3]

        # Model parameters
        if alpha < 0. or alpha > 1.:
            raise ValueError(f"Input alpha value must be in [0,1], got {alpha}")
        self._alpha = alpha       # Storage loss coefficient in [0,1]

        if beta <= 0.:
            raise ValueError(f"Input beta value must be > 0., got {beta}")
        self._beta = beta         # Rainfall-to-storage gain coefficient >0

        if gamma <= 0.:
            raise ValueError(f"Input gamma value must be > 0., got {gamma}")
        self._gamma = gamma       # Storage-to-discharge conversion coefficient >0

    def _single_iteration(self, s_t=None, p_t=None):
        """A single time iteration
        :param s_t:
        :param q_t:
        :return:
        """

        if s_t is None:
            raise ValueError("Missing input storage at time t")
        if s_t < 0.:
            raise Warning("Input storage at time t is negative")
        if p_t is None:
            raise ValueError("Missing input precipitation at time t")
        if p_t < 0.:
            raise Warning("Input precipitation at time t is negative")

        s_next = self._alpha * s_t + self._beta * p_t
        q_next = self._gamma * s_next

        return s_next, q_next

    def run(self, nb_iter=100, p_in=None):
        """
        :param nb_iter:
        :param p_in:
        :return:
        """

        # Check inputs
        if nb_iter<=0:
            raise ValueError(f"Invalid nb_iter input, should be > 0, got {nb_iter}")
        if not isinstance(p_in, np.ndarray):
            raise TypeError("Input precipitation p_in must be a np.ndarray")
        if p_in.ndim != 1:
            raise ValueError("Input precipitation p_in must be of 1D")
        if p_in.size < nb_iter:
            raise IndexError("Input precipitation is shorter than the input number of iteration to perform")

        # Set storage and discharge timeseries arrays
        s_out = np.zeros((nb_iter+1,))
        q_out = np.zeros((nb_iter+1,))

        # Initiate state vectors
        s_out[0] = self._s_0
        q_out[0] = self._s_0 * self._gamma

        # Run model
        for t in range(nb_iter):
            s_out[t+1] = self._alpha * s_out[t] + self._beta * p_in[t]
            q_out[t+1] = self._gamma * s_out[t+1]


        return s_out, q_out

    @staticmethod
    def plot_freerun(p_in, s_out, q_out):
        """
        :param p_in:
        :param s_out:
        :param q_out:
        :return:
        """

        fig, axis = plt.subplots(3,1)
        fig.suptitle("Reservoir model")

        axis[0].plot(p_in, "-b", color=(0., 0., 1.))
        axis[0].yaxis.set_inverted(True)
        axis[0].set_title("Inflow")
        axis[0].set_ylabel("Precipitation P")

        axis[1].plot(s_out, "-b", color=(0.,0.,0.75))
        axis[1].set_title("State")
        axis[1].set_ylabel("Storage S")

        axis[2].plot(q_out, "-b", color=(0.,0.25,1.))
        axis[2].set_title("Diagnostic")
        axis[2].set_ylabel("Discharge Q")
        axis[2].set_xlabel("Time iteration")

        return fig, axis

if __name__ == "__main__":
    """Main run for debug
    """

    p_in = np.zeros((101,))
    p_in[:30] = np.array([0., 2., 5., 6., 9., 10.5, 10., 5., 6.5, 6.,
                          10., 12., 18., 11., 10., 9., 8., 0., 0., 9.,
                          11., 10., 9., 8., 7., 3., 2., 7., 8., 4.])
    my_model = ReservoirModel()
    s_out, q_out = my_model.run(nb_iter=100, p_in=p_in)
    fig, axis = my_model.plot_freerun(p_in=p_in, s_out=s_out, q_out=q_out)
    plt.draw()
    plt.show()






        

