"""
bayes_estimator.py: to perform a data assimilation experiment
"""

import numpy as np
from PIL.TiffTags import SIGNED_BYTE
from matplotlib import pyplot as plt

from constants import N_SIZE_NETWORK, VEC_TOBS, VEC_T, SIG_O_Q, SIG_O_H, T_OBS_STEP, Q_T0, SIG_B_Q, QIN_TS, K_PRIOR
from river_model import RiverModel


class DAExperiment():
    """A class to perform Bayes-estimator-based data assimilation experiment
    """

    def __init__(self, forward_model=None, vec_t_model=VEC_T, vec_t_obs=VEC_TOBS, t_obs_step=T_OBS_STEP, dct_obs=None, dct_ctl=None):
        """Class constructor
        """

        # Check input - timeline compatibility

        self.model = forward_model
        self.vec_t_model = vec_t_model
        self.dct_ctl = dct_ctl

        self.dct_obs = dct_obs
        self.vec_t_obs = vec_t_obs
        self.t_obs_step = t_obs_step

        self._sig_o = None
        self._sig_b = None

    @property
    def sig_o(self):
        return self._sig_o

    @sig_o.setter
    def sig_o(self, in_sig_o):
        self._sig_o = in_sig_o

    @property
    def sig_b(self):
        return self._sig_b

    @sig_b.setter
    def sig_b(self, in_sig_b):
        self._sig_b = in_sig_b

    def plot_model_vs_obs(self, title="Model vs observations"):
        """
        """

        q_out_ts = self.model.run()
        h_out_ts = self.model.estimate_height(q_out_ts)
        flt_max_q = np.amax(q_out_ts)*1.1
        flt_max_h = np.amax(h_out_ts)*1.1

        fig, axis = plt.subplots(3, 3, figsize=(12, 9))
        l_filled_positions = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        fig.suptitle(title)

        for k, (i, j) in enumerate(l_filled_positions):
            ax = axis[i, j]
            ax.set_title(f"Reach {k + 1}")

            ax.plot(q_out_ts[k, :], "-b", label="Q free run")
            ax.set_ylabel("discharge")
            ax.set_ylim((0., flt_max_q))

            ax_bis = ax.twinx()
            ax_bis.plot(h_out_ts[k, :], "-b", color=(0.75, 0., 0.75), label="H free run")

            ax_bis.set_ylabel("height")
            ax_bis.set_ylim((0., flt_max_h))

            for e in self.dct_obs["reach"]:
                row_obs = e - 1
                if row_obs < 0:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")
                if row_obs > self.model.n_dim:
                    raise ValueError(
                        f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")
                if row_obs == k:
                    if self.dct_obs["variable"] == "q":
                        ax.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', markeredgecolor=(0., 0.0, 1.), label="Q obs")
                    elif self.dct_obs["variable"] == "h":
                        ax_bis.plot(VEC_TOBS, self.dct_obs["yobs"], '. g', markeredgecolor=(0.5, 0.0, 0.5), label="H obs")

                    else:
                        raise ValueError("Unknown observation variable, must be 'h' or 'q'")

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax_bis.get_legend_handles_labels()

            handles = handles1 + handles2
            labels = labels1 + labels2
            ax.legend(handles, labels, loc='lower right', fontsize=8)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)

        for i in range(3):
            for j in range(3):
                if (i, j) not in l_filled_positions:
                    ax = axis[i, j]
                    ax.set_visible(False)
        plt.draw()
        plt.show()

    def _perform_state_assim(self, qe=QIN_TS, q_t0_in=Q_T0, b_in=None):
        """
        """

        # Free run
        q_bck_out_ts = self.model.run()
        # Analysis run
        q_ana_out_ts = np.zeros_like(q_bck_out_ts)

        # Bayes estimator parameters
        B = self.sig_b ** 2 * np.eye(self.model.n_dim)
        if b_in is not None:
            B = b_in
        R = self.sig_o ** 2

        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0,e-1] = 1
        K = B @ np.transpose(H) / (H @ B @ np.transpose(H) + R)

        # Assimilation
        q_t0 = q_t0_in
        for assim_iter in range(self.vec_t_obs.size):

            # Propagation
            q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
            q_run = self.model.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            q_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step+1] = q_run

            # Analysis
            xb = q_run[:,-1]
            d = self.dct_obs["yobs"][assim_iter] - H @ xb
            xa = xb + K @ d

            # Cycling
            q_t0 = xa

        return q_bck_out_ts, q_ana_out_ts

    def _perform_diag_assim(self, qe=QIN_TS, q_t0_in=Q_T0, b_in=None):
        """
        :param qe:
        :param q_t0_in:
        :return:
        """

        # Free run
        q_bck_out_ts = self.model.run()
        h_bck_out_ts = self.model.estimate_height(q_bck_out_ts)

        # Analysis run
        q_ana_out_ts = np.zeros_like(q_bck_out_ts)
        h_ana_out_ts = np.zeros_like(q_bck_out_ts)

        # Bayes estimator parameters
        B = self.sig_b ** 2 * np.eye(self.model.n_dim)
        if b_in is not None:
            B = b_in
        R = self.sig_o ** 2

        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0, e - 1] = 1
        K = B @ np.transpose(H) / (H @ B @ np.transpose(H) + R)

        # Assimilation
        q_t0 = q_t0_in
        for assim_iter in range(self.vec_t_obs.size):
            # Propagation
            q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
            q_run = self.model.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            h_run = self.model.estimate_height(q_run)

            q_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1] = q_run
            h_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1] = h_run

            # Analysis
            xb = h_run[:, -1]
            d = self.dct_obs["yobs"][assim_iter] - H @ xb
            xa = xb + K @ d

            # Cycling
            qa = self.model.rating_curve(xa)
            q_t0 = qa

        return h_bck_out_ts, h_ana_out_ts

    def _perform_param_assim(self, qe=QIN_TS, q_t0_in=Q_T0):
        """
        :param qe:
        :param q_t0_in:
        :return:
        """

        # Free run
        q_bck_out_ts = self.model.run()
        # Analysis run
        q_ana_out_ts = np.zeros_like(q_bck_out_ts)

        # Bayes estimator parameters
        B = self.sig_b ** 2
        R = self.sig_o ** 2
        H = np.zeros((1, self.model.n_dim))
        for e in self.dct_obs["reach"]:
            H[0,e-1] = 1
        K = (B / (B + R))

        # Assimilation
        xb = np.zeros((VEC_TOBS.size,))
        xb[0] = K_PRIOR
        xa = np.zeros((VEC_TOBS.size,))

        # Assimilation
        q_t0 = q_t0_in
        model_assim = self.model.copy()
        for assim_iter in range(self.vec_t_obs.size):

            # Propagation
            q_in = qe[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step]
            q_run = model_assim.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)

            # Analysis
            d = self.dct_obs["yobs"][assim_iter] - H @ q_run[:, -1]
            xa[assim_iter] = xb[assim_iter] + K * d[0]

            if xa[assim_iter] < 0:
                print("Warning: negative parameter vlue after assimilation")

            # Cycling
            model_assim = RiverModel(par_k_in=xa[assim_iter])
            q_run = model_assim.run(n_iter=self.t_obs_step,
                                   vec_q_0=q_t0,
                                   mat_q_in_ts=q_in)
            q_ana_out_ts[:, assim_iter * self.t_obs_step: assim_iter * self.t_obs_step + self.t_obs_step + 1] = q_run
            q_t0 = q_run[:, -1]
            try:
                xb[assim_iter+1] = xa[assim_iter]
            except IndexError:
                pass

        # Final analysis run
        model_analysis = RiverModel(par_k_in=xa[-1])
        q_final = model_analysis.run()

        return q_bck_out_ts, q_ana_out_ts, q_final

    def perform_assim(self, b_in=None):
        """
        """

        if self.dct_ctl["exp"] == "state":
            q_bck_out_ts, q_ana_out_ts = self._perform_state_assim(b_in=b_in)
            self.plot_assim_state(q_bck_out_ts, q_ana_out_ts)
        elif self.dct_ctl["exp"] == "parameter":
            q_bck_out_ts, q_ana_out_ts, q_final = self._perform_param_assim()
            self.plot_assim_param(q_bck_out_ts, q_ana_out_ts, q_final)
        elif self.dct_ctl["exp"] == "diagnostic":
            h_bck_out_ts, h_ana_out_ts = self._perform_diag_assim(b_in=b_in)
            self.plot_assim_diag(h_bck_out_ts, h_ana_out_ts)

        else:
            raise ValueError("Unknown experiment type")

    def plot_assim_state(self, q_bck_out_ts, q_ana_out_ts, title="Assimilation results - Discharge"):
        """
        """

        flt_max_q = max(np.amax(q_bck_out_ts)*1.1, np.amax(q_ana_out_ts)*1.1)

        fig, axis = plt.subplots(self.model.n_dim, 1)
        fig.suptitle(title)

        axis[self.model.n_dim - 1].set_title("Time iteration")

        for il_i in range(self.model.n_dim):
            axis[il_i].plot(q_bck_out_ts[il_i, :], "-b")
            axis[il_i].plot(q_ana_out_ts[il_i, :], "-r")
            axis[il_i].set_ylabel(f"Reach {il_i + 1}")

        for e in self.dct_obs["reach"]:
            row_obs = e - 1
            if row_obs < 0:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")
            if row_obs > self.model.n_dim:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")

            axis[row_obs].plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="observations")
            axis[row_obs].plot(q_bck_out_ts[row_obs, :], '-b', label="free run")
            axis[row_obs].plot(q_ana_out_ts[row_obs, :], '-r', label="analysis run")

        plt.draw()
        plt.show()

    def plot_assim_diag(self, h_bck_out_ts, h_ana_out_ts, title="Assimilation results - Height"):
        """
        """

        flt_max_h = max(np.amax(h_bck_out_ts)*1.1, np.amax(h_ana_out_ts)*1.1)

        fig, axis = plt.subplots(self.model.n_dim, 1)
        fig.suptitle(title)

        axis[self.model.n_dim - 1].set_title("Time iteration")

        for il_i in range(self.model.n_dim):
            axis[il_i].plot(h_bck_out_ts[il_i, :], "-b")
            axis[il_i].plot(h_ana_out_ts[il_i, :], "-r")
            axis[il_i].set_ylabel(f"Reach {il_i + 1}")

        for e in self.dct_obs["reach"]:
            row_obs = e - 1
            if row_obs < 0:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")
            if row_obs > self.model.n_dim:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")

            axis[row_obs].plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="observations")
            axis[row_obs].plot(h_bck_out_ts[row_obs, :], '-b', label="free run")
            axis[row_obs].plot(h_ana_out_ts[row_obs, :], '-r', label="analysis run")

        plt.draw()
        plt.show()

    def plot_assim_param(self, q_bck_out_ts, q_ana_out_ts, q_final, title="Assimilation results - Discharge"):
        """
        """

        fig, axis = plt.subplots(self.model.n_dim, 1)
        fig.suptitle(title)

        axis[self.model.n_dim - 1].set_xlabel("Time iteration")

        for il_i in range(self.model.n_dim):
            axis[il_i].plot(q_bck_out_ts[il_i, :], "-b")
            axis[il_i].plot(q_ana_out_ts[il_i, :], '--r', color=(1.0, 0.5, 0.))
            axis[il_i].plot(q_final[il_i, :], "-r")
            axis[il_i].set_ylabel(f"Reach {il_i + 1}")

        for e in self.dct_obs["reach"]:
            row_obs = e - 1
            if row_obs < 0:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")
            if row_obs > self.model.n_dim:
                raise ValueError(
                    f"Invalid reach id, must be between 1 and {self.model.n_dim}, got {self.dct_obs["reach"]}.")

            axis[row_obs].plot(VEC_TOBS, self.dct_obs["yobs"], '. g', label="observations")
            axis[row_obs].plot(q_bck_out_ts[row_obs, :], '-b', label="free run")
            axis[row_obs].plot(q_ana_out_ts[row_obs, :], '--r', label="sequential analysis", color=(1.0, 0.5, 0.))
            axis[row_obs].plot(q_final[row_obs, :], "-r", label="analysis")
            axis[row_obs].legend()

        plt.draw()
        plt.show()

if __name__ == "__main__":
    """Main run
    """

    free_run = RiverModel()
    dct_obs_1 = {
        "yobs": [28.031, 31.354, 28.918, 26.057, 25.365, 26.536, 29.751, 30.046, 26.028, 26.167],
        "variable": "q",
        "reach": [5]
    }
    dct_obs_2 = {
        "yobs": [2.345, 2.594, 2.340, 2.028, 1.921, 2.086, 2.391, 2.312, 2.029, 1.949],
        "variable": "h",
        "reach": [5]
    }
    dct_obs_3 = {
        "yobs": [21.910, 26.992, 24.554, 19.885, 17.863, 19.427, 26.036, 23.850, 20.655, 19.492],
        "variable": "q",
        "reach": [3]
    }

    my_assim = DAExperiment(forward_model=free_run,
                            dct_obs=dct_obs_3)
    my_assim.plot_model_vs_obs()

    # dct_ctl = { "exp": "state" }
    # my_assim = DAExperiment(forward_model=free_run,
    #                                 dct_obs=dct_obs_1,
    #                                 dct_ctl=dct_ctl)
    # my_assim.sig_o = SIG_O_Q
    # my_assim.sig_b = 0.5
    # B_in = np.array([
    #     [SIG_B_Q**2., 0., 0., 0., SIG_B_Q**2.*0.0625],
    #     [0., SIG_B_Q**2., 0., 0., SIG_B_Q**2.*0.125],
    #     [0., 0., SIG_B_Q**2., 0., SIG_B_Q**2.*0.25],
    #     [0., 0., 0., SIG_B_Q**2., SIG_B_Q**2.*0.5],
    #     [SIG_B_Q**2.*0.0625, SIG_B_Q**2.*0.125, SIG_B_Q**2.*0.25, SIG_B_Q**2.*0.5, SIG_B_Q**2.]
    # ])
    # my_assim.perform_assim(b_in=B_in)
