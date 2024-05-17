import numpy as np
from mne.time_frequency import csd_array_multitaper


class SpectralMatrixFeatures():
    def __init__(self, dataset):
        self.freqs = None
        self.csd = None
        self.fs = dataset.fs

    def calculate_matrix(self, data, fmin=0, fmax=np.Inf, tmin=None, tmax=None, ch_names=None,
                         n_fft=None, bandwidth=None, adaptive=False, low_bias=True, projs=None, n_jobs=None,
                         verbose=None, desired_freqs=[4, 8, 13, 30]):
        csd = csd_array_multitaper(data, self.fs,
                                   fmin=fmin, fmax=fmax,
                                   ch_names=ch_names,
                                   n_fft=n_fft, bandwidth=bandwidth, verbose=verbose)
        freqs = csd.frequencies
        self.freqs = [freqs[np.argmin(np.abs(f - freqs))] for f in desired_freqs]
        self.csd = np.stack([csd.get_data(f) for f in self.freqs], axis=-1)

    def coherency_matrix(self):
        if self.csd is not None:
            psd = np.real(np.diagonal(self.csd, axis1=0, axis2=1)).transpose()
            norm_factor = np.sqrt(psd[:, np.newaxis, :] * psd[np.newaxis, :, :])
            coh_matrix = np.abs(self.csd) / norm_factor
            coh_tot, coh_ent, coh_vec = self.calculate_svd_metrics(coh_matrix)
            return coh_matrix, coh_tot, coh_ent, coh_vec
        else:
            raise ValueError("Spectral is not calculated")

    def imaginary_coherence(self):
        psd = np.diagonal(self.csd, axis1=0, axis2=1)
        norm_factor = np.sqrt(psd[:, :, np.newaxis] * psd[np.newaxis, :, :])
        return np.imag(self.csd) / norm_factor

    def phase_locking_value(self):
        return np.abs(np.mean(self.csd / np.abs(self.csd), axis=2))

    def corrected_imaginary_plv(self):
        normalized_csd = self.csd / np.abs(self.csd)
        imag_part = np.imag(normalized_csd)
        real_part = np.real(normalized_csd)
        return np.abs(np.mean(imag_part, axis=2)) / np.sqrt(1 - np.mean(real_part, axis=2) ** 2)

    def phase_lag_index(self):
        return np.abs(np.mean(np.sign(np.imag(self.csd)), axis=2))

    def directed_phase_lag_index(self):
        return np.mean(np.heaviside(np.imag(self.csd), 0), axis=2)

    def weighted_phase_lag_index(self):
        imag_csd = np.imag(self.csd)
        return np.abs(np.mean(imag_csd, axis=2)) / np.mean(np.abs(imag_csd), axis=2)

    def calculate_svd_metrics(self, sc_data):
        """
        Calculate metrics based on the singular value decomposition of spectral connectivity data.

        Args:
            spectral_conn: Spectral connectivity data from MNE (output of spectral_connectivity_epochs).

        Returns:
            Ctot: Ratio of the first singular value to the sum of all singular values for each frequency.
            Cent: Entropy measure for each frequency.
            Cvec: First singular vector for each frequency.
        """

        n_freqs = sc_data.shape[-1]
        Ctot = np.zeros(n_freqs)
        Cent = np.zeros(n_freqs)
        Cvec = np.zeros((n_freqs, sc_data.shape[1]))  # Assuming n_con is the same as the second dimension of sc_data

        for i in range(n_freqs):
            u, s, vh = np.linalg.svd(sc_data[:, :, i])
            s = np.diag(s)
            Ctot[i] = s[0, 0] / np.sum(s)
            Cent[i] = np.exp(np.mean(np.log(np.diag(s)))) / np.mean(np.diag(s))
            Cvec[i, :] = u[:, 0]

        return Ctot, Cent, Cvec
