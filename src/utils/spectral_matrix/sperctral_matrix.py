import numpy as np
from scipy.signal.windows import dpss
from scipy.linalg import svd
from scipy.fft import next_fast_len

class CrossSpectralMatrix:
    def __init__(self, params):
        self.params = params
        self.get_params()

    def get_params(self):
        self.tapers = self.params.get('tapers', [3, 5])
        self.pad = self.params.get('pad', 0)
        self.fs = self.params.get('Fs', 1)
        self.fpass = self.params.get('fpass', [0, self.fs / 2])

    def dpsschk(self, tapers, n, fs):
        """
            Check or calculate DPSS (Discrete Prolate Spheroidal Sequences) tapers.

            Parameters:
            tapers: Either precalculated tapers or a tuple/list with (NW, K) values for DPSS calculation.
            N: Number of samples in the data.
            Fs: Sampling frequency.

            Returns:
            tapers: Calculated or verified DPSS tapers.
            """
        if isinstance(tapers, (list, tuple)) and len(tapers) == 2:
            # DPSS calculation based on NW and K
            NW, Kmax = tapers
            tapers = dpss(n, NW, Kmax=Kmax, norm=2) * np.sqrt(fs)
        elif hasattr(tapers, "shape") and tapers.shape[0] != n:
            # If tapers are precalculated, check if their length matches the data length
            raise ValueError("Tapers length mismatch with data length")

        return tapers

    def mtfftc(self, data, tapers, nfft, fs):
        if data.ndim == 1:
            data = data[:, np.newaxis]
        nc, c = data.shape
        nk, k = tapers.shape
        tapers = tapers[:, :, np.newaxis].repeat(c, axis=2)
        data = data[:, np.newaxis, :].repeat(k, axis=1)
        data_proj = data * tapers
        j = np.fft.fft(data_proj, n=nfft, axis=0) / fs
        return j

    def compute(self, data, win):
        n, c = data.shape[:2]
        nfft = max(2 ** (next_fast_len(int(win * self.fs)) + self.pad), int(win * self.fs))
        self.tapers = self.dpsschk(self.tapers, int(win * self.fs), self.fs)
        nwins = n // int(win * self.fs)
        sc = np.zeros((nfft // 2 + 1, c, c), dtype=np.complex64)

        for iwin in range(nwins):
            data_segment = data[iwin * int(win * self.fs): (iwin + 1) * int(win * self.fs), :]
            j = self.mtfftc(data_segment, self.tapers, nfft, self.fs)
            for k in range(c):
                for l in range(c):
                    spec = np.mean(np.conj(j[:, k, :]) * j[:, l, :], axis=1)
                    sc[:, k, l] += spec

        sc /= nwins
        cmat = self.compute_cmat(sc)
        ctot, cent, cvec = self.compute_coherence(sc)
        f = np.linspace(0, self.fs / 2, nfft // 2 + 1)

        return sc, cmat, ctot, cvec, cent, f

    def compute_cmat(self, sc):
        c = sc.shape[1]
        cmat = np.copy(sc)
        sdiag = np.zeros((sc.shape[0], c))
        for k in range(c):
            sdiag[:, k] = sc[:, k, k]
        for k in range(c):
            for l in range(c):
                cmat[:, k, l] = sc[:, k, l] / np.sqrt(np.abs(sdiag[:, k] * sdiag[:, l]))
        return cmat

    def compute_coherence(self, sc):
        ctot = np.zeros(sc.shape[0])
        cent = np.zeros(sc.shape[0])
        cvec = np.zeros((sc.shape[0], sc.shape[1]))
        for i in range(sc.shape[0]):
            u, s, _ = svd(sc[i, :, :])
            s = np.diag(s)
            ctot[i] = s[0] / np.sum(s)
            cent[i] = np.exp(np.mean(np.log(s))) / np.mean(s)
            cvec[i, :] = np.transpose(u[:, 0])
        return ctot, cent, cvec

if __name__ == "__main__":
    # Example usage
    params = {
        'tapers': [3, 5],
        'pad': 0,
        'Fs': 1,
        'fpass': [0, 0.5]
    }
    cs_matrix = CrossSpectralMatrix(params)
    data = np.random.randn(100, 3)  # Example data with 100 samples and 2 channels
    win = 10  # Window size in samples

    # Compute the cross-spectral matrix
    sc, cmat, ctot, cvec, cent, f = cs_matrix.compute(data, win)
    sc, cmat, ctot, cvec, cent, f
