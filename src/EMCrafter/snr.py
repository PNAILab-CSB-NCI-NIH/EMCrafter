import numpy as np
import matplotlib.pyplot as plt

from EMCrafter.base import Base

FONTSIZE = 12
TICKSIZE = 10
LEGSIZE = 10

class SNR(Base):
    def __init__(self, snr=0.05, verbose=1):
        """
        Initialize the SNR object.

        Parameters
        ----------
        snr : float
            Default SNR value for a flat distribution. Default is 0.05.
        verbose : int
            Verbosity level. Default is 1.

        Returns
        -------
        None
        """
        super().__init__(verbose)
        self.empty()
        self.set_flat(snr)

    def empty(self):
        """
        Empty the SNR object.

        Sets all variables to None and removes any assigned distribution.
        """
        self.bc = None
        self.be = None
        self.bw = None
        self.snr = None
        self.type = None
        self.transform = None
    
    def set_flat(self, snr=0.05):
        """
        Set a flat SNR distribution.

        Parameters
        ----------
        snr : float
            Value of the SNR for the flat distribution. Default is 0.05.

        Returns
        -------
        None
        """
        self.snr = snr
        self.type = "flat"
        self.value = self._value_flat

    def set_histo(self, defocus_edges, snr, unit="um", emp_factor=2.):
        """
        Set the SNR as a histogram from a given array of defocus bin edges and SNR values.

        Parameters
        ----------
        defocus_edges : array
            The bin edges of the histogram in microns or Angstroms.
        snr : array
            The SNR values corresponding to the bin edges.
        unit : str, optional
            The unit of the defocus bin edges. Default is "um".
        emp_factor : float, optional
            The empirical factor to apply to the SNR values. Default is 2.

        Returns
        -------
        None
        """
        if self.v > 2: self.logger.info("Setting SNR to histogram")
        if unit not in ["um", "A"]:
            raise ValueError("Unit must be 'um' or 'A'")
        self.be = np.array(defocus_edges)
        if unit == "A": self.be /= 1e4
        self.bw = self.be[1:] - self.be[:-1]
        self.bc = (self.be[1:] + self.be[:-1])/2
        self.snr = np.array(snr)*emp_factor
        self.type = "histo"
        self.value = self._value_histo
    
    def plot(self, show=True, ax=None):
        """
        Plot the SNR distribution.

        Parameters
        ----------
        show : bool, optional
            If True, shows the plot. Default is True.
        ax : matplotlib axes, optional
            Axes to plot on. Default is None.

        Returns
        -------
        fig : matplotlib figure
            The figure object.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2))
        
        if self.type == "flat":
            ax.plot([0, 1], [self.snr, self.snr], linewidth=2., color="blue", label="Flat SNR")
            plt.xticks([])
        else:
            ax.errorbar(self.bc, self.snr, xerr=self.bw/2, marker="o", linewidth=2., linestyle="--", color="blue", label="SNR")

        ax.set_ylabel("SNR (a.u.)", fontsize=FONTSIZE)
        ax.set_xlabel(r"Defocus ($\mu m$)", fontsize=FONTSIZE)

        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
    def set_value_func(self, interpolate=True):
        """
        Set the SNR value function.

        Parameters
        ----------
        func : callable
            The function to use to calculate the SNR value.

        Returns
        -------
        None
        """
        if self.type == "flat":
            func = self._value_flat
        elif self.type == "histo":
            if interpolate:
                func = self._value_interp
            else: func = self._value_histo
        self.value = func
    
    def _value_histo(self, defocus):
        """
        Looks up the SNR value for a given defocus in the histogram.

        Parameters
        ----------
        defocus : float or array-like
            The defocus value(s) in micrometers for which to look up the SNR.

        Returns
        -------
        float or np.ndarray
            The SNR value(s) corresponding to the input defocus value(s).
        """
        # Find bin indices
        defocus = np.asarray(defocus)
        bin_indices = np.searchsorted(self.be, defocus, side="right") - 1
        
        # Clamp indices to valid range [0, len(snr)-1]
        bin_indices = np.clip(bin_indices, 0, len(self.snr) - 1)

        return self.snr[bin_indices]
    
    def _value_interp(self, defocus):
        """
        Interpolates the SNR value for a given defocus using a histogram-based approach.

        Parameters
        ----------
        defocus : float or array-like
            The defocus value(s) in micrometers for which to interpolate the SNR.

        Returns
        -------
        float or np.ndarray
            The interpolated SNR value(s) corresponding to the input defocus value(s).
        """
        return np.interp(defocus, self.bc, self.snr)
    
    def _value_flat(self, _):
        """
        Returns the SNR value for a flat distribution.

        Parameters
        ----------
        _ : ignored
            This parameter is ignored.

        Returns
        -------
        float
            The SNR value.
        """
        return self.snr
