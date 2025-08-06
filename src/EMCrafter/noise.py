import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from scipy.special import erfc
from scipy.interpolate import interp1d
from functools import partial

from EMCrafter.base import Base
from EMCrafter.sampler import Sampler
from EMCrafter.rln import rlnParticles

FONTSIZE = 12
TICKSIZE = 10
LEGSIZE = 10

# Gaussian PDF
def gaussian(x, amplitude, x0, sigma):
    """
    Calculate the Gaussian probability density function.

    Parameters
    ----------
    x : array_like
        Input values for which to calculate the Gaussian function.
    amplitude : float
        Amplitude of the Gaussian function.
    x0 : float
        Mean or center of the Gaussian function.
    sigma : float
        Standard deviation of the Gaussian function.

    Returns
    -------
    array_like
        Evaluated Gaussian function at input values x.
    """

    return amplitude*np.exp(-(x - x0)**2/(2*sigma**2))

# Skew-normal PDF
def skew_normal(x, amplitude, loc, scale, alpha):
    """
    Calculate the Skew-Normal probability density function.

    Parameters
    ----------
    x : array_like
        Input values for which to calculate the Skew-Normal function.
    amplitude : float
        Amplitude of the Skew-Normal function.
    loc : float
        Location parameter of the Skew-Normal function.
    scale : float
        Scale parameter of the Skew-Normal function.
    alpha : float
        Skewness parameter of the Skew-Normal function.

    Returns
    -------
    array_like
        Evaluated Skew-Normal function at input values x.
    """

    return amplitude*skewnorm.pdf(x, alpha, loc=loc, scale=scale)

# Exponentially Modified Gaussian PDF
def exp_mod_gaussian(x, amplitude, mu, sigma, lambd):
    """
    Calculate the Exponentially Modified Gaussian probability density function.

    Parameters
    ----------
    x : array_like
        Input values for which to calculate the Exponentially Modified Gaussian function.
    amplitude : float
        Amplitude of the Exponentially Modified Gaussian function.
    mu : float
        Center of the Exponentially Modified Gaussian function.
    sigma : float
        Standard deviation of the Exponentially Modified Gaussian function.
    lambd : float
        Exponential rate of the Exponentially Modified Gaussian function.

    Returns
    -------
    array_like
        Evaluated Exponentially Modified Gaussian function at input values x.
    """

    A, s, k = amplitude, sigma, lambd
    return A*(k/2)*np.exp(k/2*(2*mu + k*s**2 - 2*x))*erfc((mu + k*s**2 - x)/(np.sqrt(2)*s))

FIT_FUNCTIONS = {
    "gaussian": {
        "name": "Gaussian",
        "label": "Normal",
        "fn": gaussian,
        "p0": [1., 0., 1.],
        "leg_label": r"$N(\mu,\sigma^2)$"},
    "skew_normal": {
        "name": "Skew-Normal",
        "label": "Skew",
        "fn": skew_normal,
        "p0": [1., 0.85, 1.3, -1.4],
        "leg_label": r"$SN(\xi,\omega^2,\alpha)$"},
    "exp_mod_gaussian": {
        "name": "Exp. Mod. Gaussian",
        "label": "EMG",
        "fn": exp_mod_gaussian,
        "p0": [-1., 0.6, -0.84, -1.64],
        "leg_label": r"$EMG(\mu,\sigma^2,\lambda)$"}
}

class NoiseFit(Base):
    def __init__(self, file=None, verbose=1):
        """
        Initialize the NoiseFit class.

        Parameters
        ----------
        file : str or None, optional
            The path to a STAR file containing the particle data.
            If None, an empty object is created.
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.
        """
        super().__init__(verbose)
        self.file = None
        self.image_shape = None
        if file is not None:
            self.read_particles(file)
    
    @property
    def data(self):
        """
        The data stored in the associated STAR file.

        Returns
        -------
        pandas.DataFrame or None
            The data stored in the associated STAR file, or None if no file is associated.
        """
        
        if self.file is None:
            return None
        return self.file.data
    
    @data.setter
    def data(self, value):
        """
        Set the data stored in the associated STAR file.

        Parameters
        ----------
        value : pandas.DataFrame
            The data to store in the associated STAR file.
        """
        self.file.data = value
    
    @property
    def images(self):
        """
        The particle images stored in the associated STAR file.

        Returns
        -------
        list or None
            The particle images stored in the associated STAR file, or None if no file is associated.
        """
        if self.file is None:
            return None
        return self.file.images
    
    @images.setter
    def images(self, value):
        """
        Set the particle images stored in the associated STAR file.

        Parameters
        ----------
        value : list
            The particle images to store in the associated STAR file.
        """

        self.file.images = value
    
    def read_particles(self, file, n=None, ids=None):
        """
        Read a STAR file and select particles.

        Parameters
        ----------
        file : str
            The path to the STAR file containing the particle data.
        n : int, optional
            The maximum number of particles to select.
            If None, all particles are selected. Defaults to None.
        ids : list of int, optional
            The indices of the particles to select.
            If None, all particles are selected. Defaults to None.
        """
        self.file = rlnParticles(file)
        if ids is not None:
            self.data = self.data.iloc[ids]
        if n is not None:
            self.data = self.data.iloc[:n]
    
    def load_images(self, base, n=None):
        """
        Load the particle images from the STAR file.

        Parameters
        ----------
        base : str
            The base folder containing the particle images.
        n : int, optional
            The maximum number of images to load. If None, all images are loaded.
            Defaults to None.
        """
        self.file.load_images(base, n=n)
        self.image_shape = self.images[0]["data"].shape
    
    def get_noise_distribution(self, mask=None, irange=(-8, 6), nbins=100):
        """
        Compute the noise distribution from the particle images.

        Parameters
        ----------
        mask : ndarray, optional
            A boolean mask with the same shape as the images.
            If None, all pixels are used. Defaults to None.
        irange : tuple, optional
            The range of values to include in the histogram.
            Defaults to (-8, 6).
        nbins : int, optional
            The number of bins to use in the histogram.
            Defaults to 100.
        """
        if self.v > 1: self.logger.info("Extracting noise distribution")
        self.irange = irange

        bmask = np.ones(self.image_shape, dtype=bool)
        if mask is not None:
            bmask = mask > 0.5
        
        values = []
        for i in range(len(self.images)):
            hist, bin_edges = np.histogram(self.images[i]["data"][bmask].flatten(), range=irange, bins=nbins, density=True)
            values.append(hist)
        bin_centers = (bin_edges[1:] + bin_edges[0:-1])/2
        bin_width = (bin_edges[1:] - bin_edges[0:-1])/2

        counts = np.mean(np.array(values), axis=0)
        yerr = np.std(np.array(values), axis=0)
        total = np.sum(counts*bin_width*2)

        self.be = bin_edges
        self.bc = bin_centers
        self.bw = bin_width
        self.noise_dist = counts/total
        self.noise_std = yerr/total
        self.mask = self.noise_dist > 0
    
    def plot_noise_distribution(self, dmin=1e-10, dmax=0.8, show=True):
        """
        Plot the noise distribution.

        Parameters
        ----------
        dmin : float, optional
            The minimum data value that the y-axis covers. Defaults to 1e-10.
        dmax : float, optional
            The maximum data value that the y-axis covers. Defaults to 0.8.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : Figure
            The matplotlib figure object containing the plot.
        """
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.errorbar(self.bc, self.noise_dist, xerr=self.bw, yerr=self.noise_std)
        ax.set_yscale("log")

        ax.set_xlabel("Intensity (a.u.)", fontsize=FONTSIZE)
        ax.set_ylabel("Density (a.u.)", fontsize=FONTSIZE)
        ax.set_ylim([dmin, dmax])
        ax.set_xlim(self.irange)

        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
    def chi2(self, y, yerr, yh):
        """
        Calculate chi-squared between the data and a model.

        Parameters
        ----------
        y : array
            The data
        yerr : array
            The error bars on the data
        yh : array
            The model

        Returns
        -------
        chi_squared : float
            The chi-squared between the data and the model
        """
        chi_squared = np.sum(((y - yh) / yerr) ** 2)
        return chi_squared

    def red_chi2(self, y, yerr, yh, nparams=0):
        """
        Calculate the reduced chi-squared between the data and a model.

        Parameters
        ----------
        y : array
            The data
        yerr : array
            The error bars on the data
        yh : array
            The model
        nparams : int
            The number of parameters in the model

        Returns
        -------
        red_chi2 : float
            The reduced chi-squared between the data and the model
        """
        chi_squared = self.chi2(y, yerr, yh)
        ndof = len(y) - nparams
        return chi_squared/ndof

    def fit(self, pdf="gaussian", p0=None):
        """
        Fit the noise distribution to a given probability density function.

        Parameters
        ----------
        pdf : str
            The type of probability density function to fit. Must be one of the following:
            - gaussian
            - skew_normal
            - exp_mod_gaussian
        p0 : array
            The initial parameters for the fit

        Returns
        -------
        None
        """
        if self.v > 1: self.logger.info(f"Fitting noise distribution using a {pdf} PDF")
        if pdf not in FIT_FUNCTIONS.keys():
            raise ValueError(f"Unknown PDF: {pdf}, must be one of {list(FIT_FUNCTIONS.keys())}")
        
        self.fit_fn = FIT_FUNCTIONS[pdf]["fn"]
        self.fit_name = FIT_FUNCTIONS[pdf]["name"]
        self.fit_leg = FIT_FUNCTIONS[pdf]["label"]

        self.p0 = FIT_FUNCTIONS[pdf]["p0"] if p0 is None else p0
        self.fit_pars, _ = curve_fit(self.fit_fn, self.bc[self.mask], self.noise_dist[self.mask], p0=self.p0)
        self.fit_chi2 = self.red_chi2(
            self.noise_dist[self.mask], self.noise_std[self.mask],
            self.fit_fn(self.bc[self.mask], *self.fit_pars), len(self.fit_pars)
        )

    def plot_fit(self, dmin=1e-10, dmax=0.8, show=True):
        """
        Plot the noise distribution and a fit to it.

        Parameters
        ----------
        dmin : float, optional
            The minimum data value that the y-axis covers. Defaults to 1e-10.
        dmax : float, optional
            The maximum data value that the y-axis covers. Defaults to 0.8.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : Figure
            The matplotlib figure object containing the plot.
        """
        x_fit = np.linspace(self.bc[0], self.bc[-1], 500)
        y_fit = self.fit_fn(x_fit, *self.fit_pars)
        chi2_label = r"$\chi^2_{\mathrm{red}}$"
        leg = rf"{self.fit_leg} {chi2_label} = {self.fit_chi2:.2f}"

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.errorbar(self.bc, self.noise_dist, xerr=self.bw, yerr=self.noise_std, label="Data", linewidth=1.5)
        ax.plot(x_fit, y_fit, color="red", linewidth=2, label=leg)
        ax.set_yscale("log")

        ax.set_xlabel("Intensity (a.u.)", fontsize=FONTSIZE)
        ax.set_ylabel("Density (a.u.)", fontsize=FONTSIZE)
        ax.set_ylim([dmin, dmax])
        ax.set_xlim(self.irange)

        plt.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig

class NoiseShaper(Sampler):
    def __init__(self, seed=None, verbose=1):
        """
        Initialize the NoiseShaper class.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.
        """
        super().__init__(seed, verbose)
        self.fn = None
        self.type = None
        self.samples = np.array([])
        self.set_gaussian_shape()
    
    def build_sampler(self):
        """
        Builds the sampler by computing the Probability Density Function (PDF)
        and Cumulative Distribution Function (CDF) over a specified range.
        The Inverse CDF is constructed for sampling purposes.

        The sampler is built over a range of x values from -10 to 10, with a
        resolution of 10000 points. The PDF is evaluated using the currently
        set distribution function, normalized, and its CDF is computed. The
        Inverse CDF is then created to allow for efficient sampling from the
        distribution.

        Returns
        -------
        None
        """
        x = np.linspace(-10, 10, 10000)
        pdf = self.fn(x)
        pdf /= np.trapz(pdf, x)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        self.total_area = cdf[-1]
        # Inverse CDF
        self.icdf = interp1d(cdf, x, bounds_error=False, fill_value=(x[0], x[-1]))
    
    def set_gaussian_shape(self, amplitude=.4, x0=0., sigma=1.):
        """
        Set the shape of the noise sampler to a Gaussian distribution.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the Gaussian distribution. Defaults to 0.4 (~1/sqrt(2pi)).
        x0 : float, optional
            Mean or center of the Gaussian distribution. Defaults to 0.0.
        sigma : float, optional
            Standard deviation of the Gaussian distribution. Defaults to 1.0.

        Returns
        -------
        None
        """
        if self.v > 1: self.logger.info("Setting shape to Gaussian")
        self.type = "Gaussian"
        self.leg_label = r"$N(\mu,\sigma^2)$"
        self.fn = partial(gaussian, amplitude=amplitude, x0=x0, sigma=sigma)
        self.build_sampler()
    
    def set_skew_normal_shape(self, amplitude=1., loc=0.85, scale=1.3, alpha=-1.4):
        """
        Set the shape of the noise sampler to a Skew-Normal distribution.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the Skew-Normal distribution. Defaults to 1.0.
        loc : float, optional
            Location parameter of the Skew-Normal distribution. Defaults to 0.85.
        scale : float, optional
            Scale parameter of the Skew-Normal distribution. Defaults to 1.3.
        alpha : float, optional
            Skewness parameter of the Skew-Normal distribution. Defaults to -1.4.

        Returns
        -------
        None
        """
        if self.v > 1: self.logger.info("Setting shape to Skew-Normal")
        self.type = "Skew-Normal"
        self.leg_label = r"$SN(\xi,\omega^2,\alpha)$"
        self.fn = partial(skew_normal, amplitude=amplitude, loc=loc, scale=scale, alpha=alpha)
        self.build_sampler()
    
    def set_exp_mod_gaussian_shape(self, amplitude=-1., mu=0.6, sigma=-0.84, lambd=-1.64):
        """
        Set the shape of the noise sampler to an Exponentially Modified Gaussian distribution.

        Parameters
        ----------
        amplitude : float, optional
            Amplitude of the Exponentially Modified Gaussian distribution. Defaults to -1.0.
        mu : float, optional
            Mean of the Gaussian distribution. Defaults to 0.6.
        sigma : float, optional
            Standard deviation of the Gaussian distribution. Defaults to -0.84.
        lambd : float, optional
            Exponential rate of the Exponentially Modified Gaussian distribution. Defaults to -1.64.

        Returns
        -------
        None
        """
        if self.v > 1: self.logger.info("Setting shape to Exp. Mod. Gaussian")
        self.type = "Exp. Mod. Gaussian"
        self.leg_label = r"$EMG(\mu,\sigma^2,\lambda)$"
        self.fn = partial(exp_mod_gaussian, amplitude=amplitude, mu=mu, sigma=sigma, lambd=lambd)
        self.build_sampler()
    
    def sample(self, shape, n=1, store=True):
        """
        Sample the noise distribution.

        Parameters
        ----------
        shape : tuple
            Shape of the output noise samples.
        n : int, optional
            Number of samples to generate. Defaults to 1.
        store : bool, optional
            If True, store the generated samples in the `samples` attribute.
            Defaults to True.

        Returns
        -------
        samples : ndarray
            Array of shape (n, shape[0], shape[1]) containing the generated noise samples.
        """
        if self.v > 2: self.logger.info(f"Sampling {n} noise samples with shape {shape}")
        full_shape = (n,) + shape
        rand = self.random(full_shape)
        samples = self.icdf(rand)
        #means = samples.mean(axis=(1, 2), keepdims=True)
        #stds = samples.std(axis=(1, 2), keepdims=True)
        #samples = (samples - means)/stds

        if store: self.samples = samples
        return samples[0] if n == 1 else samples
    
    def get_index(self, index=0):
        """
        Get the sample at the specified index.

        Parameters
        ----------
        index : int, optional
            Index of the sample to retrieve. Defaults to 0.

        Returns
        -------
        sample : ndarray
            2D array containing the sample at the specified index.

        Raises
        ------
        ValueError
            If there are no samples stored.
        ValueError
            If the index is out of range.
        """
        if len(self.samples) == 0:
            raise ValueError("No samples to plot.")
        if index >= len(self.samples):
            raise ValueError("Index out of range.")
        return self.samples[index]
    
    def plot_image(self, index=0, cmap="gray", vmin=None, vmax=None, show=True):
        """
        Plots a 2D image of a noise sample at the specified index.

        Parameters
        ----------
        index : int, optional
            The index of the sample to plot. Defaults to 0.
        cmap : str, optional
            The colormap to use for the image. Defaults to "gray".
        vmin : float, optional
            The minimum data value that the colormap covers. Defaults to None.
        vmax : float, optional
            The maximum data value that the colormap covers. Defaults to None.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        sample = self.get_index(index)

        fig, ax = plt.subplots(figsize=(4, 3))
        c = ax.imshow(sample, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.yticks([]); plt.xticks([])
        ax.axis('off')

        cbar = plt.colorbar(c, pad=0.01)

        cbar.set_label('Intensity (a.u.)', fontsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
            
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
    def plot_density(self, index=0, bins=100, show=True, ax=None):
        """
        Plot a histogram of the noise sample at the specified index and the corresponding probability density function.

        Parameters
        ----------
        index : int, optional
            The index of the sample to plot. Defaults to 0.
        bins : int, optional
            The number of histogram bins. Defaults to 100.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.
        ax : matplotlib axes, optional
            Axes to plot on. Defaults to None.

        Returns
        -------
        fig : matplotlib figure
            The figure object containing the plot.
        """
        plot_sample = True
        try: sample = self.get_index(index)
        except: plot_sample = False
        x = np.linspace(-10, 10, 10000)
        y = self.fn(x)
        y /= np.trapz(y, x)

        # Create figure
        fig, axi = None, ax
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace':0.05})
            axi = ax[0]

        if plot_sample:
            axi.hist(sample.flatten(), bins=bins, histtype='step', density=True, linewidth=2., color="blue", label="Sampled")
        axi.plot(x, y, linewidth=1.5, linestyle="--", color="red", alpha=0.8, label=rf"{self.leg_label} PDF")
        axi.set_xlabel("Intensity (a.u.)", fontsize=FONTSIZE)
        axi.set_ylabel("Density (a.u.)", fontsize=FONTSIZE)
        axi.legend(loc="upper left", fontsize=LEGSIZE)
        axi.set_ylim([1e-10, .8])
        axi.set_xlim([-8., 6.])

        if fig is not None:
            axi = ax[1]
            if plot_sample:
                axi.hist(sample.flatten(), bins=bins, histtype='step', density=True, linewidth=2., color="blue")
            axi.plot(x, y, linewidth=1.5, linestyle="--", color="red", alpha=0.8)
            axi.set_xlabel("Intensity (a.u.)", fontsize=FONTSIZE)
            axi.set_ylabel("Density (a.u.)", fontsize=FONTSIZE, position="right")
            axi.set_yscale("log")
            axi.set_ylim([1e-10, 1.])
            axi.set_xlim([-8., 6.])

        #plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig


class NoiseModulator(Base):
    def __init__(self, verbose=1):
        """
        Initialize the NoiseModulator class.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Defaults to 1.

        Returns
        -------
        None
        """
        super().__init__(verbose)
        self.white_noise = None
        self.colored_noise = None
        self.apix = None
        self.box_size = None
        self.y = None
        self.y2D = None
        self.type = None
        self.set_flat()
    
    def set_apix(self, apix):
        """
        Set the resolution in Angstroms per pixel.

        Parameters
        ----------
        apix : float
            Resolution in Angstroms per pixel.

        Returns
        -------
        None
        """
        self.apix = apix

    def set_flat(self):
        """
        Set the noise modulation to a flat distribution.

        The modulation is constant and set to 1.0.

        Returns
        -------
        None
        """
        self.type = "flat"
    
    def set_modulation(self, bin_edges, variance, box_size):
        """
        Set the noise modulation to a user-defined histogram.

        Parameters
        ----------
        bin_edges : array
            The bin edges of the histogram.
        variance : array
            The variance values corresponding to the bin edges.
        box_size : int
            The box size of the Fourier transform.

        Returns
        -------
        None
        """
        if self.v > 2: self.logger.info("Setting noise modulation to user-defined histogram")
        self.type = "modulated"
        self.be = bin_edges
        self.bc = (bin_edges[1:] + bin_edges[:-1])/2
        self.bw = self.be[1:] - self.be[:-1]
        self.ibc = self.bc/(box_size*self.apix)
        self.ibw = self.bw/(box_size*self.apix)
        self.y = np.abs(variance)
        self.box_size = box_size
        self.box_nyquist = box_size/2
        self.y2D = self.expand_radial_profile(self.be, self.y, (box_size, box_size))
    
    def plot_modulator_1D(self, show=True, ax=None):
        """
        Plot the 1D modulation profile of the noise.

        Parameters
        ----------
        show : bool, optional
            Whether to show the plot. Default is True.
        ax : Axes, optional
            Axes to plot on. Default is None.

        Returns
        -------
        fig : Figure
            The figure object if show is False, else None.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2))

        ax.set_xlabel(r'Frequency (1/$\AA$)', fontsize=FONTSIZE)
        ax.set_ylabel(r'Variance (a.u.)$^2$', fontsize=FONTSIZE)
        if self.type == "flat":
            ax.plot([0, 1], [1, 1], linewidth=2., color="blue", label="Flat")
            plt.xticks([])
        else:
            ax.errorbar(self.ibc, self.y, xerr=self.ibw/2, linewidth=1.5, linestyle="--", color="red", alpha=0.8, marker="o", markersize=2)
            xticks = plt.xticks()[0]
            x_labels = [r"$\frac{1}{%.1f\ \AA{}}$" % (1/tick) if tick != 0 else "DC" for tick in xticks]
            plt.xticks(xticks, x_labels, fontsize=TICKSIZE)
            ax.set_xlim([-0.05, 1.1*self.box_nyquist/(self.box_size*self.apix)])
        plt.yticks(fontsize=TICKSIZE)
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
    def plot_modulator_2D(self, cmap="hot", show=True):
        """
        Plot the 2D modulation profile of the noise.

        Parameters
        ----------
        cmap : str, optional
            The colormap to use. Default is "hot".
        show : bool, optional
            Whether to show the plot. Default is True.

        Returns
        -------
        fig : Figure
            The figure object if show is False, else None.
        """
        fig, ax = plt.subplots(figsize=(4, 3))
        c = ax.imshow(self.y2D, cmap=cmap)
        plt.yticks([]); plt.xticks([])
        ax.axis('off')

        cbar = plt.colorbar(c, pad=0.01)

        cbar.set_label('Variance (a.u.)$^2$', fontsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
            
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
    def fft(self, img):
        """
        Compute the 2D Fourier transform of an image.

        Parameters
        ----------
        img : array
            The image to transform.

        Returns
        -------
        fft : array
            The 2D Fourier transform of the image.
        """
        fft = np.fft.fft2(img)
        fft = np.fft.fftshift(fft)
        return fft.copy()
    
    def ifft(self, img):
        """
        Compute the inverse 2D Fourier transform of an image.

        This function takes a frequency-domain representation of an image,
        shifts the zero-frequency component back to the corners of the spectrum,
        performs an inverse 2D Fourier transform, and returns the real part of
        the transformed image.

        Parameters
        ----------
        img : array
            The frequency-domain representation of the image.

        Returns
        -------
        array
            The spatial-domain representation of the image.
        """
        img = np.fft.ifftshift(img)
        img = np.fft.ifft2(img).real
        return img.copy()
    
    def magnitude(self, img):
        """
        Compute the magnitude of the 2D Fourier transform of an image.

        Parameters
        ----------
        img : array
            The image to transform.

        Returns
        -------
        magnitude : array
            The magnitude of the 2D Fourier transform of the image.
        """
        return np.abs(self.fft(img))
    
    def power(self, img):
        """
        Compute the power spectrum of an image.

        Parameters
        ----------
        img : array
            The image to transform.

        Returns
        -------
        power : array
            The power spectrum of the image.
        """
        return np.abs(self.fft(img))**2
    
    def radial_profile(self, img, stat='mean'):
        """
        Compute the radial profile of an image.

        Parameters
        ----------
        img : array
            The 2D image to compute the radial profile for.
        stat : str, optional
            The statistic to compute. Can be 'mean' or 'sum'.

        Returns
        -------
        bin_centers : array
            The centers of the bins in the radial direction.
        bin_edges : array
            The edges of the bins in the radial direction.
        bin_means : array
            The mean or sum of the values in each bin, depending on the stat argument.
        bin_entries : array
            The number of entries in each bin.
        """
        y, x = np.indices(img.shape)
        N = img.shape[0]
        
        center = (img.shape[0]/2, img.shape[1]/2)
        #r = np.sqrt((x - center[1] + 0.5)**2 + (y - center[0] + 0.5)**2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Bin edges and indices
        bin_edges = np.arange(int(N/2)+1)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        nbins = len(bin_centers)
        bin_values = np.zeros(nbins).astype(complex)
        bin_entries = np.zeros(nbins)

        # Flatten arrays for vectorized processing
        r_flat = r.ravel()
        v_flat = img.ravel()
        
        # Assign each (i, j) to a bin based on its radius
        bin_indices = np.digitize(r_flat, bin_edges) - 1  # subtract 1 to make indices 0-based
        
        # Mask out indices that fall outside defined bins
        valid = (bin_indices >= 0) & (bin_indices < nbins)
        bin_indices = bin_indices[valid]
        v_flat = v_flat[valid]
        
        # Accumulate real and imaginary parts
        bin_values_real = np.bincount(bin_indices, weights=v_flat.real, minlength=nbins)
        bin_values_imag = np.bincount(bin_indices, weights=v_flat.imag, minlength=nbins)
        bin_entries = np.bincount(bin_indices, minlength=nbins)
        bin_values = bin_values_real + 1j * bin_values_imag

        # Compute statistics
        if stat == 'mean':
            bin_means = bin_values/np.maximum(bin_entries, 1)
        elif stat == 'sum':
            bin_means = bin_values

        return bin_centers, bin_edges, bin_means, bin_entries

    def expand_radial_profile(self, bin_edges, radial_profile, shape):
        """
        Expand a 1D radial profile into a 2D array using given bin edges.

        Parameters
        ----------
        bin_edges : array
            The edges of the bins in the radial direction.
        radial_profile : array
            The 1D profile values corresponding to the bin edges.
        shape : tuple
            The shape of the 2D array to expand into (height, width).

        Returns
        -------
        expanded : 2D array
            A 2D array where each pixel value is determined by the radial profile.
        """

        H, W = shape
        y, x = np.indices((H, W))

        # Center of the image
        center = np.array([H // 2, W // 2])
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Determine which bin each radius falls into
        bin_indices = np.digitize(r, bin_edges) - 1  # subtract 1 to get 0-based indices
        bin_indices = np.clip(bin_indices, 0, len(radial_profile) - 1)
        expanded = radial_profile[bin_indices]

        return expanded
    
    def modulate(self, img):
        """
        Modulate an input image using a predefined noise profile.

        This function takes an input image, performs a Fourier transform, and applies
        amplitude modulation to the frequency components using a predefined noise profile.
        The result is an image where the noise characteristics have been altered according
        to the specified modulation profile.

        Parameters
        ----------
        img : ndarray
            The input image to be modulated.

        Returns
        -------
        ndarray
            The modulated image with adjusted noise characteristics.
        """
        if self.v > 2: self.logger.info("Modulating noise")

        self.white = img.copy()
        if self.y is None:
            self.colored = img.copy()
            return self.colored

        img_fft = self.fft(img)

        if len(img) != self.box_size:
            self.y2D = self.expand_radial_profile(self.be, self.y, img.shape)

        # Apply amplitude modulation to white noise FFT
        white_noise_phase = img_fft / (np.abs(img_fft) + 1e-8)  # unit magnitude
        desired_noise_fft = white_noise_phase * np.sqrt(self.y2D)
        
        # Inverse FFT and normalization
        colored = self.ifft(desired_noise_fft)
        colored /= np.std(colored)
        self.colored = colored.copy()

        return self.colored
    
    def plot_modulated(self, noise_type="", cmap_img="gray", cmap_fft="hot", vmin=None, vmax=None, show=True):
        """
        Plots the original white noise, modulated noise, and their magnitude difference.

        This function visualizes the effects of noise modulation by displaying three
        images side by side: the original white noise image, the modulated noise image,
        and the difference in magnitude between the two in the Fourier domain.

        Parameters
        ----------
        noise_type : str, optional
            A label for the type of noise being visualized. Defaults to an empty string.
        cmap_img : str, optional
            The colormap to use for the image plots. Defaults to "gray".
        cmap_fft : str, optional
            The colormap to use for the Fourier magnitude difference plot. Defaults to "hot".
        vmin : float, optional
            The minimum data value that the colormap covers for the image plots. If None,
            it is set to 1.1 times the minimum value found in the images. Defaults to None.
        vmax : float, optional
            The maximum data value that the colormap covers for the image plots. If None,
            it is set to 1.1 times the maximum value found in the images. Defaults to None.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """

        if vmin is None:
            vmin = min(np.min(self.colored), np.min(self.white))*1.1
        if vmax is None:
            vmax = max(np.max(self.colored), np.max(self.white))*1.1

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        c = ax[0].imshow(self.white, cmap=cmap_img, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(c, pad=0.01)
        cbar.set_label('Intensity (a.u.)', fontsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.yticks([]); plt.xticks([])
        ax[0].axis('off')
        ax[0].set_title(f'White {noise_type} Noise', fontsize=FONTSIZE)

        c = ax[1].imshow(self.colored, cmap=cmap_img, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(c, pad=0.01)
        cbar.set_label('Intensity (a.u.)', fontsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.yticks([]); plt.xticks([])
        ax[1].axis('off')
        ax[1].set_title('Modulated Noise', fontsize=FONTSIZE)

        c = ax[2].imshow(self.magnitude(self.colored) - self.magnitude(self.white), cmap=cmap_fft)
        cbar = plt.colorbar(c, pad=0.01)
        cbar.set_label('Magnitude (a.u.)', fontsize=FONTSIZE)
        cbar.ax.tick_params(labelsize=TICKSIZE)
        plt.yticks([]); plt.xticks([])
        ax[2].axis('off')
        ax[2].set_title('Magnitude Difference', fontsize=FONTSIZE)

        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig

class NoiseGenerator(Base):
    def __init__(self, verbose=1):
        """
        Initialize the NoiseGenerator class.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.

        Returns
        -------
        None
        """
        super().__init__(verbose)
        self.shaper = NoiseShaper(verbose=verbose)
        self.modulator = NoiseModulator(verbose=verbose)
    
    def simulate(self, shape, n=1, store=False):
        """
        Simulate a set of noisy images.

        Parameters
        ----------
        shape : tuple
            Shape of the output image(s).
        n : int, optional
            Number of images to generate. Defaults to 1.
        store : bool, optional
            Store the generated noise in the `white` attribute of the NoiseShaper object.
            Defaults to False.

        Returns
        -------
        colored : ndarray
            The colored noise image(s) with the same shape as the input `shape`.
        """
        self.white = self.shaper.sample(shape, n=n, store=store)
        if n == 1:
            self.colored = self.modulator.modulate(self.white).copy()
        else:
            self.colored = np.empty((n,) + shape)
            for i in range(n):
                self.colored[i] = self.modulator.modulate(self.white[i]).copy()
        
        return self.colored
    
    def plot(self, cmap_img="gray", cmap_fft="hot", vmin=None, vmax=None, show=True):
        """
        Plot the generated noise.

        Parameters
        ----------
        cmap_img : str, optional
            Colormap for the spatial-domain image. Defaults to "gray".
        cmap_fft : str, optional
            Colormap for the frequency-domain image. Defaults to "hot".
        vmin : float, optional
            Minimum data value that the colormap covers. Defaults to None.
        vmax : float, optional
            Maximum data value that the colormap covers. Defaults to None.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        return self.modulator.plot_modulated(
            noise_type=self.shaper.type, cmap_img=cmap_img, cmap_fft=cmap_fft, vmin=vmin, vmax=vmax, show=show
        )
    
    def load_shaper(self, file="noise_shape.pkl"):
        """
        Loads a saved NoiseShaper object from a file.

        Parameters
        ----------
        file : str, optional
            The file path from which the NoiseShaper object is to be loaded. Defaults to "noise_shape.pkl".

        Returns
        -------
        None
        """
        self.shaper = self.shaper.load(file)
    
    def load_modulator(self, file="noise_freq.pkl"):
        """
        Loads a saved NoiseModulator object from a file.

        Parameters
        ----------
        file : str, optional
            The file path from which the NoiseModulator object is to be loaded. Defaults to "noise_freq.pkl".

        Returns
        -------
        None
        """
        self.modulator = self.modulator.load(file)
