import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from EMCrafter.base import Base
from EMCrafter.rln import rlnParticles

class Sampler(Base):
    def __init__(self, seed=None, verbose=1):
        """
        Initialize a Sampler instance.

        Parameters
        ----------
        seed : int or None, optional
            The seed for the random number generator. If None, a random seed is used.
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.
        """
        super().__init__(verbose)
        self.set_seed(seed)
    
    def set_seed(self, seed=None):
        """
        Set the local seed for the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. Defaults to None.

        Returns
        -------
        None
        """
        self.rng = np.random.default_rng(seed=seed)  # Set random number generator
        self.seed = seed                             # Store local seed
        self.random = self.rng.random                # Random number
        self.uniform = self.rng.uniform              # Uniform [0.0, 1.0)
        self.normal = self.rng.normal                # Normal distribution
        self.integers = self.rng.integers            # Discrete uniform
        self.shuffle = self.rng.shuffle              # Shuffle in-place
        self.permutation = self.rng.permutation      # Random permutation
    
    def sample(self):
        """
        Draw a sample. Must be implemented by subclass.
        """
        pass


class OrientationSampler(Sampler):
    def __init__(self, seed=None, verbose=1):
        """
        Initialize an OrientationSampler instance.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level for printing messages. Defaults to 1.

        This method calls the superclass initializer and resets the instance variables.
        """
        super().__init__(seed=seed, verbose=verbose)
        self.reset_vars()
    
    def reset_vars(self):
        """
        Resets the OrientationSampler instance variables to their initial state.

        This includes clearing the lists of angles and clusters, and resetting the counters for
        the number of angles and clusters to 0.
        """
        self.angles = []
        self.clusters = []
        self.set_lengths()
    
    def set_lengths(self):
        """
        Updates the counters for the number of angles and clusters in the OrientationSampler instance.

        This method should be called after modifying the lists of angles or clusters.
        """
        self.n_angles = len(self.angles)
        self.n_clusters = len(self.clusters)
        
    def sample(self, n, limits=(360, 360, 360)):
        """
        Generates n random Euler angles between 0 and the specified limits.
        
        Parameters
        ----------
        n : int
            The number of random angles to generate.
        limits : tuple of 3 ints, optional
            The limits for each of the 3 Euler angle dimensions. Defaults to (360, 360, 360).
        
        Returns
        -------
        self.angles : np.ndarray
            The generated array of random Euler angles.
        """
        if self.v > 2: self.logger.info(f"Generating {n} random Euler angles")
        max_1, max_2, max_3 = limits
        self.reset_vars()
        
        angles_i = self.uniform(0, max_1, n)
        angles_j = self.uniform(0, max_2, n)
        angles_k = self.uniform(0, max_3, n)

        self.angles = np.transpose(np.array([angles_i, angles_j, angles_k]))
        self.set_lengths()
        assert self.n_angles == n

        return self.angles

    def sample_clusters(self, n=10, n_clusters=1, sigma=5, limits=(360, 360, 360)):
        """
        Generates clusters of Euler angles with specified properties.
        
        Parameters
        ----------
        n : int, optional
            The number of random angles to generate per cluster. Defaults to 10.
        n_clusters : int, optional
            The number of clusters of angles to generate. Defaults to 1.
        sigma : float, optional
            The standard deviation for the Gaussian distribution used to generate angles around 
            each cluster center. Defaults to 5.
        limits : tuple of 3 ints, optional
            The limits for each of the 3 Euler angle dimensions. Defaults to (360, 360, 360).
        
        Returns
        -------
        self.angles : np.ndarray
            The generated array of random Euler angles.
        self.clusters : np.ndarray
            The array of cluster centers.
        """
        if self.v > 2: self.logger.info(f"Generating {n_clusters} clusters of {n} random Euler angles each")
        max_1, max_2, max_3 = limits
        self.reset_vars()
        
        for i in range(n_clusters):
            x1 = self.uniform(0, max_1)
            x2 = self.uniform(0, max_2)
            x3 = self.uniform(0, max_3)
            self.clusters.append([x1, x2, x3])

            angles_i = self.normal(x1, sigma, n)
            angles_j = self.normal(x2, sigma, n)
            angles_k = self.normal(x3, sigma, n)

            angles_i = np.transpose(np.array([angles_i, angles_j, angles_k]))
            self.angles.extend(angles_i)
    
        self.shuffle(self.angles)
        self.angles = np.array(self.angles)
        self.n_angles = len(self.angles)
        self.clusters = np.array(self.clusters)
        self.n_clusters = len(self.clusters)
        assert n_clusters == self.n_clusters
        
        return self.angles, self.clusters

    def plot(self, limits=[360, 360, 360], show=True):
        """
        Plots the Euler angles stored in the OrientationSampler instance.

        Parameters
        ----------
        limits : list of 3 ints, optional
            The limits for each of the 3 Euler angle dimensions. Defaults to [360, 360, 360].
        show : bool, optional
            Whether to show the plot. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')
        colors = ["black", "red", "green", "blue", "magenta", "cyan", "darkgreen", "grey", "pink", "brown"]
    
        n, c = 1, {}
        if self.n_clusters > 0:
            km = KMeans(n_clusters=len(self.clusters), init=self.clusters, max_iter=5, tol=0.0001).fit(self.angles)
            res = km.predict(self.angles)
            for i in range(len(res)):
                try: c[str(res[i])].append(self.angles[i])
                except:
                    c[str(res[i])] = []
                    c[str(res[i])].append(self.angles[i])
            n = len(c)
        else: c["0"] = self.angles
        
        l1, l2, l3 = [0, limits[0]], [0, limits[1]], [0, limits[2]]
        for i in range(n):
            x, y, z = np.array(c[str(i)]).T
            if limits[0] == 360:
                l1 = [-180, 180]
                for j in range(len(x)):
                    if x[j] > 180: x[j] = x[j] - 360
            if limits[1] == 360:
                l2 = [-180, 180]
                for j in range(len(x)):
                    if y[j] > 180: y[j] = y[j] - 360
            if limits[2] == 360:
                l3 = [-180, 180]
                for j in range(len(x)):
                    if z[j] > 180: z[j] = z[j] - 360
    
            # Make the direction data for the arrows
            u, v, w = x, y, z
            
            ax.quiver(x, y, z, u, v, w, length=20, normalize=True, color=colors[i%len(colors)])
        ax.set_xlim(l1); ax.set_ylim(l2); ax.set_zlim(l3)
        ax.set_xlabel('alt'); ax.set_ylabel('az'); ax.set_zlabel('phi')
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig


class DefocusSampler(Sampler):
    def __init__(self, arr=None, file=None, seed=None, verbose=1):
        """
        Initialize the DefocusSampler.
        
        Parameters
        ----------
        arr : array, optional
            An array of defocus values
        file : str, optional
            The path to a STAR file containing the defocus values
        """
        super().__init__(seed=seed, verbose=verbose)
        self.arr = None
        self.samples = []
        if arr is not None:
            self.set_arr(arr)
        if file is not None:
            self.read_defocus(file)

    def read_defocus(self, file):
        """
        Reads defocus values from a STAR file and sets the defocus array.

        Parameters
        ----------
        file : str
            The path to the STAR file containing defocus values.

        Returns
        -------
        np.ndarray
            An array of average defocus values computed as the mean of 'rlnDefocusU' and 'rlnDefocusV'.
        """
        self.data, self.optics = rlnParticles(file, self.v).get()
        
        defocus_u = self.data['rlnDefocusU'].to_numpy()
        defocus_v = self.data['rlnDefocusV'].to_numpy()
        defocus = (defocus_u + defocus_v)/2

        self.set_arr(defocus)
        return self.arr

    def set_arr(self, arr):
        """
        Sets the defocus array, given as an input.

        Parameters
        ----------
        arr : array
            An array of defocus values

        Raises
        ------
        ValueError
            If the provided array is empty or None.
        """
        if arr is not None and len(arr) > 0:
            self.arr = arr
        else: raise ValueError(f"Cannot use the provided array as defocus distribution.")

    def build_sampler(self, arr, n_bins=100, validate=True):
        """
        Builds a sampler from the given array of defocus values.
        
        Parameters
        ----------
        arr : array
            An array of defocus values
        n_bins : int, optional
            The number of bins to use when computing the histogram
        validate : bool, optional
            If True, plots the histogram of the defocus distribution

        Returns
        -------
        None
        """
        if self.v > 2: self.logger.info(f"Building sampler from {len(arr)} defocus values")
        self.x, self.y = self.get_histo(arr, n_bins)
        if validate: self.validate(self.x, self.y)

    def sample(self, n=1):
        """
        Samples random defocus values based on the histogram distribution.

        Parameters
        ----------
        n : int, optional
            The number of random defocus values to sample. Defaults to 1.

        Returns
        -------
        np.ndarray
            An array of sampled defocus values.
        """
        if self.v > 2: self.logger.info(f"Sampling {n} defocus values")
        w_2 = (self.x[1] - self.x[0])/2
        widths = np.random.uniform(-w_2, w_2, n)
        values = np.random.choice(self.x, p=self.y, size=n) + widths
        self.samples = np.array(values)
        return self.samples

    def validate(self, x, y, show=True):
        """
        Validates the sampler by comparing the histogram of sampled defocus values
        with the expected distribution. This function is only used for testing
        purposes.

        Parameters
        ----------
        x : array
            The x-coordinates of the histogram
        y : array
            The y-coordinates of the histogram
        show : bool, optional
            If True, displays the plot

        Returns
        -------
        None
        """
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
        n_bins = len(x)
        ns = [1000, 20000, 100000]#[1_000, 10_000, 100_000]
    
        for i, n in enumerate(ns):
            # Choose unit label
            if n >= 1_000_000:
                label = f"{n // 1_000_000}M"
            elif n >= 1_000:
                label = f"{n // 1_000}k"
            else:
                label = str(n)
    
            values = self.sample(n)
            axs[i].hist(values, bins=n_bins, range=(self.arr.min(), self.arr.max()), label=f"{label} picks")
            axs[i].scatter(x, y * n, color="orange", label="expected", alpha=0.5)
            axs[i].set_ylabel(r"Counts")
            axs[i].legend()
        
        axs[i].set_xlabel(r"Defocus ($\AA$)")
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig

    def get_histo(self, vals, n_bins=100):
        """
        Compute the histogram of the input values.

        Parameters
        ----------
        vals : array_like
            Input values
        n_bins : int, optional
            Number of bins (default is 100)

        Returns
        -------
        x : array
            Bin centers
        y : array
            Normalized histogram values
        """
        x_min = vals.min()
        x_max = vals.max()
    
        # Compute bin edges (n_bins + 1 values)
        edges = np.linspace(x_min, x_max, n_bins + 1)
    
        # Compute bin centers (midpoints of each bin)
        centers = 0.5 * (edges[:-1] + edges[1:])
    
        # Sanity checks
        assert len(edges) == n_bins + 1
        assert len(centers) == n_bins
        assert round(min(centers), 6) == round(x_min + (edges[1] - edges[0]) / 2, 6)
        assert round(max(centers), 6) == round(x_max - (edges[1] - edges[0]) / 2, 6)
    
        # Get histogram
        y, _ = np.histogram(vals, bins=edges)
    
        # Normalize
        y = y / y.sum()
    
        return centers, y

    def load(self, file, verbose=None, validate=False):
        """
        Loads a saved DefocusSampler object from a file using pickle.

        Parameters
        ----------
        file : str
            The file path from which the DefocusSampler object will be loaded.
        validate : bool, optional
            If True, validate the loaded sampler against the original data.

        Returns
        -------
        self : DefocusSampler
            The loaded DefocusSampler object.
        """
        obj = super().load(file, verbose)
        if validate:
            obj.validate(self.x, self.y)
        return obj