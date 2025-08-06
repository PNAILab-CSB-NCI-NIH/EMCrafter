import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import time
import inspect
from itertools import chain

from EMAN2 import EMData, EMNumPy

from EMCrafter.base import Base
from EMCrafter.signal import SignalGenerator, DEFOCUS_NORM
from EMCrafter.noise import NoiseGenerator
from EMCrafter.snr import SNR
from EMCrafter.rln import Exporter
from EMCrafter.utils import time_format

FONTSIZE = 15
TICKSIZE = 12
LEGSIZE = 12

MAX_QUICK = 100
MAX_PLOT = 5

class SimResults(Base):
    def __init__(self, rtype="memory", n=None, verbose=1):
        """
        Initialize the SimResults class.

        Parameters
        ----------
        rtype : str
            Results type. Either "memory" (default) or "disk" (written).
        n : int
            Number of particles to simulate.
        verbose : int
            Verbosity level. 0 is quiet, 1 is normal, 2 is very verbose.
        """
        super().__init__(verbose)
        self.particles = []
        self.parameters = {}
        self.plot_ids = None
        if n is not None: self.set_particle_number(n)
        if rtype is not None: self.set_type(rtype)

    def set_particle_number(self, n):
        """
        Set the number of particles in the simulation.

        Parameters
        ----------
        n : int
            Number of particles to simulate.
        """
        self.n = n
    
    def set_type(self, t):
        """
        Set the type of simulation.

        Parameters
        ----------
        t : str
            Type of simulation. Either "memory" (default) or "disk" (written).
        """
        self.type = t

    def set_parameters(self, parameters):
        """
        Set the parameters of the simulation.

        Parameters
        ----------
        parameters : dict
            Simulation parameters.
        """
        self.parameters = parameters
    
    def set_box_size(self, box_size):
        """
        Set the box size of the simulation.

        Parameters
        ----------
        box_size : int
            Box size of the particles.
        """
        self.box_size = box_size
    
    def merge(self, results=[]):
        """
        Merge results from multiple simulations into one.

        Parameters
        ----------
        results : list
            List of files to load and merge.

        Returns
        -------
        SimResults
            The merged simulations.
        """
        # Instantiate simulations
        sims, particles  = [], []
        for i in range(len(results)): 
            sim_i = SimResults().load(results[i])
            sims.append(sim_i)
            particles.append(sim_i.particles.copy())

        # Merge
        self.particles = list(chain.from_iterable(particles))

        # Set attributes
        self.set_particle_number(len(self.particles))
        self.set_parameters(sims[0].parameters)
        self.set_box_size(sims[0].box_size)

        return self
    
    def append(self,
               angle, defocus, snr,
               signal, corrupted,
               noisy, index=None):
        """
        Append a particle to the results.

        Parameters
        ----------
        angle : float
            The angle of the particle.
        defocus : float
            The defocus of the particle.
        snr : float
            The SNR of the particle.
        signal : numpy.ndarray
            The clean signal of the particle.
        corrupted : numpy.ndarray
            The corrupted signal of the particle.
        noisy : numpy.ndarray
            The noisy signal of the particle.
        index : int, optional
            The index of the particle. Defaults to None.
        """
        self.particles.append({
            "angle": angle,
            "defocus": defocus,
            "snr": snr,
            "signal": signal,
            "corrupted": corrupted,
            "noisy": noisy,
            "index": index
        })
    
    def append_array(self,
                    angles, defocus, snr,
                    signal, corrupted,
                    noisy, index=None):
        
        """
        Append a set of particles to the results.

        Parameters
        ----------
        angles : list of float
            The angles of the particles.
        defocus : list of float
            The defocus of the particles.
        snr : list of float
            The SNR of the particles.
        signal : list of numpy.ndarray
            The clean signal of the particles.
        corrupted : list of numpy.ndarray
            The corrupted signal of the particles.
        noisy : list of numpy.ndarray
            The noisy signal of the particles.
        index : list of int, optional
            The index of the particles. Defaults to None.
        """
        if index is None:
            index = [None]*self.n
            self.type = "memory"
        else: self.type = "disk"
        if angles is None: angles = [None]*self.n
        if defocus is None: defocus = [None]*self.n
        if snr is None: snr = [None]*self.n
        if signal is None: signal = [None]*self.n
        if corrupted is None: corrupted = [None]*self.n
        if noisy is None: noisy = [None]*self.n
        
        for i in range(self.n):
            self.append(
                angles[i], defocus[i], snr[i],
                signal[i], corrupted[i],
                noisy[i], index[i])
    
    def plot(self, cmap="gray", ids=None, show=True):
        """
        Plot the particles in the simulation.

        Parameters
        ----------
        cmap : str, optional
            The colormap to use for the images. Defaults to "gray".
        ids : list of int, optional
            The ids of the particles to plot. Defaults to None.
        show : bool, optional
            If True, displays the plot. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        if ids is not None: self.set_plot_ids(ids)
        if self.plot_ids is None: self.set_random_ids()

        p, ids = self.particles, self.plot_ids
        rows, columns = 0, max(5, len(ids))
        if p[ids[0]]["signal"] is not None: rows += 1
        if p[ids[0]]["corrupted"] is not None: rows += 1
        if p[ids[0]]["noisy"] is not None: rows += 1
        
        fig, axs = plt.subplots(rows, columns, figsize=(2*columns, 1.5*rows))
        for i in range(len(ids)):
            k = 0
            index = p[ids[i]]["index"]
            signal = p[ids[i]]["signal"]
            corrupted = p[ids[i]]["corrupted"]
            noisy = p[ids[i]]["noisy"]
            
            if signal is not None:
                if type(signal) == str:
                    signal = EMNumPy.em2numpy(EMData(signal, index))
                fbox = signal.shape[0]/20
                axi = axs[k][i] if rows > 1 else axs[i]
                ci = axi.imshow(signal, cmap)
                cbar_i = plt.colorbar(ci, pad=0.01)
                cbar_i.set_label('Intensity (a.u.)', fontsize=10)
                plt.yticks([]); plt.xticks([])
                axi.axis('off')
                axi.text(fbox, 2*fbox, 'Clean Signal', color='white', fontsize=8, fontweight='bold')
                axi.text(fbox, 19*fbox, f'{tuple(np.round(p[ids[i]]["angle"]).astype(np.int32))}', color='white', fontsize=8, fontweight='bold')
                k += 1
            if corrupted is not None:
                if type(corrupted) == str:
                    corrupted = EMNumPy.em2numpy(EMData(corrupted, index))
                fbox = corrupted.shape[0]/20
                axi = axs[k][i] if rows > 1 else axs[i]
                ci = axi.imshow(corrupted, cmap)
                cbar_i = plt.colorbar(ci, pad=0.01)
                cbar_i.set_label('Intensity (a.u.)', fontsize=10)
                plt.yticks([]); plt.xticks([])
                axi.axis('off')
                axi.text(fbox, 2*fbox, 'CTF Corrupted', color='white', fontsize=8, fontweight='bold')
                axi.text(fbox, 19*fbox, rf'Defocus: {p[ids[i]]["defocus"].astype(np.int32)} $\AA$', color='white', fontsize=8, fontweight='bold')
                k += 1
            if noisy is not None:
                if type(noisy) == str:
                    noisy = EMNumPy.em2numpy(EMData(noisy, index))
                fbox = noisy.shape[0]/20
                axi = axs[k][i] if rows > 1 else axs[i]
                ci = axi.imshow(noisy, cmap)
                cbar_i = plt.colorbar(ci, pad=0.01)
                cbar_i.set_label('Intensity (a.u.)', fontsize=10)
                plt.yticks([]); plt.xticks([])
                axi.axis('off')
                axi.text(fbox, 2*fbox, 'Noisy Image', color='white', fontsize=8, fontweight='bold')
                axi.text(fbox, 19*fbox, f'SNR: {p[ids[i]]["snr"]:.5f}', color='white', fontsize=8, fontweight='bold')
                k += 1
        plt.tight_layout()
        if show: plt.show()
        else: plt.close()
        return fig
        
    def set_plot_ids(self, ids):
        """
        Set the particle IDs to be plotted in the show method.

        Parameters
        ----------
        ids : list or array
            The IDs of the particles to be plotted. If the length is larger than MAX_PLOT,
            only the first MAX_PLOT IDs will be used.

        Returns
        -------
        None
        """
        self.plot_ids = ids[:MAX_PLOT]
    
    def set_random_ids(self):
        """
        Set the particle IDs to be plotted in the show method to a random subset.
        The length of the subset is MAX_PLOT. If the length of the particles is
        smaller than MAX_PLOT, all particles will be used.
        """
        if self.n <= MAX_PLOT:
            self.plot_ids = [i for i in range(self.n)]
        else:
            self.plot_ids = np.random.choice(np.arange(self.n), size=MAX_PLOT, replace=False)
    
    def export_particles(self, output, export_type="link", export_which="noisy", extension="mrcs", shuflle=False, force=False):
        """
        Export the particles to a Relion project.

        Parameters
        ----------
        output : str
            Output folder containing the relion project structure.
        export_type : str, optional
            The type of file building. Needs to be one of ["copy", "move", "link"].
            Defaults to "link".
        export_which : str, optional
            The type of particles to export. Needs to be one of ["clean", "corrupted", "noisy"].
            Defaults to "noisy".
        extension : str, optional
            The extension of the particle files. Defaults to "mrcs".
        shuflle : bool, optional
            Whether to shuffle the particles before exporting. Defaults to False.
        force : bool, optional
            If True, force the file creation, overwriting existing files if necessary.
            Defaults to False.

        Returns
        -------
        pd.DataFrame
            The data frame related to the star file built.
        """
        if self.v: self.logger.info(f"Starting '{inspect.stack()[0][3]}'...")
        start_time = time.time()

        self.rln = Exporter(verbose=self.v)
        self.rln.set_output_dir(output)
        df = self.rln.export(
            self.particles, self.parameters, self.box_size,
            export_type, export_which, extension, shuflle, force)

        if self.v: self.logger.info(f"Finished '{inspect.stack()[0][3]}' in {time_format(start_time, time.time())}.")
        return df


class ParticleSimulator(Base):
    def __init__(self, verbose=1, n_cpus=1):
        """
        Initialize the ParticleSimulator class.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Defaults to 1.
        n_cpus : int, optional
            Number of CPUs to be used. Defaults to 1.

        Returns
        -------
        None
        """
        super().__init__(verbose)
        self.signal = SignalGenerator(verbose=verbose)
        self.noise = NoiseGenerator(verbose=verbose)
        self.snr = SNR(verbose=verbose)
        self.set_cpus(n_cpus)
        self.sim = None
    
    def set_cpus(self, n_cpus):
        """
        Set the number of CPUs to be used.

        Parameters
        ----------
        n_cpus : int
            The number of CPUs to be used. If less than 1, 1 CPU is used. If more
            than the total number of cores available, the total number of cores
            minus one is used.

        Returns
        -------
        None
        """
        total_cores = multiprocessing.cpu_count()
        if n_cpus < 1:
            cores_to_use = 1
        else:
            cores_to_use = n_cpus if n_cpus < total_cores else total_cores-1
        self.cpus = cores_to_use

    def normalize(self, p):
        """
        Normalize the input array to zero mean and unit variance.

        Parameters
        ----------
        p : ndarray
            The input array to normalize. Can be 2D or 3D.

        Returns
        -------
        ndarray
            The normalized array with zero mean and unit variance.
        """

        if len(p.shape) > 2:
            mean = p.mean(axis=(1, 2), keepdims=True)
            std = p.std(axis=(1, 2), keepdims=True)
            return (p - mean) / std
        else: return (p - p.mean())/p.std()
    
    def simulate(self, alt, az, phi, defocus):
        """
        Simulate a particle given Euler angles and a defocus value.

        Parameters
        ----------
        alt : float
            The altitude angle in degrees.
        az : float
            The azimuth angle in degrees.
        phi : float
            The in-plane rotation angle in degrees.
        defocus : float
            The defocus value in Angstroms.

        Returns
        -------
        tuple
            A tuple containing the simulated particle and the mask.
        """
        sgen, ngen = self.signal, self.noise
        volume = sgen.volume

        signal, pmask = sgen.simulate(volume, alt, az, phi, defocus, store=False)
        background = ngen.simulate(signal.shape, n=1)
        
        sim = signal + background/np.sqrt(self.snr.value(defocus/DEFOCUS_NORM))
        sim = self.normalize(sim)
        return sim, pmask

    def _simulate_quick(self, angles, defocus):
        """
        Simulate a set of particles using the quick mode.

        Parameters
        ----------
        angles : array
            A 2D array of shape (n_particles, 3) containing the Euler angles.
        defocus : array
            A 1D array of shape (n_particles,) containing the defocus values.

        Returns
        -------
        SimResults
            A SimResults object containing the simulated particles.
        """
        n_particles = len(angles)
        if n_particles > MAX_QUICK:
            self.logger.warning("Quick mode not supported for more than 10 particles, simulating only 10 particles.")
        sgen, ngen = self.signal, self.noise
        volume = sgen.volume
        
        projection, corrupted, normalized = [], [], []
        if self.v:
            pbar = tqdm(total=n_particles, desc="> Sim. Signal", ascii=True, unit=" image", unit_scale=True, ncols=80, file=sys.stdout)
        for i in range(n_particles):
            alt, az, phi = angles[i]
            _ = sgen.simulate(volume, alt, az, phi, defocus[i], store=True)
            projection.append(sgen.signal_clean.copy())
            corrupted.append(sgen.signal_corrupted.copy())
            normalized.append(sgen.signal_norm.copy())
            if self.v: pbar.update(1)
        
        if self.v:
            pbar.close()
            self.logger.info("> Sim. Noise (vectorized)...")
        background = ngen.simulate(sgen.image_shape, n=n_particles)
        
        if self.v: self.logger.info("> Adding noise (vectorized)...")
        signal = np.array(normalized)
        sqrt_snr = np.sqrt(self.snr.value(defocus/DEFOCUS_NORM))
        if self.snr.type == "flat":
            sqrt_snr = np.array([sqrt_snr]*n_particles)
        noisy = signal + background/(sqrt_snr[:, None, None])

        if self.v: self.logger.info("> Normalizing (vectorized)...")
        noisy = self.normalize(noisy)

        sim = SimResults(rtype="memory", n=n_particles, verbose=self.v)
        sim.set_parameters(sgen.parameters)
        sim.set_box_size(sgen.image_shape[0])
        snr = sqrt_snr**2
        sim.append_array(angles, defocus, snr, projection, normalized, noisy)
        return sim
    
    def _simulate_core_eman(self, args):
        # Unpack arguments
        angles, defocus, store = args
        sgen, ngen = self.signal, self.noise
        n_particles = len(angles)

        # Simulate a set of particles
        particles, indexes = [], np.arange(n_particles).astype(int)
        proj_paths, corrupted_paths = sgen.simulate_n(angles, defocus, store)
        for i in range(n_particles):
            particle_i = EMNumPy.em2numpy(EMData(corrupted_paths, i))
            particles.append(particle_i.copy())
        signal = np.array(particles)
        
        # Get SNR
        sqrt_snr = np.sqrt(self.snr.value(defocus/DEFOCUS_NORM))
        if self.snr.type == "flat":
            sqrt_snr = np.array([sqrt_snr]*n_particles)

        # Simulate and add noise
        noisy_paths = []
        background = ngen.simulate(sgen.image_shape, n=n_particles)
        noisy = signal + background/(sqrt_snr[:, None, None])
        noisy = self.normalize(noisy)
        for i in range(n_particles):
            EMNumPy.numpy2em(noisy[i]).write_image(store["n"], i)
            noisy_paths.append(store["n"])
        
        proj_paths = [proj_paths] * n_particles
        corrupted_paths = [corrupted_paths] * n_particles
        sim = SimResults(rtype="disk", n=n_particles, verbose=0)
        sim.set_parameters(sgen.parameters)
        sim.set_box_size(sgen.image_shape[0])
        proj_paths = None if len(proj_paths) == 0 else proj_paths
        corrupted_paths = None if len(corrupted_paths) == 0 else corrupted_paths
        noisy_paths = None if len(noisy_paths) == 0 else noisy_paths
        snr = sqrt_snr**2
        sim.append_array(angles, defocus, snr, proj_paths, corrupted_paths, noisy_paths, indexes)
        return sim.particles

    def _simulate_core(self, args):
        """
        Simulate a set of particles and store the results on disk.

        Parameters
        ----------
        args : tuple
            A tuple containing:
            - angles : array
                A 2D array of shape (n_particles, 3) containing the Euler angles.
            - defocus : array
                A 1D array of shape (n_particles,) containing the defocus values.
            - store : dict
                A dictionary indicating paths for storing projections, corrupted, 
                and noisy signals with keys "p", "c", "n".

        Returns
        -------
        list
            A list of particle data stored on disk.
        """

        angles, defocus, store = args
        
        n_particles = len(angles)
        sgen, ngen = self.signal, self.noise
        volume = sgen.volume
        
        particles, indexes = [], np.arange(n_particles).astype(int)
        proj_paths, corrupted_paths, noisy_paths = [], [], []
        for i in range(n_particles):
            alt, az, phi = angles[i]
            p, _ = sgen.simulate(volume, alt, az, phi, defocus[i], store=True)
            particles.append(p.copy())
            if store["p"]:
                EMNumPy.numpy2em(sgen.signal_clean).write_image(store["p"], i)
                proj_paths.append(store["p"])
            if store["c"]:
                EMNumPy.numpy2em(sgen.signal_norm).write_image(store["c"], i)
                corrupted_paths.append(store["c"])
        background = ngen.simulate(sgen.image_shape, n=n_particles)
        
        signal = np.array(particles)
        sqrt_snr = np.sqrt(self.snr.value(defocus/DEFOCUS_NORM))
        if self.snr.type == "flat":
            sqrt_snr = np.array([sqrt_snr]*n_particles)
        noisy = signal + background/(sqrt_snr[:, None, None])
        noisy = self.normalize(noisy)

        if store["n"]:
            for i in range(n_particles):
                EMNumPy.numpy2em(noisy[i]).write_image(store["n"], i)
                noisy_paths.append(store["n"])
        
        sim = SimResults(rtype="disk", n=n_particles, verbose=0)
        sim.set_parameters(sgen.parameters)
        sim.set_box_size(sgen.image_shape[0])
        proj_paths = None if len(proj_paths) == 0 else proj_paths
        corrupted_paths = None if len(corrupted_paths) == 0 else corrupted_paths
        noisy_paths = None if len(noisy_paths) == 0 else noisy_paths
        snr = sqrt_snr**2
        sim.append_array(angles, defocus, snr, proj_paths, corrupted_paths, noisy_paths, indexes)
        return sim.particles
    
    def _simulate_dataset(self, angles, defocus, n_cpus, chunk_size, output, intermediates=True, use_eman=False):
        """
        Simulate a set of particles and store the results on disk.

        Parameters
        ----------
        angles : array
            A 2D array of shape (n_particles, 3) containing the Euler angles.
        defocus : array
            A 1D array of shape (n_particles,) containing the defocus values.
        n_cpus : int
            The number of CPUs to use for simulation.
        chunk_size : int
            The number of particles to simulate per chunk.
        output : str
            The directory where to store the simulated particles.
        intermediates : bool, optional
            Whether to store intermediate results (clean and corrupted signals). Defaults to True.
        use_eman : bool, optional
            Whether to use EMAN framewokr for density projections. Defaults to False.

        Returns
        -------
        SimResults
            A SimResults object containing the simulated particles.
        """
        if not os.path.exists(output):
            os.mkdir(output)

        args = []
        for i in range(0, len(angles), chunk_size):
            out_files_i = {
                "p": f"{output}/clean_{i}.mrcs" if intermediates else None,
                "c": f"{output}/corrupted_{i}.mrcs" if intermediates else None,
                "n": f"{output}/noisy_{i}.mrcs"
            }
            arg = (
                angles[i:i+chunk_size],
                defocus[i:i+chunk_size],
                out_files_i
            )
            args.append(arg)
        if self.v: self.logger.info(f"Simulating {len(angles)} particles distributed in {len(args)} chunks of size {chunk_size}...")
        fn = self._simulate_core if not use_eman else self._simulate_core_eman
        with Pool(processes=n_cpus) as pool:
            results = list(tqdm(
                        pool.imap(fn, args, chunksize=1),
                        total=len(args),
                        desc="Simulating",
                        ncols=80,
                        file=sys.stdout
                ))
        
        sim = SimResults(rtype="disk", verbose=self.v)
        sim.particles = list(chain.from_iterable(results))
        sim.set_particle_number(len(sim.particles))
        sim.set_parameters(self.signal.parameters)
        sim.set_box_size(self.signal.image_shape[0])
        return sim

    def simulate_dataset(self, angles, defocus, n_cpus=1, chunk_size=5, quick=False, output="particles", intermediates=True, use_eman=False):
        """
        Simulate a set of particles and store the results on disk.

        Parameters
        ----------
        angles : array
            A 2D array of shape (n_particles, 3) containing the Euler angles.
        defocus : array
            A 1D array of shape (n_particles,) containing the defocus values.
        n_cpus : int, optional
            The number of CPUs to use for simulation. Defaults to 1.
        chunk_size : int, optional
            The number of particles to simulate per chunk. Defaults to 5.
        quick : bool, optional
            If True, simulate quickly by vectorizing projection and corruption steps. Defaults to False.
        output : str, optional
            The directory where to store the simulated particles. Defaults to "particles".
        intermediates : bool, optional
            Whether to store intermediate results (clean and corrupted signals). Defaults to True.
        use_eman : bool, optional
            If True, use EMAN2 framework to simulate project the density. Defaults to False.

        Returns
        -------
        SimResults
            A SimResults object containing the simulated particles.
        """
        if self.v: self.logger.info(f"Starting '{inspect.stack()[0][3]}'...")
        start_time = time.time()

        if n_cpus is None: n_cpus = self.cpus
        self.angles, self.defocus = angles, defocus
        if quick: sim = self._simulate_quick(angles, defocus)
        else: sim = self._simulate_dataset(angles, defocus, n_cpus, chunk_size, output, intermediates, use_eman)

        if self.v: self.logger.info(f"Finished '{inspect.stack()[0][3]}' in {time_format(start_time, time.time())}.")
        return sim
    
    def info(self, show=True):
        """
        Display plots for noise shaper, noise modulator, and SNR.

        Parameters
        ----------
        show : bool, optional
            If True, displays the plots. If False, the figure is closed. Defaults to True.

        Returns
        -------
        fig : Figure
            The matplotlib figure object containing the plots.
        """

        # Create combined figure
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)

        # Generate individual plots
        _ = self.noise.shaper.plot_density(show=False, ax=ax[0])
        _ = self.noise.modulator.plot_modulator_1D(show=False, ax=ax[1])
        _ = self.snr.plot(show=False, ax=ax[2])

        ax[0].legend(fontsize=10)
        ax[0].set_title("Noise Shaper", fontsize=FONTSIZE)
        ax[1].set_title("Noise Modulator", fontsize=FONTSIZE)
        ax[2].set_title("SNR Scaled", fontsize=FONTSIZE)
        for i in range(3):
            ax[i].xaxis.label.set_size(FONTSIZE-2)
            ax[i].yaxis.label.set_size(FONTSIZE-2)
        if show: plt.show()
        else: plt.close(fig)
        return fig
    
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
