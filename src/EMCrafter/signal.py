import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import multiprocessing
from multiprocessing import Pool
import subprocess
import time
import inspect
from pathlib import Path
from scipy.ndimage import binary_dilation

from EMAN2 import EMData, EMNumPy, parsemodopt, parsesym

from EMCrafter.base import Base
from EMCrafter.utils import time_format

DEFOCUS_NORM = 10000
FONTSIZE = 15
TICKSIZE = 12
LEGSIZE = 12

class SignalGenerator(Base):
    def __init__(self, eman_dir=None, force=False, n_cpus=1, env=None, verbose=1):
        """
        Initialize a SignalGenerator object.
        
        Parameters
        ----------
        eman_dir : str
            Directory path to EMAN2.
        force : bool
            Force execution even if the output files already exist.
        n_cpus : int
            Number of CPUs to be used during the simulation.
        env : str
            Path to the conda environment to be used for the simulation. If not provided, the environment variable $CONDA_DEFAULT_ENV will be used.
        verbose : int
            Verbosity level. (Default: 1)
        """
        super().__init__(verbose)
        self.eman_dir = None
        self.pdb = None
        self.output = None
        self.env = False
        self.python = None
        self.volume_path = None
        self.volume = None
        self.signal_clean = None
        self.signal_mask = None
        self.signal_corrupted = None
        self.signal_norm = None
        self.noise = None
        self.stored = False
        self.stored_clean = False
        self.stored_corrupted = False
        self.stored_normalized = False
        self.store_volume = True
        self.logs = {}
        self.set_env(env)
        self.set_cpus(n_cpus)
        self.set_force(force)
        if eman_dir: self.set_eman_dir(eman_dir)
    
    def __getstate__(self):
        """
        Return a picklable state of the object.
        
        If the 'store_volume' flag is False, the 'volume' attribute is not stored in the state.
        """

        state = self.__dict__.copy()
        if not self.store_volume:
            if 'volume' in state:
                del state['volume']
        return state
    
    def __setstate__(self, state):
        """
        Restore the state of the SignalGenerator object from a given state.

        Parameters
        ----------
        state : dict
            The state dictionary from which to restore the object's attributes.
        """

        self.__dict__.update(state)
        store_volume = state.get('store_volume', None)
        if not store_volume:
            volume_path = state.get('volume_path', None)
            if volume_path is not None:
                self.load_volume(volume_path)
    
    def set_force(self, force):
        """
        Set whether to force the simulation, even if the output files already exist.

        Parameters
        ----------
        force : bool
            If True, force the simulation, overwriting existing files if necessary.
            If False, do not force and throw an error if the output files already exist.

        Returns
        -------
        None
        """
        self._force = force
    
    def validate_init(self):
        """
        Validate that the necessary variables have been set before running a simulation.

        This checks that the EMAN directory, PDB file, and output directory have been set, and that the Python executable has been set.

        Raises
        ------
        ValueError
            If any of the necessary variables have not been set.
        FileNotFoundError
            If any of the necessary variables point to a non-existent file or directory.
        """
        if self.eman_dir is None:
            raise ValueError("Eman directory was not set, please use 'set_eman_dir' method before starting simulations.")
        if self.pdb is None:
            raise ValueError("PDB file was not set, please use 'set_pdb' method before starting simulations.")
        if self.output is None:
            raise ValueError("Output directory was not set, please use 'set_output' method before starting simulations.")
        if not os.path.exists(self.output):
            raise FileNotFoundError("Output directory does not exist, please use 'set_output' method before starting simulations.")
        if not os.path.exists(self.python):
            raise FileNotFoundError("Python not set, please use 'set_env' method before starting simulations.")
    
    def validate_files(self, files=[], labels=[]):
        """
        Validate that the given files exist.

        Parameters
        ----------
        files : list
            The list of files to validate
        labels : list
            The labels to use for each file if it does not exist

        Raises
        ------
        FileNotFoundError
            If any of the files do not exist.
        """
        for i in range(len(files)):
            if not os.path.exists(files[i]):
                raise FileNotFoundError(f"File {labels[i]} could not be found at: {files[i]}")

    def validate_output(self, output_path, output=None):
        """
        Validate that the process produced output.

        Parameters
        ----------
        output_path : str
            The path to the file that should have been produced by the process.
        output : subprocess.CompletedProcess, optional
            The output of the process. If this is given and the file does not exist, the error message will include the process's output.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(output_path):
            err = f"Process did not produce output"
            if output is not None:
                err = err + f""", please check the logs:
                    {self.get_error(output)}"""
            raise FileNotFoundError(err)
    
    def get_error(self, output):
        """
        Return a string containing the error message of a subprocess.

        Parameters
        ----------
        output : subprocess.CompletedProcess
            The output of the process.

        Returns
        -------
        str
            A string containing the error message of the process.
        """
        args   = output.args
        exit   = output.returncode
        stderr = output.stderr.decode()
        stdout = output.stdout.decode()
        return f"""------------------------------------------------------------------
                EXIT CODE: {exit}
                ARGUMENTS: {args}
                ---------------------------- STDOUT ------------------------------
                {stdout}
                ---------------------------- STDERR ------------------------------
                {stderr}
                ------------------------------------------------------------------"""

    def set_eman_dir(self, eman_dir):
        """
        Set the path to the EMAN2 directory.

        Parameters
        ----------
        eman_dir : str
            The path to the EMAN2 directory.

        Raises
        ------
        TypeError
            If the path is not a string.
        FileNotFoundError
            If the specified directory does not exist.
        """
        if self.v > 1: self.logger.info(f"Setting EMAN directory to: {eman_dir}")
        if type(eman_dir) != str:
            raise TypeError("Eman directory must be a string pointing to the path of the eman2!")
        abs_path = os.path.abspath(eman_dir)
        if not os.path.exists(abs_path):
            raise FileNotFoundError("Eman directory specified does not exist!")
        self.eman_dir = abs_path
    
    def set_pdb(self, pdb):
        """
        Set the path to the PDB file.

        Parameters
        ----------
        pdb : str
            The path to the PDB file.

        Raises
        ------
        TypeError
            If the path is not a string.
        FileNotFoundError
            If the specified file does not exist.
        """
        if self.v > 1: self.logger.info(f"Setting PDB file to: {pdb}")
        if type(pdb) != str:
            raise TypeError("PDB must be a string pointing to the path of a PDB file!")
        if not os.path.exists(pdb):
            raise FileNotFoundError("PDB file specified does not exist!")
        self.pdb = os.path.abspath(pdb)
        self.pdb_name = self.pdb.split("/")[-1].split(".")[0]

    def set_output(self, output):
        """
        Set the output directory.

        Parameters
        ----------
        output : str
            The path to the output directory.

        Raises
        ------
        TypeError
            If the path is not a string.
        FileExistsError
            If the specified directory already exists, and `force` is False.
        """
        if self.v > 1: self.logger.info(f"Setting output directory to: {output}")
        if type(output) != str:
            raise TypeError("Output must be a string pointing to the output path!")
        path = os.path.abspath(output)
        if not os.path.exists(path):
            os.makedirs(path)
        elif not self._force:
            raise FileExistsError("Output directory already exists!")
        tmp_path = os.path.join(path, ".tmp")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        self.output = path
        self.tmp = tmp_path

    def set_env(self, env=None):
        """
        Set the conda environment for the simulation.

        Parameters
        ----------
        env : str, optional
            The path to the conda environment to be used. If not provided, the current system's default environment is used.

        Raises
        ------
        TypeError
            If the provided environment path is not a string.
        FileNotFoundError
            If the environment could not be determined or does not exist.
        """
        if self.v > 1 and env is not None: self.logger.info(f"Setting environment to: {env}")
        if env is not None and type(env) != str:
            raise TypeError("Environment must be a string pointing to the conda path!")
        if env is not None:
            self.conda_env = env
        else:
            self.conda_env = sys.exec_prefix
        if self.conda_env is None:
            raise FileNotFoundError("Environment could not be found.")
        self.python = f"{self.conda_env}/bin/python"
        print_path = os.sep.join(self.python.strip(os.sep).split(os.sep)[-5:])
        if self.v > 1: self.logger.info(f"Environment set to: {print_path}")

    def set_cpus(self, n_cpus):
        """
        Set the number of CPUs to use for the simulation.

        Parameters
        ----------
        n_cpus : int
            The number of CPUs to use. If less than 1, 1 CPU is used. If more
            than the total number of cores available, the total number of cores
            minus one is used.
        """
        if self.v > 1: self.logger.info(f"Setting number of CPUs to: {n_cpus}")
        total_cores = multiprocessing.cpu_count()
        if n_cpus < 1:
            cores_to_use = 1
        else:
            cores_to_use = n_cpus if n_cpus < total_cores else total_cores-1
        self.cpus = cores_to_use
    
    def set_parameters(self, apix, voltage=300, cs=2.7, bfactor=100, ampcont=0.1):
        """
        Set the simulation parameters.

        Parameters
        ----------
        apix : float
            Resolution in Angstroms per pixel.
        voltage : float, optional
            Voltage in kV. Default is 300.
        cs : float, optional
            Spherical aberration in mm. Default is 2.7.
        bfactor : float, optional
            Bfactor in A^2. Default is 100.
        ampcont : float, optional
            Contrast in percent. Default is 0.1.
        """
        if self.v > 1: self.logger.info(f"Setting parameters to: {apix=}, {voltage=}, {cs=}, {bfactor=}, {ampcont=}")
        res2 = (2.5*apix)**2
        self.parameters = {
            "noiseamp":      0, # Noise amplitude
            "voltage":       voltage, # Voltage
            "apix":          apix, # Angstroms per pixel
            "cs":            cs, # Spherical aberration
            "bfactor":       bfactor - 4*res2, # Bfactor
            "defocus":       None,
            "ampcont":       ampcont, # Contrast
        }
        
    
    def pdb2map(self, apix, box=None, res=None, center=True, stdout=True):
        """
        Convert a PDB file into a 3D electron density map.

        Parameters
        ----------
        apix : float
            Resolution in Angstroms per pixel.
        box : int, optional
            Size of the output box in pixels. Default is None.
        res : float, optional
            Resolution of the output map in Angstroms. Default is 2.5 times the apix value.
        center : bool, optional
            If True, the map is centered. Default is True.
        stdout : bool, optional
            If True, the output is printed. Default is True.
        """
        if self.v: self.logger.info(f"Starting '{inspect.stack()[0][3]}'...")
        start_time = time.time()
        self.validate_init()

        script_path = f"{self.eman_dir}/programs/e2pdb2mrc.py"
        output_path = f"{self.output}/{self.pdb_name}_electron_density.mrc"
        self.validate_files([script_path], ["e2pdb2mrc.py"])
        
        if res is None:
            res = apix*2.5

        commands = [
                self.python, script_path, self.pdb, output_path,
                "--apix", str(apix), "--res", str(res)]
        if box is not None:
            commands.extend(["--box", str(box)])
        if center:
            commands.append("--center")
        output = subprocess.run(commands, capture_output=True, shell=False)

        self.validate_output(output_path, output)
        if stdout:
            print("-------------------------------------")
            print(output.stdout.decode())
            print("-------------------------------------")

        self.logs["pdb2map"] = output
        self.volume_path = output_path
        self.load_volume(self.volume_path)
        self.image_shape = (box, box)
        
        if self.v: self.logger.info(f"Finished '{inspect.stack()[0][3]}' in {time_format(start_time, time.time())}.")
    
    def load_volume(self, volume_path=None):
        """
        Load the volume from the given path or use the internal volume path.

        Parameters
        ----------
        volume_path : str, optional
            The path to the volume file. Default is None, which uses the internal path.
        """
        if volume_path is None:
            volume_path = self.volume_path
        self.volume = EMData(volume_path)
    
    def em2np(self, obj):
        """
        Convert an EMData or EMData array to a numpy array.

        Parameters
        ----------
        obj : EMData or EMData array
            The object to convert.

        Returns
        -------
        numpy array
            The converted numpy array.
        """
        return EMNumPy.em2numpy(obj)
    
    def np2em(self, obj):
        """
        Convert a numpy array to an EMData or EMData array.

        Parameters
        ----------
        obj : numpy array
            The object to convert.

        Returns
        -------
        EMData or EMData array
            The converted EMData or EMData array.
        """
        return EMNumPy.numpy2em(obj)
    
    def make_mask(self, p, thr=0.01, pad=15):
        p_np = EMNumPy.em2numpy(p).copy()
        pmask = (p_np > thr) * 1
        if pad > 0:
            pmask = binary_dilation(pmask, iterations=pad)
        return pmask

    def density_projections_core(self, args):
        """
        Run the e2project3d.py script in the background using subprocess.

        Parameters
        ----------
        args : tuple
            A tuple containing the arguments for the subprocess.run call.
            The tuple should contain the following elements in order:
            (script_path: str, input_file: str, output_path_i: str, orientations: str, verbose: int)

        Returns
        -------
        subprocess.CompletedProcess
            The CompletedProcess object returned by subprocess.run.
        """
        script_path, input_file, output_path_i, orientations, verbose = args
        commands = [
            self.python, script_path, input_file, f"--outfile={output_path_i}",
            orientations, "--projector=standard", f"--verbose={verbose}"]
        output = subprocess.run(commands, capture_output=True, shell=False)
        return output
    
    def merge_single_files(self, files, output_path, clean=False, verbose=0):
        """
        Merge multiple EMData files into a single output file.

        Parameters
        ----------
        files : list of str
            List of file paths to the input EMData files to be merged.
        output_path : str
            Path to the output file where the merged data will be saved.
        clean : bool, optional
            If True, delete the original input files after merging. Default is False.
        verbose : int, optional
            Verbosity level for logging and progress updates. Default is 0.

        Returns
        -------
        str
            The path to the output file containing the merged data.
        """

        if verbose: pbar = tqdm(total=len(files), ncols=80, file=sys.stdout, desc = 'Merging Files  ')

        for i, file in enumerate(files):
            particle_i = EMData(file)
            particle_i.write_image(output_path, i)
            if verbose: pbar.update(1)
        if verbose: pbar.close()

        if clean:
            if verbose: self.logger.info("Cleaning temporary files...")
            safe_tmp = Path(self.tmp).resolve()
            for file in files:
                tmp_file = Path(file).resolve()
                if tmp_file.is_relative_to(safe_tmp):
                    os.remove(tmp_file)
                else:
                    self.logger.warning(f"Skipping unsafe file {tmp_file} outside tmp directory.")
        return output_path
    
    def simulate_single_eman(self, args):
        """
        Run a single e2project3d.py subprocess to simulate a projection from an EMData file.

        Parameters
        ----------
        args : tuple
            A tuple with the following elements:
            (script_path: str, input_file: str, output_path_i: str, orientations: str, verbose: int)

        Returns
        -------
        subprocess.CompletedProcess
            The CompletedProcess object returned by subprocess.run.
        """
        script_path, input_file, output_path_i, orientations, verbose = args
        commands = [
            self.python, script_path, input_file, f"--outfile={output_path_i}",
            orientations, "--projector=standard", f"--verbose={verbose}"]
        output = subprocess.run(commands, capture_output=True, shell=False)
        return output
    
    def simulate_n_projections(self, angles, store, cpus=1, verbose=0):
        """
        Simulate n 2D projections from a given set of angles.

        Parameters
        ----------
        angles : numpy array
            A 2D numpy array of shape (n_projections, 3) containing the Euler angles for each projection.
        store : dict
            A dictionary containing the path to the output file in the 'p' key.
        cpus : int, optional
            The number of CPUs to use for parallel processing. Default is 1.
        verbose : int, optional
            The verbosity level for logging and progress updates. Default is 0.
        
        Returns
        -------
        str
            The path to the output file containing the simulated projections.
        """
        script_path = f"{self.eman_dir}/programs/e2project3d.py"
        input_file = f"{self.output}/{self.pdb_name}_electron_density.mrc"
        output_path = store['p']
        name_path = Path(output_path).stem
        self.validate_files([script_path, input_file], ["e2project3d.py", "input"])

        assert len(angles) > 0, "Empty array..."
        assert np.shape(angles)[1] == 3, "Array is not 3D..."
        n_projections = len(angles)

        files, args = [], []
        for i in range(n_projections):
            angle = angles[i]
            orientations = f"--orientgen=single:alt={angle[0]}:az={angle[1]}:phi={angle[2]}"
            output_path_i = f"{self.tmp}/{name_path}_{i}.hdf"
            arg = (script_path, input_file, output_path_i, orientations, 0)
            args.append(arg)
            files.append(output_path_i)

        if cpus > 1:
            with Pool(processes=self.cpus) as pool: #chunksize=1
                _ = list(
                    tqdm(
                        pool.imap(self.simulate_single_eman, args, chunksize=1),
                        total=len(args),
                        file=sys.stdout,
                        desc="2-D Projections",
                        ncols=80,
                        disable=verbose < 1
                    ))
        else:
            for arg in args:
                self.simulate_single_eman(arg)
        
        self.n_projections = self.merge_single_files(files, output_path, clean=True)
        self.validate_output(output_path)
        return output_path
    
    def corrupt_n_particles(self, file, defocus, store, thr=0.01, pad=15, norm=True):
        """
        Corrupt a set of particles by applying a contrast transfer function and normalization.

        Parameters
        ----------
        file : str
            The path to the input file containing particle data.
        defocus : array-like
            A list or array of defocus values for each particle.
        store : dict
            A dictionary containing the path to the output file in the 'c' key.
        thr : float, optional
            Threshold for mask generation. Default is 0.01.
        pad : int, optional
            Padding size for mask generation. Default is 15.
        norm : bool, optional
            Whether to normalize the particles after processing. Default is True.

        Returns
        -------
        str
            The path to the output file containing the corrupted particles.
        """

        output_path = store['c']

        n = len(defocus)
        for i in range(n):
            i_pars = self.parameters.copy()
            p = EMData(file, i)
            pmask = self.make_mask(p, thr, pad)
            i_pars["defocus"] = defocus[i]/DEFOCUS_NORM
            p = p.process('math.simulatectf', i_pars)
            p = p.process('normalize')
            if norm:
                p_np = EMNumPy.em2numpy(p).copy()
                p_np = (p_np - p_np[pmask].mean())/p_np[pmask].std()
                p_norm = EMNumPy.numpy2em(p_np.copy())
                p_norm.write_image(output_path, i)
            else:
                p.write_image(output_path, i)

        return output_path
    
    def simulate_n(self, angles, defocus, store, cpus=1, verbose=0):
        """
        Simulate a set of particles and corrupt them by applying a contrast transfer function and normalization.

        Parameters
        ----------
        angles : array-like
            A 2D array of shape (n_particles, 3) containing the Euler angles.
        defocus : array-like
            A list or array of defocus values for each particle.
        store : dict
            A dictionary containing the path to the output file in the 'p' key.
        cpus : int, optional
            The number of CPUs to use for simulation. Default is 1.
        verbose : int, optional
            The level of verbosity. Default is 0.

        Returns
        -------
        tuple
            A tuple containing the paths to the simulated and corrupted particles.
        """
        self.validate_init()
        sim_path = self.simulate_n_projections(angles, store, cpus, verbose)
        corr_path = self.corrupt_n_particles(sim_path, defocus, store, thr=0.01, pad=15, norm=True)
        return sim_path, corr_path

    def project_signal(self, volume, alt, az, phi, mask=True, thr=0.01, pad=15, store=False, astype="em"):
        """
        Project a volume onto a plane given Euler angles.

        Parameters
        ----------
        volume : EMData or numpy array
            The volume to project.
        alt : float
            The altitude angle in degrees.
        az : float
            The azimuth angle in degrees.
        phi : float
            The in-plane rotation angle in degrees.
        mask : bool, optional
            Whether to generate a mask. Defaults to True.
        thr : float, optional
            The threshold value for the mask. Defaults to 0.5.
        pad : int, optional
            The number of iterations of binary dilation to apply to the mask. Defaults to 20.
        store : bool, optional
            Whether to store the clean signal and mask. Defaults to False.
        astype : str, optional
            The type of the output. Options are "em" for EMData and "np" for numpy array. Defaults to "em".

        Returns
        -------
        tuple
            A tuple containing the projected particle and the mask.
        """
        orientgen = f'single:alt={alt}:az={az}:phi={phi}'
        projector = 'standard'
        [og_name, og_args] = parsemodopt(orientgen)
        sym_object = parsesym("C1")
        eulers = sym_object.gen_orientations(og_name, og_args)
        euler = eulers[0]
        
        transform = {"transform": euler}
        p = volume.project(projector, transform)
        p.set_attr("xform.projection", euler)
        p.set_attr("ptcl_repr", 0)
        
        pmask = self.make_mask(p, thr, pad) if mask else \
                np.ones(self.image_shape, dtype=bool)
            
        self.stored_clean = store
        if store:
            self.signal_clean = EMNumPy.em2numpy(p).copy()
            self.signal_mask = pmask.copy()
        
        if astype == "np": p = self.em2np(p)
        
        return p, pmask

    def corrupt_signal(self, p, defocus, store=False):
        """
        Apply a CTF corruption to the signal.

        Parameters
        ----------
        p : EMData or ndarray
            The signal to corrupt.
        defocus : float
            The defocus value in Angstroms.
        store : bool, optional
            Store the corrupted signal. Defaults to False.

        Returns
        -------
        p : EMData or ndarray
            The corrupted signal.
        """
        pars = self.parameters.copy()
        pars["defocus"] = defocus/DEFOCUS_NORM
        p = p.process('math.simulatectf', pars)
        p = p.process('normalize')
        p_np = EMNumPy.em2numpy(p).copy()
        self.stored_corrupted = store
        if store: self.signal_corrupted = p_np.copy()
        return p_np.copy()
    
    def normalize_signal(self, p, pmask, store=False):
        """
        Normalize a signal to zero mean and unit variance using the given mask.

        Parameters
        ----------
        p : EMData or ndarray
            The signal to normalize.
        pmask : ndarray
            The binary mask to use for normalization.
        store : bool, optional
            Store the normalized signal. Defaults to False.

        Returns
        -------
        p : EMData or ndarray
            The normalized signal.
        """
        bmask = pmask.astype(bool)
        p = (p - p[bmask].mean())/p[bmask].std()
        self.stored_normalized = store
        if store: self.signal_norm = p.copy()
        return p.copy()
    
    def simulate(
            self, volume, alt, az, phi, defocus,
            mask=True, thr=0.5, pad=20, corrupt=True,
            normalize=True, store=False):
        """
        Simulate a particle given Euler angles and a defocus value.

        Parameters
        ----------
        volume : EMData or ndarray
            The volume to project.
        alt : float
            The altitude angle in degrees.
        az : float
            The azimuth angle in degrees.
        phi : float
            The in-plane rotation angle in degrees.
        defocus : float
            The defocus value in Angstroms.
        mask : bool, optional
            Whether to generate a mask. Defaults to True.
        thr : float, optional
            The threshold value for the mask. Defaults to 0.5.
        pad : int, optional
            The number of iterations of binary dilation to apply to the mask. Defaults to 20.
        corrupt : bool, optional
            Whether to corrupt the signal with a CTF. Defaults to True.
        normalize : bool, optional
            Whether to normalize the signal. Defaults to True.
        store : bool, optional
            Store the simulated particle, mask, and intermediate steps. Defaults to False.

        Returns
        -------
        tuple
            A tuple containing the simulated particle and the mask.
        """
        p, pmask = self.project_signal(volume, alt, az, phi, mask=mask, thr=thr, pad=pad, store=store)
        if corrupt: p = self.corrupt_signal(p, defocus, store=store)
        if normalize: p = self.normalize_signal(p, pmask, store=store)
        self.stored = store
        return p, pmask

    def plot(self, p=None, pmask=None, use_mask=True, cmap="gray", show=True):
        """
        Plot the simulated signal and its intermediate states.

        Parameters
        ----------
        p : EMData or ndarray, optional
            The simulated signal to plot. If None, uses stored data. Default is None.
        pmask : ndarray, optional
            The binary mask to apply to the signal. Default is None.
        use_mask : bool, optional
            Whether to apply the mask to the signal during plotting. Default is True.
        cmap : str, optional
            The colormap to use for plotting. Default is "gray".
        show : bool, optional
            Whether to display the plot. If False, the figure is closed. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        Raises
        ------
        Exception
            If no data is available to plot and no particle argument is provided.
        """

        if p is None and not self.stored:
            raise Exception("No data to plot. Please run 'simulate' first with store=True"
                "or provide the the 'p' (particle) argument.")
        
        toplot, labels = [], []
        mask = np.ones(self.image_shape)
        if p is not None:
            toplot.append(p)
            labels.append("Simulated Signal")
            if use_mask and pmask is not None:
                mask = pmask
        elif self.stored:
            if self.stored_clean:
                toplot.append(self.signal_clean)
                labels.append("Clean Signal")
            if self.stored_corrupted:
                toplot.append(self.signal_corrupted)
                labels.append("CTF Corrupted")
            if self.stored_normalized:
                toplot.append(self.signal_norm)
                labels.append("Normalized")
            if use_mask and self.signal_mask is not None:
                mask = self.signal_mask
        
        n_plots = len(toplot)
        fig, ax = plt.subplots(1, n_plots, figsize=(n_plots*4, 3))
        for i in range(n_plots):
            axi = ax[i] if n_plots > 1 else ax
            c = axi.imshow(toplot[i]*mask, cmap=cmap)
            cbar = plt.colorbar(c, pad=0.01)
            cbar.set_label('Intensity (a.u.)', fontsize=FONTSIZE)
            cbar.ax.tick_params(labelsize=TICKSIZE)
            plt.yticks([]); plt.xticks([])
            axi.axis('off')
            axi.set_title(labels[i], fontsize=FONTSIZE)
        plt.tight_layout()
        if show: plt.show()
        else: plt.close(fig)
        return fig
