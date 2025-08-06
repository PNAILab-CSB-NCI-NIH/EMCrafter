import os
import numpy as np
import pandas as pd
import starfile
import subprocess
from collections import defaultdict
import mrcfile

from EMCrafter.base import Base

EXPORT_TYPES = ["copy", "move", "link"]
EXPORT_WHICH = ["signal", "corrupted", "noisy"]
DEFOCUS_NORM = 10_000

class rlnParticles(Base):
    def __init__(self, file=None, verbose=1):
        """
        Initialize a rlnParticles object.

        Parameters
        ----------
        file : str or None, optional
            The path to a STAR file containing the particle data.
            If None, an empty object is created.
        """
        super().__init__(verbose)
        self.data = pd.DataFrame()
        self.optics = pd.DataFrame()
        self.base = None
        if file is not None:
            self.read(file)
    
    def _extract_params(self):
        """
        Extract parameters from the optics DataFrame.
        """
        self.params = {
            "voltage": self.optics["rlnVoltage"].iloc[0] if "rlnVoltage" in self.optics else None,
            "cs": self.optics["rlnSphericalAberration"].iloc[0] if "rlnSphericalAberration" in self.optics else None,
            "ampcont": self.optics["rlnAmplitudeContrast"].iloc[0] if "rlnAmplitudeContrast" in self.optics else None,
            "apix": self.optics["rlnImagePixelSize"].iloc[0] if "rlnImagePixelSize" in self.optics else None,
            "box_size": self.optics["rlnImageSize"].iloc[0] if "rlnImageSize" in self.optics else None
        }

    def set_base_folder(self, base):
        """
        Set the base folder for reading micrograph and particle files.

        Parameters
        ----------
        base : str
            The base folder containing the micrograph and particle files.
        """
        self.base = base

    def read(self, file):
        """
        Reads a STAR file and sets the data and optics attributes.

        Parameters
        ----------
        file : str
            The path to the STAR file containing the particle data.
        """
        raw = starfile.read(file)
        self.data = pd.DataFrame(raw["particles"])
        self.optics = pd.DataFrame(raw["optics"])
        self._extract_params()
    
    def load_images(self, base, n=None):
        """
        Loads the particle images from the STAR file.

        Parameters
        ----------
        base : str
            The base folder containing the particle images.
        n : int, optional
            The maximum number of images to load. If None, all images are loaded.
            Defaults to None.
        """
        self.set_base_folder(base)

        img_paths = self.data["rlnImageName"].to_list()

        self.images, counter = [], 0
        for i in range(len(img_paths)):
            index, path = img_paths[i].split('@')
            index = int(index)-1
            path = os.path.join(self.base, path)
            if not os.path.exists(path): continue
            with mrcfile.mmap(path, mode='r') as mrc:
                data = None
                if len(mrc.data.shape) > 2:
                    data = mrc.data[index]
                else: data = mrc.data
                self.images.append({
                    "i": i,
                    "data": data.copy(),
                    "index": index,
                    "path": path,
                })
            counter += 1
            if n is not None and counter >= n: break

    def get(self):
        """
        Retrieve the particle and optics data.

        Returns
        -------
        tuple
            A tuple containing two pandas.DataFrames:
            - The first DataFrame contains the particle data.
            - The second DataFrame contains the optics data.
        """

        return self.data, self.optics
    
    def set_data(self, data, optics):
        """
        Set the particle and optics data.

        Parameters
        ----------
        data : pandas.DataFrame
            The particle data.
        optics : pandas.DataFrame
            The optics data.
        """
        self.data = data
        self.optics = optics
        self._extract_params()

    def write(self, file, att=None, n=None, ids=None):
        """
        Writes the particle and optics data to a STAR file.

        Parameters
        ----------
        file : str
            The path to the output STAR file.
        att : str, optional
            The attribute to use for selecting particles. If None, all particles are used.
            Defaults to None.
        n : int, optional
            The maximum number of particles to write. If None, all particles are written.
            Defaults to None.
        ids : list of int, optional
            The indices of the particles to write. If None, all particles are written.
            Defaults to None.
        """
        data = self.data if att is None else self.data[att]
        data = data.iloc[ids] if ids is not None else data
        data = data.iloc[:n] if n is not None else data
        starfile.write(
            {'optics': self.optics, 'particles': data},
            file,
        )

class Exporter(Base):
    def __init__(self, output_dir=None, verbose=1):
        """
        Initialize an Exporter object.

        Parameters
        ----------
        output_dir : str
            Directory where the relion project will be exported.
            If None, the export directory needs to be set later with set_output_dir.
        verbose : int
            Verbosity level. 0 is quiet, 1 is normal, higher values are more verbose.
        """
        super().__init__(verbose)
        self.job_number = 1
        self.job_manager = None
        self.job_temporary = None
        self.micrographs = None
        self.project_identifier = None
        self.extract_folder = None
        self.particles_file = None
        self.particles_movies = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

    def set_output_dir(self, output_dir):
        """
        Set the output directory for the relion project.

        Parameters
        ----------
        output_dir : str
            Directory where the relion project will be exported.
            If the directory already exists, existing files may be overwritten.
        """
        if type(output_dir) != str:
            raise TypeError(f"Output directory must be a string!")
        self.output_dir = os.path.abspath(output_dir)
        if os.path.exists(self.output_dir):
            self.logger.warning(f"Output directory exists, existing files may be overwritten.")
        else:
            if self.v: self.logger.info(f"Creating output directory...")
            os.makedirs(self.output_dir, exist_ok=True)
        if self.v:
            self.logger.info(f"Output directory set to: {os.path.relpath(self.output_dir)}.")
        self.job_manager = f"{self.output_dir}/default_pipeline.star"
        self.job_temporary = f"{self.output_dir}/.TMP_runfiles"
        self.relative_micrographs = "Movies"
        self.micrographs = f"{self.output_dir}/{self.relative_micrographs}"
        self.project_identifier = f"{self.output_dir}/.gui_projectdir"
        self.extract_folder = f"{self.output_dir}/Extract/job001"
        self.particles_file = f"{self.extract_folder}/particles.star"
        self.particles_movies = f"{self.extract_folder}/Movies"
        self.build_output()

    def build_output(self):
        """
        Build the output directory by creating the necessary subdirectories and files.
        """
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder)
        if not os.path.exists(self.particles_movies):
            os.mkdir(self.particles_movies)
        if not os.path.exists(self.micrographs):
            os.makedirs(self.micrographs, exist_ok=True)
        if not os.path.exists(self.job_manager):
            with open(self.job_manager, "w") as fp:
                fp.write(DEFAULT_JOB)
        if not os.path.exists(self.project_identifier):
            with open(self.project_identifier, "w") as fp:
                fp.write("")
        if not os.path.exists(self.job_temporary):
            os.mkdir(self.job_temporary)
        files_needed = ["RELION_JOB_EXIT_SUCCESS", "run.out", "run.err"]
        for f in files_needed:
            fp = open(f"{self.extract_folder}/{f}", "w")
            fp.close()
    
    def validate_export(self, particles, parameters, box_size, export_type="link", export_which="noisy", extension="mrcs", shuflle=False):
        """
        Validate the arguments for exporting particles to a relion project.

        Parameters
        ----------
        particles : list
            The list of particles to export.
        parameters : dict
            The simulation parameters.
        box_size : int
            The box size of the particles.
        export_type : str, optional
            The type of export to perform. Options are "copy", "move" or "link".
            Defaults to "link".
        export_which : str, optional
            The type of particles to export. Options are "noisy" or "dose_weighted".
            Defaults to "noisy".
        extension : str, optional
            The extension of the particle files. Defaults to "mrcs".
        shuflle : bool, optional
            Whether to shuffle the particles before exporting. Defaults to False.

        Returns
        -------
        p : numpy.ndarray
            The validated and shuffled (if needed) list of particles.

        Raises
        ------
        TypeError
            If any of the arguments are of the wrong type.
        ValueError
            If the list of particles is empty or if a required parameter is missing.
        """
        if self.v: self.logger.info(f"> Export: Validate arguments")
        if export_type not in EXPORT_TYPES:
            raise TypeError(f"Export type must be one of {EXPORT_TYPES}")
        if export_which not in EXPORT_WHICH:
            raise TypeError(f"Export type must be one of {EXPORT_WHICH}")
        if len(particles) == 0:
            raise ValueError("Empty list of particles")
        if type(box_size) != int or box_size <= 0:
            raise TypeError(f"Box size must be a positive integer")
        if type(extension) != str or extension == "":
            raise TypeError(f"Extension must be a non-empty string")
        must_have = ["voltage", "apix", "cs", "ampcont"]
        for mh in must_have:
            if mh not in parameters:
                raise ValueError(f"Parameter {mh} is missing from parameters.")
        if len(particles) == 0:
            raise ValueError("Empty list of particles")
        p = np.array(particles.copy())
        if shuflle: np.random.shuffle(p)
        return p
    
    def format_optics(self, parameters, box_size):
        """
        Format the optics data for a relion project.

        Parameters
        ----------
        parameters : dict
            The simulation parameters.
        box_size : int
            The box size of the particles.

        Returns
        -------
        optics : pandas.DataFrame
            The formatted optics data.
        """
        if self.v: self.logger.info(f"> Export: Format optics")
        optics_data = {
            "rlnOpticsGroupName": ["opticsGroup1"],
            "rlnOpticsGroup": [1],
            "rlnVoltage": [parameters["voltage"]],
            "rlnSphericalAberration": [parameters["cs"]],
            "rlnAmplitudeContrast": [parameters["ampcont"]],
            "rlnImagePixelSize": [parameters["apix"]],
            "rlnImageSize": [box_size],
            "rlnImageDimensionality": [2],
            "rlnCtfDataAreCtfPremultiplied": [0],
        }
        optics = pd.DataFrame(optics_data)
        return optics
    
    def format_particles(self, particles, export_which):
        """
        Format the particle data for export to a relion project.

        Parameters
        ----------
        particles : list
            The list of particle dictionaries to format.
        export_which : str
            The key indicating which type of particle data to export.

        Returns
        -------
        data : defaultdict(list)
            The formatted particle data with keys from the original particle items.
        n_particles : int
            The number of particles.
        original_keys : list
            The list of keys that were retained in the formatted data.
        """

        if self.v: self.logger.info(f"> Export: Format particles")

        # Format data
        n_particles = len(particles)
        data = defaultdict(list)
        original_keys = ["defocus", export_which, "index"]
        for p in particles:
            for k, v in p.items():
                data[k].append(v)
        for k in list(data.keys()):
            if k not in original_keys:
                del data[k]
        
        return data, n_particles, original_keys
    
    def create_fake_micrograph(self, path):
        """
        Create an empty file at the specified path.

        Parameters
        ----------
        path : str
            The path to create the empty file.
        """
        _ = subprocess.run(["touch", path], shell=False)
    
    def copy(self, source, destination, copy_type="link", force=False):
        """
        Copy, move, or link a file from the source to the destination.

        Parameters
        ----------
        source : str
            The path to the source file.
        destination : str
            The path to the destination file.
        copy_type : str, optional
            The type of operation to perform: "link" (default), "move", or "copy".
        force : bool, optional
            If True, force the operation, overwriting existing files if necessary. Default is False.

        Raises
        ------
        subprocess.CalledProcessError
            If the command execution fails.
        """
        cmd = {
            "link": ["ln", "-s"],
            "move": ["mv"],
            "copy": ["cp"]
        }[copy_type]
        if force: cmd += ["-f"]
        
        command = cmd + [source, destination]
        _ = subprocess.run(command, capture_output=True, check=True, shell=False)
    
    def build_particles(self, data, n, export_type, export_which, extension, force):
        """
        Build physical files for the particles in the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The data frame containing the particles information.
        n : int
            The number of particles in the data.
        export_type : str
            The type of file building: "link", "move", or "copy".
        export_which : str
            The column name in the data frame indicating which files to export.
        extension : str
            The file extension to use for the exported files.
        force : bool
            If True, force the file creation, overwriting existing files if necessary.

        Returns
        -------
        pd.DataFrame
            The data frame with the locations of the exported files.
        """
        if self.v: self.logger.info(f"> Export: Build physical files")

        # Create files
        file_paths = np.unique(data[export_which])
        path_converter = {}
        for i, f in enumerate(file_paths):
            label = os.path.basename(f).split(".")[0]
            mic_path = f"{self.relative_micrographs}/micrograph_{i:05}_{label}.mrc"
            self.create_fake_micrograph(f"{self.output_dir}/{mic_path}")
            img_path = f"Extract/job001/Movies/micrograph_{i:05}_{label}_particles.{extension}"
            self.copy(source=f, destination=f"{self.output_dir}/{img_path}", copy_type=export_type, force=force)
            path_converter[f] = (mic_path, img_path)

        # Store locations on data
        for i in range(n):
            mic_i, img_i = path_converter[data[export_which][i]]
            data["rlnMicrographName"].append(mic_i)
            data["rlnImageName"].append(img_i)
        return data

    def set_particles_attributes(self, data, original_keys, which):
        # Create dataframe
        """
        Set attributes for the particles in the given data.

        Parameters
        ----------
        data : list of dict
            The data containing the particles information.
        original_keys : list
            The column names in the original data.
        which : str
            The particle type: "noisy" or "clean".

        Returns
        -------
        pd.DataFrame
            The data frame with the set attributes.
        """
        df = pd.DataFrame(data)
        n = len(df)

        # Set attributes
        df["rlnImageName"] = df.apply(lambda x: f'{x["index"] + 1}@{x["rlnImageName"]}', axis=1)
        df["rlnDefocusU"] = df["defocus"]
        df["rlnDefocusV"] = df["defocus"]
        df["rlnDefocusAngle"] = 0.
        df["rlnCtfBfactor"] = 0.
        df["rlnCtfScalefactor"] = 1.
        df["rlnPhaseShift"] = 0.
        df["rlnOpticsGroup"] = 1
        df["rlnClassNumber"] = 0
        df["rlnCoordinateX"] = np.random.randint(0, 10000, n)
        df["rlnCoordinateY"] = np.random.randint(0,  8000, n)

        df["rlnAutopickFigureOfMerit"] = 0.15
        df["rlnCtfMaxResolution"] = 5.00
        df["rlnCtfFigureOfMerit"] = 0.05

        # Reorder
        for k in original_keys: del df[k]
        columns_order = [
            "rlnImageName", "rlnMicrographName",
            "rlnOpticsGroup", "rlnClassNumber",
            "rlnCoordinateX", "rlnCoordinateY",
            "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle",
            "rlnCtfBfactor", "rlnCtfScalefactor", "rlnPhaseShift",
            "rlnAutopickFigureOfMerit",
            "rlnCtfMaxResolution", "rlnCtfFigureOfMerit",
        ]

        return df[columns_order]
    
    def write(self, particles, optics):
        """
        Write a STAR file with the given particles and optics data.

        Parameters
        ----------
        particles : pd.DataFrame
            The particles data frame.
        optics : pd.DataFrame
            The optics data frame.
        """
        if self.v: self.logger.info(f"> Export: Compile STAR file")
        starfile.write(
            {'optics': optics, 'particles': particles},
            self.particles_file,
        )
    
    def export(self, particles, parameters, box_size, export_type="link", export_which="noisy", extension="mrcs", shuflle=False, force=False):
        """
        Export particles to a relion project.

        Parameters
        ----------
        particles : list
            The list of particles to export.
        parameters : dict
            The simulation parameters.
        box_size : int
            The box size of the particles.
        export_type : str, optional
            The type of export to perform. Options are "copy", "move" or "link".
            Defaults to "link".
        export_which : str, optional
            The type of particles to export. Options are "noisy" or "clean".
            Defaults to "noisy".
        extension : str, optional
            The extension of the particle files. Defaults to "mrcs".
        shuflle : bool, optional
            Whether to shuffle the particles before exporting. Defaults to False.
        force : bool, optional
            Whether to override the existing files. Defaults to False.

        Returns
        -------
        pd.DataFrame
            The data frame with the set attributes.
        """
        p_val = self.validate_export(particles, parameters, box_size,export_type, export_which, extension, shuflle)
        opt = self.format_optics(parameters, box_size)
        data, n, keys = self.format_particles(p_val, export_which)
        data = self.build_particles(data, n, export_type, export_which, extension, force)
        par = self.set_particles_attributes(data, keys, export_which)
        self.write(par, opt)
        return par


DEFAULT_JOB = """
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                      2
 

# version 30001

data_pipeline_processes

loop_ 
_rlnPipeLineProcessName #1 
_rlnPipeLineProcessAlias #2 
_rlnPipeLineProcessTypeLabel #3 
_rlnPipeLineProcessStatusLabel #4 
Extract/job001/       None relion.extract  Succeeded 

# version 30001

data_pipeline_nodes

loop_ 
_rlnPipeLineNodeName #1 
_rlnPipeLineNodeTypeLabel #2
Extract/job001/particles.star ParticlesData.star.relion 

# version 30001

data_pipeline_input_edges

loop_ 
_rlnPipeLineEdgeFromNode #1 
_rlnPipeLineEdgeProcess #2 
AutoPick/job000/autopick.star Extract/job001/ 
 

# version 30001

data_pipeline_output_edges

loop_ 
_rlnPipeLineEdgeProcess #1 
_rlnPipeLineEdgeToNode #2  
Extract/job001/ Extract/job001/particles.star 
"""