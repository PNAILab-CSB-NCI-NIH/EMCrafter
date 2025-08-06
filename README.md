# EMCrafter

**Realistic Cryo-EM Simulation Toolkit**

[![build](https://github.com/PNAILab-CSB-NCI-NIH/EMCrafter/actions/workflows/ci.yml/badge.svg)](https://github.com/PNAILab-CSB-NCI-NIH/EMCrafter/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)   
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Documentation](https://img.shields.io/badge/docs-latest-success.svg)](docs/build/html/index.html)  

---

**EMCrafter** is a simulation package for cryo-electron microscopy (cryo-EM) designed to reproduce the complex background characteristics found in experimental datasets. By extracting noise statistics and structural properties directly from real experimental particles, EMCrafter enables more realistic and customizable dataset simulations which can be ideal for benchmarking, algorithm development, or training deep learning models.

This framework can also be used to systematically test the limitations of existing cryo-EM algorithms by simulating datasets with controlled SNR levels and background noise characteristics. By adjusting these parameters, researchers/engineers/developers can evaluate algorithm robustness, identify failure modes, and, when feasible, develop strategies to correct them, ultimately leading to more resilient and accurate pipelines.

## Key Features

- **Realistic background modeling** from experimental cryo-EM micrographs.  
- **Parallel processing** and **vectorized operations** for high-throughput simulations.  
- **Modular architecture** supporting customizable signal + background pipelines.  
- **Seamless integration** with common cryo-EM pipelines (RELION, CryoSPARC, etc.).  
- Output formats compatible with standard single-particle workflows.  

## Why EMCrafter?

Traditional simulation tools often use oversimplified noise models (e.g., white Gaussian noise), which fail to capture the intricate structure of background signal in real cryo-EM datasets, without even considering the reality of Signal-to-Noise Ratio distributions. EMCrafter addresses this by:

- Modeling frequency-dependent and spatially correlated background.
- Modeling Signal-to-Noise Ratio estimates.
- Providing control over background noise distributions (Gaussian, skewed, exponential-modified, etc.), frequency modulation and scaling power.
- Ensuring simulations closely mimic experimental datasets for more accurate benchmarking.

## Installation

### 1. Clone the repository
```bash
# Clone the repository
git clone https://github.com/PNAILab-CSB-NCI-NIH/EMCrafter.git
cd EMCrafter
```
### 2. Setup environment
To create a Conda environment for this project, run the following commands:
```bash
# Create conda environemnt
conda env create -f environment.yml
conda activate emcraft
```

### 3. Install EMCrafter
```bash
# Install
pip install -e .
```

### 4. Include Kernel into Jupyter
```bash
# Make environment usable into Jupyter
python -m ipykernel install --user --name emcraft
```

### 5. Install EMAN2
We utilize EMAN2’s pdb2mrc script to generate the 3D density map from a PDB file. For this reason:

- Clone the EMAN2 repository:
    ```bash
    git clone https://github.com/cryoem/eman2.git
    ```
- Note the path where you cloned EMAN2; this will be required when creating the density map from a pdb input file.

## Documentation

Full documentation with API references, tutorials, and technical background:  
**[EMCrafter Docs](docs/build/html/index.html)**

## Quick Start Tutorials
The [TUTORIAL](TUTORIAL) directory contains a set of Jupyter notebooks that demonstrate how to use EMCrafter's simulation components in a modular and reproducible way. These tutorials walk you through each major part of the simulation pipeline:

- [`OrientationSampling`](TUTORIAL/1-OrientationSampling.ipynb): Learn how to generate uniformly distributed or custom-oriented random orientations for particle projection simulations.
- [`DefocusSampling`](TUTORIAL/2-DefocusSampling.ipynb): Create realistic defocus values based on the empirical distribution found in experimental cryo-EM datasets.
- [`NoiseShape`](TUTORIAL/3-NoiseShape.ipynb): Understand how to model the spatial distribution of background noise in real experimental particles into simulations using real-space statistics.
- [`NoiseFrequency`](TUTORIAL/4-NoiseFrequency.ipynb): Learn how to model the frequency-dependent characteristics of experimental background noise, matching frequency-dependency to real data.
- [`NoiseGenerator`](TUTORIAL/5-FullNoiseGenerator.ipynb): A wrapper class that unifies both spatial and frequency-based noise generation into a single callable object.
- [`SignalNoiseRatio`](TUTORIAL/6-SignalNoiseRatio.ipynb): Shows how to control the SNR of your simulated data as a function of defocus, which will be used for noise scaling.
- [`SignalGeneration`](TUTORIAL/7-SignalGeneration.ipynb): Demonstrates how to simulate particle signal from atomic models (e.g., PDB files) and CTF corruption.
- [`FullSimulation`](TUTORIAL/8-FullSimulation.ipynb): End-to-end examples of creating small- and large-scale synthetic cryo-EM datasets by combining all the previously introduced components.
- [`ExportSTAR`](TUTORIAL/9-ExportSTAR.ipynb): Convert your synthetic dataset into a .star file compatible with mainstream cryo-EM software like RELION and CryoSPARC.

You can also find a concise version of the simulation pipeline in the root directory:

Additionally, inside [scripts](scripts) you can find a concise version of the simulation pipeline as:
- `EMCrafter.ipynb`: A streamlined Jupyter notebook with minimal explanations. For detailed walkthroughs, see the [TUTORIAL](TUTORIAL).
- `simulate.py`: Runs the full simulation pipeline programmatically.
- `merge.py`: Merges multiple simulated datasets into a single combined dataset.

## Running at Scale

To run efficiently on HPC clusters or multi-core systems, you can distribute the simulation workload across multiple independent jobs. Our tests at the NCI FRCE cluster (**[Frederick Research Computing Environment](https://ncifrederick.cancer.gov/staff/FRCE/Documentation)**) shows that 25,000 particles can be generated in around 50 min using a single machine with 32 CPUs, simulating 100 images/job. Under this information, 500,000 particles could be simulated under an hour considering 20 parallel jobs with 32 CPUs each. The number and size of these jobs should be chosen based on your cluster’s resource availability, such as the number of available CPUs, memory limits, and queue policies.

Once all jobs are complete, the SimResults class can seamlessly merge the outputs from multiple runs into a single consolidated dataset using the merge method (see **[EMCrafter Class Reference](docs/build/html/_modules/EMCrafter/sim.html#SimResults.merge)** and example at [merge.py](scripts/merge.py)) and exporting it as a single dataset. This enables scalable dataset generation while maintaining modularity and reproducibility.

## Citing EMCrafter

If EMCrafter helped your research, please cite:

Zenodo:
```bibtex
@software{emcrafter2025,
  author       = {Degenhardt, Hermann F. and Degenhardt, Maximilia F. S. and Wang, Yun-Xing},
  title        = {{EMCrafter: Large-scale Cryo-EM Simulation Toolkit driven by experimental data profiles}},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16366716},
  url          = {https://doi.org/10.5281/zenodo.16366716}
}
```