.. EMCrafter documentation master file, created by
   sphinx-quickstart on Tue Jul 29 12:25:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EMCrafter documentation
=======================

**EMCrafter** is a Python package designed to simulate cryo-electron microscopy (cryo-EM) datasets
under experimentally realistic conditions, with a focus on accurately modeling the background signal.
Unlike traditional simulation tools that often rely on idealized or overly simplistic noise models,
EMCrafter seeks to replicate the subtle complexities of real experimental data by analyzing and
reproducing the statistical and structural properties of background noise directly from micrographs.

Cryo-EM datasets are influenced by numerous experimental variables, including sample preparation
methods, imaging conditions, microscope hardware, and particle characteristics. These factors introduce
heterogeneity and artifacts into the background that can significantly affect particle detection,
classification, and reconstruction. EMCrafter addresses this challenge by extracting the relevant
statistical signatures from real cryo-EM datasets, such as spatial correlations, amplitude distributions,
and frequency-domain features,  and incorporating them into simulation workflows.

In addition to its realism-focused modeling, EMCrafter is built for efficiency. The package leverages
parallel computing and vectorized operations wherever possible, allowing users to generate large datasets
at scale without compromising performance. This makes it ideal for high-throughput testing of algorithms,
large-scale reconstructions, or training machine learning models with realistic synthetic data.

Its modular architecture allows for fine-grained control over signal generation, noise modeling, and
dataset synthesis. EMCrafter integrates smoothly with common cryo-EM software ecosystems and supports
output formats compatible with RELION, CryoSPARC, and other popular pipelines.

Whether you aim to test the limits of reconstruction methods or simulate realistic training dataset
for deep learning models, EMCrafter provides a powerful, flexible framework for crafting detailed,
noise-aware cryo-EM simulations.

Modules
=======

.. automodule:: EMCrafter
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   EMCrafter