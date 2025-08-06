from pathlib import Path
import matplotlib.pyplot as plt
from EMCrafter.sampler import OrientationSampler

THIS_DIR = Path(__file__).parent

# Setup
verbose = 2
show = False

# Initialize
osampler = OrientationSampler(verbose=verbose)

# Generating random Euler angles
angles = osampler.sample(n=100)
fig = osampler.plot(show=show)

# Generating clusters
angles = osampler.sample_clusters(  
    n = 10,             # Generate 10 random orientations per cluster
    n_clusters = 50,    # Generate 20 random clusters of orientations 
    sigma = 10)         # Sigma as 10 degrees
fig = osampler.plot(show=show)   # Visualize
plt.close(fig)

# Storing/Loading class
osampler.save(f"{THIS_DIR}/test_data/orientation_sampler.pkl")
osampler = OrientationSampler().load(f"{THIS_DIR}/test_data/orientation_sampler.pkl")
