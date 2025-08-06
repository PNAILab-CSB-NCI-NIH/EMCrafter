import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.sampler import DefocusSampler

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Initialize sampler
dsampler = DefocusSampler(verbose=verbose)

# Read defocus
def_array = dsampler.read_defocus(f"{TUTO_PATH}/data/exp_particles/particles_defocus.star")

# Build the sampler
dsampler.build_sampler(
    def_array,            # Defocus array
    n_bins=50,            # Number of bins
    validate=False)       # Plot sampling

# Sample and plot
samples = dsampler.sample(n=20000)
fig = plt.hist(samples, bins=50)
plt.close()

# Storing/Loading class
dsampler.save(f"{THIS_DIR}/test_data/defocus_sampler.pkl")
dsampler = DefocusSampler().load(f"{THIS_DIR}/test_data/defocus_sampler.pkl", validate=False)
