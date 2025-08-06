import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.noise import NoiseFit, NoiseShaper
from EMCrafter.utils import circular_mask

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Initialize NoiseFit
particles_file = f"{TUTO_PATH}/data/exp_particles/particles_nshape.star"
nfit = NoiseFit(particles_file, verbose=2)

# Load images
base = f"{TUTO_PATH}/data/exp_particles/"
nfit.load_images(base)

# Create a mask
radius, soft = 120, 0
imask, omask = circular_mask(nfit.image_shape, radius=radius, soft=soft)

# Get noise distribution
nfit.get_noise_distribution(mask=omask)
fig = nfit.plot_noise_distribution(dmin=1e-5, dmax=0.6, show=show)
plt.close(fig)

# Fits
nfit.fit("gaussian")
gauss_params = nfit.fit_pars.copy()
fig = nfit.plot_fit(dmin=1e-5, dmax=1., show=show)
plt.close(fig)

nfit.fit("exp_mod_gaussian")
emg_params = nfit.fit_pars.copy()
fig = nfit.plot_fit(dmin=1e-5, dmax=1., show=show)
plt.close(fig)

nfit.fit("skew_normal")
skew_params = nfit.fit_pars.copy()
fig = nfit.plot_fit(dmin=1e-5, dmax=1., show=show)
plt.close(fig)

# Initialize Noise Shaper and sample: Gaussian
gauss_shaper = NoiseShaper(verbose=verbose)
gauss_shaper.set_gaussian_shape(*gauss_params)
sample = gauss_shaper.sample(shape=(480,480), n=1)
fig = gauss_shaper.plot_image(show=show)
plt.close(fig)
fig = gauss_shaper.plot_density(show=show)
plt.close(fig)

# Initialize Noise Shaper and sample: Exponentially Modified Gaussian
emg_shaper = NoiseShaper(verbose=verbose)
emg_shaper.set_exp_mod_gaussian_shape(*emg_params)
sample = emg_shaper.sample(shape=(480,480), n=1)
fig = emg_shaper.plot_image(show=show)
plt.close(fig)
fig = emg_shaper.plot_density(show=show)
plt.close(fig)

# Initialize Noise Shaper and sample: Skew-Normal
skew_shaper = NoiseShaper(verbose=verbose)
skew_shaper.set_skew_normal_shape(*skew_params)
amp   =  0.99957600  # Amplitude
loc   =  0.85578852  # Location
scale =  1.31405641  # Scale
alpha = -1.41129763  # Skewness
skew_shaper.set_skew_normal_shape(amp, loc, scale, alpha)
sample = skew_shaper.sample(shape=(480,480), n=1)
fig = skew_shaper.plot_image(show=show)
plt.close(fig)
fig = skew_shaper.plot_density(show=show)
plt.close(fig)

# Storing/Loading class
skew_shaper.save(f"{THIS_DIR}/test_data/noise_shape.pkl")
skew_shaper = NoiseShaper().load(f"{THIS_DIR}/test_data/noise_shape.pkl")
