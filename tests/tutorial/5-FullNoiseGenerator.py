import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.noise import NoiseGenerator

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Parameters
apix = 0.732                       # Resoluiton in A/pixel
box_size = 480                     # Box size
image_shape = (box_size, box_size) # Image shape

# Initialize NoiseGenerator
ngen = NoiseGenerator(verbose=verbose)
noise_image = ngen.simulate(shape=image_shape, n=1)
fig = ngen.plot(show=show)
plt.close(fig)

# Use pickle
ngen = NoiseGenerator(verbose=verbose)
ngen.load_shaper(f"{THIS_DIR}/test_data/noise_shape.pkl")
ngen.load_modulator(f"{THIS_DIR}/test_data/noise_freq.pkl")
noise_image = ngen.simulate(shape=image_shape, n=1)
fig = ngen.plot(show=show)
plt.close(fig)

bin_edges = np.array([
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  15,
     20,  25,  30,  40,  60,  80, 100, 120, 160, 180, 200, 205, 210,
    215, 220, 225, 230, 235, 240])
variance = np.array([
    2.11325322e-11, 5.40978151e-01, 5.40628613e-01, 4.58879223e-01, 4.14320672e-01, 4.09713557e-01, 4.07239538e-01, 4.24285155e-01,
    4.39586586e-01, 4.58531033e-01, 4.64357290e-01, 3.93507035e-01, 2.36888590e-01, 1.83334006e-01, 1.79167898e-01, 1.83943069e-01,
    2.35892694e-01, 3.54219475e-01, 5.37067798e-01, 6.84487136e-01, 9.46889104e-01, 1.22259777e+00, 1.41436848e+00, 1.53763435e+00,
    1.58937269e+00, 1.64073593e+00, 1.69385432e+00, 1.74332645e+00, 1.79907557e+00, 1.85219952e+00, 1.90266923e+00])

# Initialize the class from scratch
ngen = NoiseGenerator(verbose=verbose)
ngen.modulator.set_apix(apix=apix)
ngen.shaper.set_skew_normal_shape(amplitude=0.999576, loc=0.85578852, scale=1.31405641, alpha=-1.41129763)
ngen.modulator.set_modulation(bin_edges, variance, box_size=box_size)
noise_image = ngen.simulate(shape=image_shape, n=1)
fig = ngen.plot(show=show)
plt.close(fig)

# Storing/Loading class
ngen.save(f"{THIS_DIR}/test_data/noise_generator.pkl")
ngen = NoiseGenerator().load(f"{THIS_DIR}/test_data/noise_generator.pkl")
