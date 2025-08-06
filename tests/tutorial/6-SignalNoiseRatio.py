import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.snr import SNR

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Initializes flat SNR
snr = SNR(snr=0.05, verbose=verbose)
fig = snr.plot(show=show)

# Set values
bin_edges = np.array([4000, 8000, 12000, 16000, 25000]).astype(np.float16)
values = np.array([0.00050, 0.00075, 0.00100, 0.00165])
unit = "A" # or "um"
snr.set_histo(bin_edges, values, unit, emp_factor=2.)
fig = snr.plot(show=show)
plt.close(fig)

# Getting values
snr.set_value_func(interpolate=True)
defocus_single = .9
snr.value(defocus_single)

defocus_arr = [.5, 1., 1.5, 2.] # In micrometers
snr.value(defocus_arr)

snr.set_value_func(interpolate=False)
snr.value(defocus_arr)

# Storing/Loading class
snr.save(f"{THIS_DIR}/test_data/snr.pkl")
snr = SNR().load(f"{THIS_DIR}/test_data/snr.pkl")
