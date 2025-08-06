import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.signal import SignalGenerator

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Set variables:
EMAN2_DIR = f"{THIS_DIR}/../../../../../eman2/" # EMAN2 directory path
PDB = f"{TUTO_PATH}/data/pdbs/V1.pdb"           # PDB file path
OUTPUT = f"{THIS_DIR}/test_data"                # Output path

# Initialize class
sgen = SignalGenerator(force=True, verbose=verbose)
sgen.set_eman_dir(EMAN2_DIR)
sgen.set_pdb(PDB)
sgen.set_output(OUTPUT)
sgen.validate_init()

# Density map:
apix = 0.732       # Resolution (A/pixel)
box_size = 480     # Image size
stdout = True      # Print process result
sgen.pdb2map(apix=apix, box=box_size, stdout=stdout)

# Experimental parameters
voltage = 300  # Voltage
cs = 2.7       # Spherical aberration
bfactor = 55   # Bfactor
ampcont = 0.1  # Contrast
sgen.set_parameters(apix=apix, voltage=voltage, cs=cs, bfactor=bfactor, ampcont=ampcont)

# Simulate signal
volume = sgen.volume          # Generated Density Map
alt, az, phi = 135, -45, 75   # Euler angles (EMAN2 convention)
defocus = 10000               # Defocus value (angstrom)
p, pmask = sgen.project_signal(volume, alt, az, phi)

p_np = sgen.em2np(p)
fig = sgen.plot(p_np, show=show)
plt.close(fig)

# CTF corruption
pc = sgen.corrupt_signal(p, defocus)
fig = sgen.plot(pc, show=show)
plt.close(fig)

# Normalize
pn = sgen.normalize_signal(pc, pmask)
fig = sgen.plot(pn, show=show)
plt.close(fig)

# Full simulation process
p, pmask = sgen.simulate(sgen.volume, alt, az, phi, defocus, store=True)
fig = sgen.plot(show=show)
plt.close(fig)

# Storing/Loading class
sgen.save(f"{THIS_DIR}/test_data/signal_generator.pkl")
sgen = SignalGenerator().load(f"{THIS_DIR}/test_data/signal_generator.pkl")
