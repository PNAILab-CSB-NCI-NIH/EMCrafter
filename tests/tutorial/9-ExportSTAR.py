from pathlib import Path
from EMCrafter.sim import SimResults

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

# Setup
verbose = 2
show = False

# Load simulated dataset
sim = SimResults().load(f"{THIS_DIR}/test_data/dataset_full.pkl")

# Export as link
export_output = f"{THIS_DIR}/test_data/rln_project"
df = sim.export_particles(export_output, "link", force=True)
#df = sim.export_particles(export_output, "copy", force=True) # Symlink already exists, will fail
#df = sim.export_particles(export_output, "move", force=True) # Move file, so cannot use afterwards
