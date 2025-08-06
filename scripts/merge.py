import sys
from pathlib import Path
sys.path.append(str(Path.cwd()) + "/../src")

from EMCrafter.sim import SimResults

# Set the paths to your simulated datasets and desired output path
input_files = [
    "data/sim_dataset.pkl", # Just for exemplification,
    "data/sim_dataset.pkl", # the same dataset is used twice
]
output_path = "data"

# Initialize SimResults, merge dataset and save
dataset = SimResults().merge(input_files)
dataset.save(f"{output_path}/merged_dataset.pkl")

# Export the merged dataset into a STAR file
export_output = f"{output_path}/rln_project"  
_ = dataset.export_particles(export_output, "link", force=True)
