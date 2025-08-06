import sys
from pathlib import Path
sys.path.append(str(Path.cwd()) + "/../src")

from EMCrafter.sim import ParticleSimulator
from EMCrafter.sampler import OrientationSampler, DefocusSampler

def main():
    output_path = "data" 

    # Instantiate Simulator
    pSimulator = ParticleSimulator()

    # Load classes
    pkl_dir = "../TUTORIAL/data/pickle"
    pSimulator.signal = pSimulator.signal.load(f"{pkl_dir}/signal_generator.pkl")
    pSimulator.noise  = pSimulator.noise.load(f"{pkl_dir}/noise_generator.pkl")
    pSimulator.snr    = pSimulator.snr.load(f"{pkl_dir}/snr.pkl")
    orient_sampler    = OrientationSampler().load(f"{pkl_dir}/angle_sampler.pkl")
    defocus_sampler   = DefocusSampler().load(f"{pkl_dir}/defocus_sampler.pkl", validate=False)

    # Store summary plot
    snr_fig = pSimulator.info(show=False)
    snr_fig.savefig(f"{output_path}/simulator_info.png")

    # Setup
    n_particles = 100                           # Number of particles to simulate
    n_cpus = 10                                 # Number of CPUs for parallel computing
    batch_size = 10                             # Number of particles to simulate per subjob
    angles  = orient_sampler.sample(n_particles)
    defocus = defocus_sampler.sample(n_particles)

    # Simulate
    sim = pSimulator.simulate_dataset(
        angles,                                 # Sampled orientations
        defocus,                                # Sampled defocus
        n_cpus=n_cpus,                          # Parallelization
        batch_size=batch_size,                  # Number of simulated particles per subjob
        output=f"{output_path}/sim_particles")  # Output directory
    sim.save(f"{output_path}/sim_dataset.pkl")

    # Exporting:
    export_output = f"{output_path}/rln_project"  
    df = sim.export_particles(export_output, "link", force=True)

# Main function is needed for multiprocessing in MacOS
# For linux, this is not needed
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, but safe
    multiprocessing.set_start_method("fork")
    main()