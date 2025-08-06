import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from EMCrafter.sim import ParticleSimulator
from EMCrafter.sampler import OrientationSampler, DefocusSampler

THIS_DIR = Path(__file__).parent
TUTO_PATH = f"{THIS_DIR}/../../TUTORIAL"

def main():
    # Setup
    verbose = 2
    show = False

    # General parameters
    n_cpus = 10                        # Number of CPUs for parallelization
    apix = 0.732                       # Resolution (A/pixel)
    box_size = 480                     # Image size
    image_shape = (box_size, box_size) # Image shape

    # Instantiate Simulator
    pSimulator = ParticleSimulator(verbose=verbose)
    pSimulator.info(show=show)

    # Load classes
    pSimulator.signal = pSimulator.signal.load(f"{THIS_DIR}/test_data/signal_generator.pkl")
    pSimulator.noise  = pSimulator.noise.load(f"{THIS_DIR}/test_data/noise_generator.pkl")
    pSimulator.snr    = pSimulator.snr.load(f"{THIS_DIR}/test_data/snr.pkl")
    pSimulator.info(show=show)

    # Set parameters
    alt, az, phi = 75, -35, 75  # Orientation
    defocus = 10000             # Defocus
    p, pmask = pSimulator.simulate(alt, az, phi, defocus)
    sgen = pSimulator.signal
    ngen = pSimulator.noise
    fig = sgen.plot(p, show=show)
    plt.close(fig)

    # Load our angle and defocus samplers
    osamp = OrientationSampler().load(f"{THIS_DIR}/test_data/orientation_sampler.pkl")
    dsamp = DefocusSampler().load(f"{THIS_DIR}/test_data/defocus_sampler.pkl", validate=False)

    # Generate a sample for 5 particles
    n_particles = 5
    angles  = osamp.sample(n_particles)
    defocus = dsamp.sample(n_particles)

    # Quickly simulate N particles on memory
    sim = pSimulator.simulate_dataset(angles, defocus, quick=True)
    fig = sim.plot(show=show)
    plt.close(fig)
    sim.save(f"{THIS_DIR}/test_data/dataset_quick.pkl")

    # Large-scale simulation
    n_particles = 10
    n_cpus = 5
    chunk_size = 2

    # Load orientation and defocus samplers
    osamp = OrientationSampler().load(f"{THIS_DIR}/test_data/orientation_sampler.pkl")
    dsamp = DefocusSampler().load(f"{THIS_DIR}/test_data/defocus_sampler.pkl", validate=False)
    angles  = osamp.sample(n_particles)
    defocus = dsamp.sample(n_particles)

    # Simulate
    sim = pSimulator.simulate_dataset(
        angles,                       # Sampled orientations
        defocus,                      # Sampled defocus
        n_cpus=n_cpus,                # Parallelization
        chunk_size=chunk_size,        # Number of simulated particles per job
        output=f"{THIS_DIR}/test_data/sim_particles",  # Output directory
        intermediates=True,           # To be more efficient, set intermediates to FALSE.
        use_eman=False
    )
    fig = sim.plot(show=show)
    plt.close(fig)
    sim.save(f"{THIS_DIR}/test_data/dataset_full.pkl")

    # Simulate using eman
    sim = pSimulator.simulate_dataset(
        angles,                       # Sampled orientations
        defocus,                      # Sampled defocus
        n_cpus=n_cpus,                # Parallelization
        chunk_size=chunk_size,        # Number of simulated particles per job
        output=f"{THIS_DIR}/test_data/sim_particles", # Output directory
        intermediates=True,           # To be more efficient, set intermediates to FALSE.
        use_eman=True
    )
    fig = sim.plot(show=show)
    plt.close(fig)
    sim.save(f"{THIS_DIR}/test_data/dataset_full.pkl")

    # Storing/Loading ParticleSimulator
    pSimulator.save(f"{THIS_DIR}/test_data/simulator.pkl")
    pSimulator = ParticleSimulator().load(f"{THIS_DIR}/test_data/simulator.pkl")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, but safe
    multiprocessing.set_start_method("fork")
    main()