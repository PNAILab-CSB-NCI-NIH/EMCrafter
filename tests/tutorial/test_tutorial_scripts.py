import subprocess
import pytest
from pathlib import Path

# Path to the scripts directory
SCRIPTS_DIR = Path(__file__).parent

# List of script files to test
SCRIPTS = [
    f"{SCRIPTS_DIR}/1-OrientationSampling.py",
    f"{SCRIPTS_DIR}/2-DefocusSampling.py",
    f"{SCRIPTS_DIR}/3-NoiseShape.py",
    f"{SCRIPTS_DIR}/4-NoiseFrequency.py",
    f"{SCRIPTS_DIR}/5-FullNoiseGenerator.py",
    f"{SCRIPTS_DIR}/6-SignalNoiseRatio.py",
    f"{SCRIPTS_DIR}/7-SignalGeneration.py",
    f"{SCRIPTS_DIR}/8-FullSimulation.py",
    f"{SCRIPTS_DIR}/9-ExportSTAR.py",
]

@pytest.mark.parametrize("script", SCRIPTS)
def test_script_execution(script):
    path = Path(script)
    if not path.exists():
        pytest.skip(f"Script not found: {script}")

    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("=== STDERR ===")
        print(result.stderr)
        print("=== STDOUT ===")
        print(result.stdout)

    assert result.returncode == 0, f"Script failed: {script}"