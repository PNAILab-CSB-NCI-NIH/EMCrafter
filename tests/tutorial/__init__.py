import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

BASE_PATH = f"{THIS_DIR}/../.."
TUTO_PATH = f"{BASE_PATH}/TUTORIAL"
SRC_PATH  = f"{BASE_PATH}/src"

sys.path.append(SRC_PATH)