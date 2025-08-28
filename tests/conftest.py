import sys
from pathlib import Path

# Ensure project root is on sys.path so `owl_wms` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure owl-vaes submodule is importable as `owl_vaes`
OWL_VAES_ROOT = PROJECT_ROOT / "owl-vaes"
if OWL_VAES_ROOT.exists():
    vaes_path = str(OWL_VAES_ROOT)
    if vaes_path not in sys.path:
        sys.path.insert(0, vaes_path)


