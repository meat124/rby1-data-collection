import importlib.util
import sys
from pathlib import Path


def ensure_local_rby1_sdk_path() -> str | None:
    existing_spec = importlib.util.find_spec("rby1_sdk")
    if existing_spec is not None and existing_spec.origin is not None:
        return str(Path(existing_spec.origin).resolve().parent.parent)

    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir.parent / "rby1-sdk",
        current_dir.parent / "rby1-sdk" / "python",
        current_dir / "rby1-sdk" / "python",
    ]

    for candidate in candidates:
        package_dir = candidate / "rby1_sdk"
        has_bindings = package_dir.is_dir() and any(package_dir.glob("_bindings*"))
        if has_bindings:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str

    return None