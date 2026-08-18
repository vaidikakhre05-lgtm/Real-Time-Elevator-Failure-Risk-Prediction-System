import os
import json
from pathlib import Path
from typing import Dict, Any

def create_directory_structure(root_path: Path) -> None:
    """
    Creates the standardized project folder structure under the specified root path.
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/synthetic",
        "notebooks",
        "reports/figures",
        "reports/metrics",
        "models/classifier",
        "models/regressor",
        "logs",
        "assets/images",
        "assets/icons",
        "src/data",
        "src/models",
        "src/visualization",
        "src/simulation",
        "src/utils",
        "streamlit_app/pages",
        "streamlit_app/components",
        "tests"
    ]
    for directory in directories:
        (root_path / directory).mkdir(parents=True, exist_ok=True)
        # Create empty .gitkeep in each directory
        gitkeep_file = root_path / directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Saves a dictionary as a JSON file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Loads a JSON file as a dictionary.
    """
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
