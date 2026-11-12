
from pathlib import Path

import pandas as pd

import yaml

# ingestion.py -> data -> aislca -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path=None) -> dict : 
    """Read the YAML config file and return it as a dictionary."""

    path = Path(config_path) if config_path else CONFIG_PATH 
    with open(path,"r",encoding="utf-8") as file :
        return yaml.safe_load(file)


def load_student_data(subject : str = 'mat', config:dict = None) -> pd.DataFrame:
    """Load one raw dataset.

    subject: "mat" for Math or "por" for Portuguese.
    """
    if subject not in ("mat", "por"):
        raise ValueError("subject must be 'mat' or 'por'")

    config = config or load_config()
    data_cfg = config["data"]

    file_path = PROJECT_ROOT / data_cfg["raw_dir"] / data_cfg["raw_files"][subject]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path, sep=data_cfg["separator"])



def load_processed_data(subject: str = "mat", config: dict = None) -> pd.DataFrame:

    """Load a cleaned dataset from data/processed/."""
    config = config or load_config()
    path = PROJECT_ROOT / config["data"]["processed_dir"] / f"student_{subject}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {path}. Run: python -m aislca.data.cleaning"
        )
    return pd.read_csv(path)
