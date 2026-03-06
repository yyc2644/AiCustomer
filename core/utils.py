import yaml
import json
from pathlib import Path

def load_config(file_path=None):
    if file_path is None:
        file_path = Path(__file__).parent.parent / "config" / "systems.yaml"
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_testcases(file_name="sample_faq.json"):
    file_path = Path(__file__).parent.parent / "config" / "testcases" / file_name
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
