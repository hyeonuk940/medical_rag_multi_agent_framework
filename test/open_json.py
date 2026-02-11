import json
import os
from pprint import pprint

def check_medical_json(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print({list(data.keys())})
        pprint(data, indent=2, sort_dicts=False)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")

sample_file_path = 'data/datasets/TL_내과'

