import argparse, json, os

import _helpers as hp


parser = argparse.ArgumentParser(description='Run preprocessing for distortion-aware brushing')
parser.add_argument("--spec", "-s", type=str, help="ID of the specification to run preprocessing")

args = parser.parse_args()
spec_id = args.spec


"""
Read specification 
"""

if os.path.exists(f"./specs/{spec_id}.json"):
	with open(f"./specs/{spec_id}.json", "r") as f:
		spec = json.load(f)
else:
	raise Exception(f"Specification not found: {spec_id}")

"""
RUN SANITY CHECK
"""

REQUIRED_KEYS = ["dataset", "projection", "techniques"] 
OPTIONAL_KEYS = ["labels", "sampling_rate", "distance", "max_neighbors"]

#### 1. Check if there exists non-approved keys
keys = spec.keys()
for key in keys:
	if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
		raise Exception(f"Invalid key in the specification: {key}")

#### 2. Check if there exists required keys
for key in REQUIRED_KEYS:
	if key not in keys:
		raise Exception(f"Required key not found in the specification: {key}")


"""
READ A DATASET and LABELS (if exists)
Then sample the dataset based on the sepcification
"""

data, labels = hp.read_dataset(spec["dataset"])
data, labels = hp.sample_dataset(data, labels, spec)

