import argparse, json, os

import _helpers as hp
import _projection as prj
import _preprocess as pp

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
print("STEP 1: SANITY CHECK")
print("#### Running sanity check for the specification...")

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


print("#### Sanity check passed!")
print()

"""
READ A DATASET and LABELS (if exists)
Then sample the dataset based on the sepcification
"""

print("STEP 2: READ and SAMPLE DATASET")
print("#### Reading and Sampling the dataset...")

data, labels, directory = hp.read_dataset(spec["dataset"])
data, labels = hp.sample_dataset(data, labels, spec)

print("#### Dataset reading and sampling finished!")
if labels is not None:
	print(f"#### - size # / dim # / label #:  {data.shape[0]} / {data.shape[1]} / {len(set(labels))}")
else:
	print("#### - size # / dim #: ", data.shape[0], data.shape[1])
print()

"""
Generate projection if not exists
"""

print("STEP 3: GENERATE A LOW-D PROJECTION")
print("#### Checking whether projection exists...")
ld = prj.check_and_generate_projection(data, spec, directory)
print()

"""
Preprocess the data, generating required information to run brushing techniques
"""

print("STEP 4: PREPROCESSING")

print("#### Start preprocessing...")
preprocessed = pp.preprocess(data, ld, labels, spec)

print("#### Preprocessing finished! Saving preprocessed data...")
with open(f"./preprocessed/{spec_id}_preprocessed.json", "w") as f:
	json.dump(preprocessed, f)

print("\nALL SET!!")