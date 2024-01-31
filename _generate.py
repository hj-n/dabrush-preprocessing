import numpy as np
import os, json
from _snnknn import KnnSnn as ks 
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix



def read_dataset(dataset_name):
	"""
	read file from the path
	"""

	if os.path.exists(f"./example_datasets/{dataset_name}/data.npy"):
		data_path = f"./example_datasets/{dataset_name}/data.npy"
		label_path = f"./example_datasets/{dataset_name}/label.npy"
	elif os.path.exists(f"./datasets/{dataset_name}/data.npy"):
		data_path = f"./datasets/{dataset_name}/data.npy"
		label_path = f"./datasets/{dataset_name}/label.npy"
	else:
		raise Exception(f"Dataset not found in either example_datasets/ and datasets/ directory: {dataset_name}")

	data = None
	labels = None
	
	data = np.load(data_path).astype(np.float32)
	if os.path.exists(label_path):
		labels = np.load(label_path).astype(np.int32)

	return data, labels


def knn(data, param):
	"""
	Compute distance matrix based on k-nearest neighbor (KNN)
	"""
	DEFAULT_K = 10

	k = param["k"] if "k" in param.keys() else DEFAULT_K

	kSnn = ks(k)
	knn_results = kSnn.knn(data)

	return knn_results


def snn(data, knn_results, param):
	"""
	Compute distance matrix based on shared nearest neighbor (SNN)
	"""
	DEFAULT_K = 10

	k = param["k"] if "k" in param.keys() else DEFAULT_K
	kSnn = ks(k)
	snn_results = kSnn.snn(knn_results)

	## normalize snn
	snn_results /= np.max(snn_results)

	return snn_results

def compute_distance(data, distance):
	"""
	Parse distance dictionary and 
	Computer distance matrix from the hd data
	"""
	metric = distance["metric"]
	param = distance["param"] if "param" in distance.keys() else None

	if metric == "euclidean" or metric == "cosine":
		dist_matrix = cdist(data, data, metric=metric)
	elif metric == "snn":
		knn_results = knn(data, param=param)
		dist_matrix = snn(data, knn_results, param=param)

	return dist_matrix




def generate(spec_id):
	"""
  Main function of `_generate.py`
	Get specification as input and computes the preprocessed file to run brushing techniques
	"""

	REQUIRED_KEYS = ["dataset", "techniques"]
	OPTIONAL_KEYS = ["labels", "sampling_rate", "distance", "max_neighbors"]

	DEFAULT_SAMPLING = 1.0

	### Read the specification

	if os.path.exists(f"./specs/{spec_id}.json"):
		with open(f"./specs/{spec_id}.json", "r") as f:
			spec = json.load(f)
	else:
		raise Exception(f"Specification not found: {spec_id}")
	
	### Sanity checks for the specification

	#### 1. Check if there exists non-approved keys
	keys = spec.keys()
	for key in keys:
		if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
			raise Exception(f"Invalid key in the specification: {key}")

	#### 2. Check if there exists required keys
	for key in REQUIRED_KEYS:
		if key not in keys:
			raise Exception(f"Required key not found in the specification: {key}")

	### Read and sample the dataset
	data, labels = read_dataset(spec["dataset"])

	#### Sample by label
	if "labels" in keys:
		selected_label_indices = spec["labels"]
	else:
		selected_label_indices = list(set(labels))
	filterer = np.isin(labels, selected_label_indices)
	data = data[filterer]
	labels = labels[filterer]

	#### Sample by sampling rate
	if "sampling_rate" in keys:
		sampling_rate = spec["sampling_rate"]
	else:
		sampling_rate = DEFAULT_SAMPLING
	
	sample_size = int(data.shape[0] * sampling_rate)
	sampled_indices = np.random.choice(data.shape[0], sample_size, replace=False)
	data = data[sampled_indices]
	labels = labels[sampled_indices]







