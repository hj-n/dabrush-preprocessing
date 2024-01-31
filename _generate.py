import numpy as np
import os, json
from _snnknn import KnnSnn as ks 
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix




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




def generate(spec):
	"""
  Main function of `_generate.py`
	Get specification as input and computes the preprocessed file to run brushing techniques
	"""


	DEFAULT_SAMPLING = 1.0

	### Read the specification


	
	keys = spec.keys()

	### Read and sample the dataset
	data, labels = read_dataset(spec["dataset"])








