import numpy as np
import os
from _snnknn import KnnSnn as ks 



def reader(dataset_name):
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
		raise Exception(f"File not found: {dataset_name}")

	data = None
	labels = None
	
	data = np.load(data_path).astype(np.float32)
	if os.path.exists(label_path):
		labels = np.load(label_path).astype(np.int32)

	return data, labels
	

