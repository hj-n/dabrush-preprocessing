import numpy as np
import os
from _snnknn import KnnSnn as ks 



def reader(dataset_name, is_example=True):
	"""
	read file from the path
	"""
	if is_example:
		data_path = f"./example_datasets/{dataset_name}/data.npy"
		label_path = f"./example_datasets/{dataset_name}/label.npy"
	else:
		data_path = f"./datasets/{dataset_name}/data.npy"
		label_path = f"./datasets/{dataset_name}/label.npy"

	data = None
	labels = None
	
	if not os.path.exists(data_path):
		raise Exception(f"File not found: {data_path}")
	else:
		data = np.load(data_path).astype(np.float32)

	if os.path.exists(label_path):
		labels = np.load(label_path).astype(np.int32)

	return data, labels
	

