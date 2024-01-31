import os
import numpy as np

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

def sample_dataset(data, labels, spec):

	DEFAULT_SAMPLING = 1.0

	keys = spec.keys()

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

	return data, labels