## Preprocessing & Test code for 2024 Distortion-Aware Brushing


### Functions for preprocessing

- `_snnknn.py`: File providing functions for computing SNN-based similarity measure (KNN + SNN)
  - imported from https://github.com/hj-n/gpu-knn-snn-graph

- `_generate.py`: Run this script to 

### File directories

- `example_datasets/`: Provides multiple example datasets for testing distortion-aware brushing
  - All files are in `.npy` format
	- currently supports: `fashion_mnist`, `dry_bean`

- `datasets/`: Place your own datasets here
	- All files should be in `.npy` format
	- The file containing the data should be named `data.npy`
	- The file containing the labels should be named `label.npy`
	- The files should be placed in the directory named after the dataset name
	  - e.g., file path should be `datasets/dataset_name/data.npy` and `datasets/dataset_name/label.npy`
	- the data file should also be able to be represented in `float` data type (e.g., `np.float32`)
	- the label file should also be able to be represented in `int` data type (e.g., `np.int32`)

### Test code for SNN-based similarity measure

In the current (2024) version of distortion-aware brushing,