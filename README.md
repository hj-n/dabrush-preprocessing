## Preprocessing & Test code for 2024 Distortion-Aware Brushing


### Functions for preprocessing

- `_snnknn.py`: File providing functions for computing SNN-based similarity measure (KNN + SNN)
  - imported from https://github.com/hj-n/gpu-knn-snn-graph

- `_generate.py`: Run this script to 

### File directories

- `example_datasets/`: Provides multiple example datasets for testing distortion-aware brushing
  - All files are in `.npy` format
	- currently supports: `fashion_mnist`, `dry_bean`

### Test code for SNN-based similarity measure

In the current (2024) version of distortion-aware brushing,