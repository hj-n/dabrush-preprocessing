## Preprocessing & Test code for 2024 Distortion-Aware Brushing

- This repository provides the preprocessing code for Distortion-Aware Brushing
- Also provides the preprocessing code for alternative MD brushing techniques: Similarity Brushing, M-Ball Brushing, Data-Driven Brushing

## Quick run with example dataset

1. Clone the repository
2. Install dependencies (TBA)
3. Run the script using the example dataset (fashion_mnist, sampled with UMAP projection)
```sh
python3 _run.py -s fmnist_small_umap
```

### Functions for preprocessing

- `_snnknn.py`: File providing functions for computing SNN-based similarity measure (KNN + SNN)
  - imported from https://github.com/hj-n/gpu-knn-snn-graph

- `_projection.py`: this script generates dimensionality reduction projections (i.e., multidimensional projections). Note that you can also use your own custom projections
- Arguments

- `_preprocess.py`: this script finally generates preprocessed files

- `_helpers.py`: this script provides helper functions for preprocessing

- `_run.py`: THIS SCRIPT IS THE FILE THAT USED SHOULD RUN.
  - The script will automatically...
	  1. Run sanity check for the specification
	  2. read and sample data based on specification
		3. Check whether the projection exists and if not, generate
		4. Run preprocessing and save the preprocessed file as JSON


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

- `specs/`: Place your specification for generating distortion-aware data here
  - the name of the file should be identical to the id of the specification

- `preprocessed/`: Place for your preprocessed file to be saved
  - the name of the file will be identical to the id of the specification, but will ends with `_preprocessed.json`

### Specifications

Specifications should be declared to generate the preprocessed data for the brushing techniques.  Specification should be a `.json` file that is consists of a single dictionary, containing following fields:

- `dataset` (REQUIRED): name of the dataset (e.g., "fashion_mnist")
  - `_generate.py` file automatically searches whether the dataset exists in `example_datasets/` or `datasets/` directory
- `projection` (REQUIRED): option to generate multidimensional projections
  - the str provided here is interpreted as a file name of the projected data
	- e.g., if `"umap"` is given, the system will read the file `umap.npy` in the same directory
	- if the given name is not found, the system will generate the projection if the available projection methods are provided
	  - currently supported: `umap`, `tsne`, `pca`
- `techniques` (REQUIRED): List of the brushing techniques to be applied
  - currently supported:
	  - `dab`: Distortion-Aware Brushing
	- will be implemented:
		- `sb`: Similarity brushing (Novotny and Hauser)
		- `mbb`: M-Ball Brushing (Aupetit et al.)
		- `ddb`: Data-Driven Brushing (Martin and Ward)
  - e.g., `["dab, "sb", "mbb"]`,
- `labels`: List of the labels to be sampled
	- e.g., `[0, 2, 5, 6]`
	- default: `None` (all labels)
- `sampling_rate`: percentage of the data to be sampled
	- e.g., `0.1` for 10% of the data
	- default: `1.0` (100%)
- `distance`: the specification that contains the info of the distance function used for distortion-aware brushing and similarity brushing
  - detailed specifiaction: 
    > ```json
		> {
		> 	`metric`: `euclidean` | `cosine` | `snn`,
		> 	`params`: {
		> 		`k`: // number of nearest neighbors to be used for computing SNN
		>   }
		> }
		> ```
	- default: `snn` with `k=10`
- `max_neighbors`: maximum number of neighbors to be considered for distortion-aware brushing
	- default: 100 
		



### Test code for SNN-based similarity measure

In the current (2024) version of distortion-aware brushing,