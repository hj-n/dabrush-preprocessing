import umap
import os
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def generate_projection(data, spec, directory):
	"""
	Generate projection based on the projection name
	"""
	projection = spec["projection"]
	dataset = spec["dataset"]
	if projection == "umap":
		ld = umap.UMAP().fit_transform(data)
	elif projection == "tsne":
		ld = TSNE(n_components=2).fit_transform(data)
	elif projection == "pca":
		ld = PCA(n_components=2).fit_transform(data)
	else:
		raise Exception(f"Generation failed due to invalid projection name: {projection}")

	np.save(f"./{directory}/{dataset}/{projection}.npy", ld)

	print("#### Projection generated!!")



def check_and_generate_projection(data, spec, directory):
	projection = spec["projection"]
	dataset = spec["dataset"]

	### check whehter projection exists in the directory
	if not os.path.exists(f"./{directory}/{dataset}/{projection}.npy"):
		print(f"#### Projection not found, generating projection...")
		projection = generate_projection(data, spec, directory)
	else:
		print(f"#### Projection found, skipping projection generation.")
	
	return