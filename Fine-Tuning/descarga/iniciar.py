import data_set  as ds
import tensorflow_datasets as tfds

#### Descarga del Dataset para entrenamiento ####
# DATASET: https://huggingface.co/datasets/viewer/?dataset=mlsum
#### Nos permite iniciar la libreria proporcionada por HuggingFace ####
# https://github.com/huggingface/datasets/blob/master/datasets/mlsum/mlsum.py

path = "data"
dm = tfds.download.DownloadManager(download_dir=path)
b = ds.Mlsum.BUILDER_CONFIGS


ds.Mlsum._split_generators(b, dm)