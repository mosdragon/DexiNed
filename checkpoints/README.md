# Model Weights for DexiNed

## Raw Model Weights
Download the following required files to run `python dexined_network.py`:
```
gsutil -m cp -r \
gs://ds-osama/postprocess/dexined/DXN_BIPED/ .
```
* train_1 is the perceptually better model, and this model's results
were showcased at the WACV 2020 conference.
* train_2 is quantitatively better, but it's fine-tuned on a separate
dataset.


## Frozen Graph Models
There will be two frozen graph files to choose from, both of which are generated
by running `python dexined_network.py`. They can also just be pulled from the
GCP buckets via the command below:
```
gsutil -m cp -r \
gs://ds-osama/postprocess/dexined/dexined_frozen_graph_v* .
```

## Inference Using Frozen Graph Models.
From `<PROJECT-ROOT>` you can do the following:

```python
from dexinded_edges import get_dexined_edges

# Read in an RGB image as a numpy array.
img_uri = "example/living_room.png"
img = imread(img_uri, pilmode="RGB")
img = np.asarray(img)

# Produce the edgemap. This helper function runs the image through the network.
edgemap = get_dexined_edges(img)
```
