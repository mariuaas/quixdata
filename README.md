# litdata
Local Indexed Tar Dataset is a minimal wrapper for sharded tar-datasets with multiple modalities based on the WebDataset format.
Currently maintained and used by the DSB group at the University of Oslo.

## The TLDR;
We utilize the specifications of WebDataset to make a simple way of training with sharded datasets
using more standard PyTorch conventions. This allows entry-level users to work with locally hosted sharded 
datasets without much hassle, and allows us to better maintain an increasing zoo of dataset formats
over multiple HPC resources. 

## The What
litdata contains the LITDataset class which handles locally stored datasets 
using .tar shards, e.g. the WebDataset ([WDS](https://webdataset.github.io/webdataset/)) format. 
The behaviour is a mix of standard `torch` dataset implementations, and we borrow the API 
for applying mappings from WebDataset after dataset initialization, (e.g., `map`, `map_tuple`). 

## The Why
Designed to simplify some of the implementation choices used in WDS for online hosting, 
such as missing length (`__len__`) and buffered sampling. These can be an issue for locally 
hosted datasets. These quirks make WDS slightly less attractive for entry-level implementation.

LITDataset allows the data to be shuffled / batched / etc. using native DataLoader classes. 
This also improves shuffling behaviour for tasks which require high stochasticity, such as contrastive learning.


## The How
Instead of sequential iteration over shards, which is unnecessary if files are hosted locally, 
this format creates an index of the byte offsets of all files in the shards, which can be serialized
for faster subsequent initialisation. These indices are then concatenated by taking the union over 
elements matching the supplied extensions. 

### Config Files
LITDataset relies on a supplied config file, formatted as a JSON which contains info on training / validation
folds, default extensions. Typically this is formatted as follows (example from ImageNet1k):
```
{'train': '/train_{0000..0071}.tar',
 'val': '/val_{0000..0003}.tar',
 'extensions': ['jpg', 'cls'],
 'metadata': {'num_classes': 1000,
  'num_train': 1281167,
  'num_val': 50000,
  'website': 'https://www.image-net.org/'}}
```
The shards are listed using brace expansion. In addition, the config file includes a set of (default) extensions.
These can be overrided in the initialisation. In addition, LITDataset allows for customizable decoders for different
extensions, which can be provided using the `override_decoders` argument. 

For now, the default decoders use `PIL` for image formats, and `cls` for classification labels. 
Work is in progress to expand the default decoders to multiple classes and modalities, including
text, bounding boxes, segmentation, audio, and more.

Currently, the config file is required to be in the directory of the dataset, and 
defaults to `config.json`, but can be specified. 

### Offset indexing
Looking up names in a tarfile is a bit inefficient for large shards. Instead, the LITDataset looks up the offsets
for each file in all shards, and generates an index. If these are not provided, this is generated on the fly, but
can be serialized for faster subsequent initialization. This allows uncompressed tar shards to be quickly accessed
int the `__getitem__` method.


## TODOS:

1. Better support for (default) decoders.
2. Make a module of writers / sharders for easy generation of LITDatasets.
3. Improve documentation.
