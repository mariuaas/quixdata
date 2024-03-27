'''
QuixData
========

This module provides a comprehensive suite for handling, encoding, and writing datasets 
specifically tailored for the Quix framework. It combines functionalities from 
several submodules to offer a streamlined experience for dataset manipulation, 
including dataset creation, encoding/decoding of various data types, and efficient 
data storage and retrieval.

The module includes the following components:

- Dataset Handling: Classes for creating and manipulating datasets, including
  `QuixDataset` and `QuixUnionDataset`, which offer flexible options for dataset
  representation and access.

- Encoding/Decoding: A collection of encoder/decoder classes, such as `EncoderDecoder`,
  `CLS`, `PIL`, `NPY`, `RLE`, `SEG8`, `SEG16`, `SEG24`, `SEG32`, and default encoder
  and decoder sets. These are designed to handle various data types and formats,
  making it easier to work with complex datasets.

- Writing Utilities: Classes like `IndexedTarWriter`, `IndexedShardWriter`, and
  `QuixWriter` provide efficient mechanisms for writing data to disk in a structured
  and indexed manner, facilitating the creation of large-scale datasets.

- Augmentation Parsing: Functions `parse_train_augs` and `parse_val_augs` help in
  parsing and applying augmentations to the datasets during training and validation.

Together, these components form a versatile toolkit for dataset management within the
Quix framework, catering to a wide range of data handling and processing needs.

Example Usage
-------------
Creating and manipulating a QuixDataset:

```python
from quix_data_module import QuixDataset

# Initialize a QuixDataset instance
dataset = QuixDataset(...)

# Access and use the dataset
for data in dataset:
    # Process data
```

Encoding data and writing to a dataset:

```python
from quix_data_module import QuixWriter, PIL, DEFAULT_ENCODERS

# Initialize a QuixWriter
writer = QuixWriter(...)

# Use the writer to store data
writer.train.write({'__key__': 'sample1', 'image.jpg': image_data})

# Finalize and close the writer
writer.close()
```

This module simplifies and streamlines data handling processes, aiding in the development
and management of large-scale datasets for various applications within the Quix framework.


Author
------
Marius Aasan <mariuaas@ifi.uio.no>

Contributors
------------
Please help by considering contributing to the project.
'''
from typing import Generic
from .dataset import QuixDataset, QuixUnionDataset
from .encoders import (
    EncoderDecoder, CLS, PIL, NPY, RLE, SEG8, SEG16, SEG32, SEG24, DEFAULT_DECODERS, DEFAULT_ENCODERS
)
from .writer import IndexedTarWriter, IndexedShardWriter, QuixWriter