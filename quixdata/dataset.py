'''
QuixDataset
===========

This module contains the QuixDataset and QuixUnionDataset classes, which are 
designed for efficient handling of locally stored datasets using .tar shards, 
such as those in the WebDataset (WDS) format. The implementation aims to 
simplify and optimize the usage of WDS for local datasets.

The QuixDataset class offers an improved approach for handling datasets stored 
in .tar format, addressing challenges such as the absence of dataset length and 
buffered sampling, common in online hosting scenarios. This class allows for 
flexible data manipulation, including shuffling and batching, using standard 
PyTorch DataLoader classes. It is particularly beneficial for tasks requiring 
high stochasticity, like contrastive learning.

The QuixUnionDataset class extends this functionality by allowing simultaneous 
sampling from multiple QuixDataset instances in a round-robin manner. It ensures 
that all datasets appear to be of the same length, aligning with the length of 
the longest dataset through a modulo operation.

Extended Summary
----------------
Instead of the sequential iteration over shards typically used in web-based 
datasets, the QuixDataset class creates an index of byte offsets for all files 
within the shards. This index can be serialized, enabling faster initialization 
in subsequent uses. The dataset items are then accessed directly through these 
offsets, significantly improving data access speed and flexibility.

The QuixUnionDataset class concatenates multiple QuixDataset instances, 
facilitating experiments and training scenarios that might require combining 
data from multiple sources or formats. Its round-robin sampling strategy ensures 
an even distribution of data from each dataset in the union.

Both classes are optimized for local filesystems with high-speed access, making 
them well-suited for scenarios where datasets are not streamed from the web but 
are available locally.

Classes
-------
- QuixDataset : Handles locally stored datasets using .tar shards.
- QuixUnionDataset : Concatenates multiple QuixDataset objects for simultaneous sampling.
- MapTuple : Applies a tuple of functions to corresponding items in a dataset.
- MapAll : Applies a single function to all items in a dataset.

Author
------
Marius Aasan <mariuaas@ifi.uio.no>
'''

from __future__ import annotations
import re
import json
import tarfile
import os
import inspect
import random
import copy
import pkg_resources
import torch
import torch.nn as nn
import torchvision
from contextlib import contextmanager
from torch.utils.data import Dataset

from typing import (
    Tuple, Dict, Iterable, List, Callable, Union, TypeVar, Optional, 
    Any, Sequence
)

from .encoders import DEFAULT_DECODERS

# Add flag for using TV Tensors 
# TODO: Add support for 0.15 datapoints
USE_TV_TENSOR = pkg_resources.parse_version(torchvision.__version__) >= pkg_resources.parse_version('0.16')

class MapTuple:

    def __init__(self, maps):
        assert all([isinstance(m, Callable) for m in maps])
        #assert all([len(inspect.signature(m).parameters) == 1 for m in maps]) # Ignore this assertion for now
        self.maps = maps
    
    def fnapply(self, fn, x):
        return fn(x)

    def __call__(self, y):
        return list(map(self.fnapply, self.maps, y))
    

class MapAll:
    def __init__(self, mapping):
        assert isinstance(mapping, Callable)
        #assert len(inspect.signature(mapping).parameters) == 1 # Ignore this assertion for now
        self.mapping = mapping

    def __call__(self, y):
        return list(map(self.mapping, y))


class QuixDataset(Dataset):
    """Quix Local Indexed Tar Dataset.

    A dataset class for handling locally stored datasets using tar shards. It simplifies
    some of the implementation choices used in the WebDataset format, which is primarily
    designed for streaming from online hosting. This class enables the dataset to be 
    shuffled, batched, and manipulated using native DataLoader classes in PyTorch.

    Attributes
    ----------
    fold : str
        Indicates whether the dataset is for training ('train') or validation ('val').
    root : str
        Root directory of the dataset.
    use_extensions : Tuple[str, ...]
        Extensions of the files to be used in the dataset.
    decoders : Tuple[Callable, ...]
        Decoders for each file extension in use_extensions.
    shard_list : Tuple[str, ...]
        List of paths to tar shards making up the dataset.
    shard_index : Dict[str, Dict[str, int]]
        Index mapping each file in the tar shards to its offset.
    offset_index : List[Tuple[str, int]]
        List of tuples containing shard names and offsets for each dataset item.
    transforms : List[Callable]
        List of transformations to be applied to each dataset item.

    Parameters
    ----------
    dataset : str
        Name of the dataset, expected to be a subfolder of `loc`.
    loc : str
        Location of the dataset.
    train : bool, optional
        Whether to use training data. Defaults to True.
    override_extensions : Optional[Tuple[str, ...]], optional
        Extensions to be used instead of those defined in the dataset's config.
    override_decoders : Optional[Iterable[Tuple[str, Callable]]], optional
        Custom decoders for specific extensions.
    config_fname : str, optional
        Name of the dataset configuration file. Defaults to 'config.json'.
    train_idx_fname : str, optional
        Name of the training index file. Defaults to 'train.idx.json'.
    val_idx_fname : str, optional
        Name of the validation index file. Defaults to 'val.idx.json'.
    serialize : bool, optional
        If True, serializes the dataset indices. Defaults to False.
    mod_length : int, optional
        Artificial length for the dataset, for modulo sampling. Defaults to -1 (not used).
    deterministic_offset_order : bool, optional
        Ensures deterministic ordering of dataset offsets. Defaults to True.
    debug : bool, optional
        Enables debug prints during dataset loading. Defaults to False.

    Methods
    -------
    shufflecontext()
        Context manager to shuffle data with respect to shards.
    __len__()
        Returns the length of the dataset.
    __getitem__(key)
        Retrieves a dataset item by its key.
    __repr__()
        Returns a string representation of the dataset object.
    braceexpand(input_string, root)
        Expands a brace-formatted string to generate a list of paths.
    map_tuple(*maps)
        Applies given mappings to individual extensions of dataset items.
    map_all(mapping)
        Applies a given mapping to all extensions of dataset items.
    map(mapping)
        Applies a given mapping to the tuple of extensions of dataset items.
    set_decoders(override)
        Sets the decoders for the dataset extensions.
    generate_shard_index()
        Generates an index of files and their offsets within each tar shard.
    generate_offset_index()
        Generates an index of offsets for each dataset item.

    Notes
    -----
    - The class returns selected extensions in the order they are passed in
        override_extensions. For image extensions, the loader will return a PIL Image object.
    - TODO: Add option to pass `idx` as extension to retrieve sample indices.
    - TODO: Add option to pass `name` as extension to retrieve sample names.

    Examples
    --------
    Creating a QuixDataset instance for a training set:

    >>> dataset = QuixDataset(
            dataset='example_dataset',
            loc='/path/to/dataset',
            train=True,
            override_extensions=('.jpg', '.txt')
        )
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in dataloader:
    ...     # Process batch

    Using the shuffle context:

    >>> with dataset.shufflecontext():
    ...     dataloader = DataLoader(dataset, batch_size=32)
    ...     for batch in dataloader:
    ...         # Process batch
    """
    def __init__(
        self, 
        dataset:str, 
        loc:str, 
        train:bool=True, 
        override_extensions:Optional[Sequence[str]]=None,
        override_decoders:Optional[Sequence[Tuple[str, Callable]]]=None,
        config_fname:str='config.json',
        train_idx_fname:str='train.idx.json',
        val_idx_fname:str='val.idx.json',
        serialize:bool=False,
        mod_length:int=-1,
        deterministic_offset_order=True,
        debug:bool=False
    ):
        '''Initializes the QuixDataset
        
        NOTE: This class will return the selected extensions in the order
              they are passed in use_extensions. For image extensions, the loader 
              will return a PIL Image object. Work on support for other extensions is
              ongoing.
                      
        Parameters
        ----------
        dataset : str
            Name of the webdataset. Should be a subfolder of loc.
        loc : str
            Locations of the webdataset.
        train : Optional[bool]
            Whether to use training or validation. Default is True.
        override_extensions : Optional[Sequence[str]]
            Extension overrides.
        override_decoders : Optional[Sequence[Tuple[str, Callable]]]
            Decoder overrides.
        config_fname : str
            The config file name to use, default to 'config.json'.
        train_idx_fname : str
            The train index file name to use, default to 'train.idx.json'.
        val_idx_fname : str
            The validation file name to use, default to 'val.idx.json'.
        serialize : bool
            If True, serializes the offsets in the dataset folder.
        mod_length : int
            Set artificial length, sampling modulo for `key > mod_length`.
        deterministic_offset_order : bool
            Flag to ensure the offsets are loaded deterministically.
        debug : bool
            If True, prints to stdout during extraction of filenames.
        '''
        # Set initial parameters
        self.debug = debug
        self.fold = 'train' if train else 'val'
        self.root = os.path.join(loc, dataset)
        self.config_fname = config_fname
        self.index_fname = train_idx_fname if train else val_idx_fname
        self.mod_length = mod_length
                
        # Verify path, check for config / index files
        assert os.path.isdir(self.root)
        cfgpath = os.path.join(self.root, config_fname)
        assert os.path.isfile(cfgpath), f'Config at {cfgpath} is invalid!' 
        with open(cfgpath, 'r') as infile:
            self.cfg = json.load(infile)
                
        # Set extensions
        if override_extensions is not None:
            self.use_extensions = self._fixext(override_extensions)
        else:
            self.use_extensions = self._fixext(self.cfg['extensions'])
            
        # Set decoders
        self.decoders = self.set_decoders(override_decoders)
                
        # Generate shard indices
        self.shard_list = self.braceexpand(self.cfg[self.fold], self.root)
        
        # Check for serialized indices and load if exists
        idxpath = os.path.join(self.root, self.index_fname)
        if os.path.isfile(idxpath):
            with open(idxpath, 'r') as infile:
                shard_index = json.load(infile)
                
        # If serialized indices does not exist, generate from scratch.
        else:
            shard_index = self.generate_shard_index()        
            if serialize:
                with open(idxpath, 'w') as outfile:
                    json.dump(shard_index, outfile)
        
        # Save shard keys
        self.shard_keys = sorted(list(shard_index.keys()))
        
        # Generate offset index.
        self.offset_index = self.generate_offset_index(shard_index, deterministic_offset_order)
        self.offset_order = torch.arange(len(self.offset_index)).share_memory_()
        self.shard_bincount = self.offset_index[:,0].bincount()
        
        # Initialize transforms
        self.transforms = []

        # Set epoch counter and seed
        self.epoch = 0
        self.seed = 0

    
    @staticmethod
    def _shard_offset_key(x):
        return (x[0], min(x[1:]))
    
    @staticmethod
    def supports_tv_tensor():
        '''Checks if torchvision supports use of tv_tensors.

        Returns
        -------
        bool
            Whether torchvision version supports use of tv_tensors.
        '''
        return USE_TV_TENSOR

    @contextmanager
    def shufflecontext(self):
        """Context manager to shuffle data with respect to shards.

        Temporarily shuffles the order of the offset indices in the dataset. Indices
        from the same shard are grouped together, and the groups are randomized. This
        approach reduces disk I/O while providing randomness.

        Notes
        -----
        It is important to ensure that the DataLoader or whatever method you 
        are using to fetch the data respects the order of `self.offset_index`. 
        If it samples data points randomly from `self.offset_index`, then the 
        effort to reduce disk I/O by grouping together data points from the same 
        shard will be ineffective.

        Yields
        ------
        None
            Yields no value but modifies offset_index in place.

        Example
        -------
        >>> with dataset.shufflecontext():
        ...     dataloader = DataLoader(dataset)
        ...     for batch in dataloader:
        ...         # Processing code
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shard_offsets = self.shard_bincount.cumsum(-1) - self.shard_bincount
        shard_order = torch.randperm(len(self.shard_bincount), generator=g)
        new_order = torch.cat([
            torch.randperm(self.shard_bincount[i].item(), generator=g) + shard_offsets[i] #type:ignore
            for i in shard_order
        ])
        try:
            self.offset_order[:] = new_order
            yield
        finally:
            self.offset_order[:] = torch.arange(len(self.offset_index))
            self.epoch += 1

    def set_shuffle_epoch(self, epoch):
        '''Method to set the current epoch for shuffling using shufflecontext.

        Parameters
        ----------
        epoch : int
            Epoch to set for deterministic pseudorandom shuffling.
        '''
        self.epoch = epoch

    def set_shuffle_seed(self, seed):
        '''Method to set the seed for shuffling using shufflecontext.

        Parameters
        ----------
        seed : int
            Seed to set for deterministic pseudorandom shuffling.
        '''
        self.seed = seed
                        
    def __len__(self):
        return self.mod_length if self.mod_length > 0 else len(self.offset_index)
        
    def __getitem__(self, key):
        # Check modulo sampling
        key = key % len(self.offset_index) if self.mod_length > 0 else key
        order_index = self.offset_order[key]
        
        # Load keys, incl. shard name and extension offsets.
        key_tuple = self.offset_index[order_index]
        shard_name_idx, offsets = key_tuple[0], key_tuple[1:]
        shard_name = self.shard_keys[shard_name_idx]
        
        # Generate list for storing outputs
        out = [None]*len(offsets)
        
        # Load shard and retrieve files with extensions at offsets
        shard_path = os.path.join(self.root, shard_name)
        with tarfile.open(shard_path, 'r') as tar:
            for extidx in offsets.argsort():
                offset = offsets[extidx].item()
                tar.fileobj.seek(offset)                        # type: ignore
                member = tarfile.TarInfo.fromtarfile(tar)
                data = tar.extractfile(member).read()           # type: ignore
                out[extidx] = self.decoders[extidx](data)
        
        # Apply transformations
        if len(self.transforms) > 0:
            for t in self.transforms:
                out = t(out)
        
        # if len(out) == 1:
        #     return out[0]

        return tuple(out)
    
    def __repr__(self):
        fold = self.fold
        length = len(self)
        use_extensions = self.use_extensions
        metadata_str = ',\n'.join([f'\t{k} = {v}' for k,v in self.cfg['metadata'].items()])
        return (f'{self.__class__.__name__}(\n    ' +
            f'\tfold={fold},\n' +
            f'\tlength={length},\n' +
            f'\tuse_extensions={use_extensions},\n' +
            metadata_str + ',\n'
        ')')
    
    @staticmethod
    def braceexpand(input_string:str, root:str) -> Tuple[str,...]:
        """Performs brace expansion for a string given a root.

        Parameters
        ----------
        input_string : str
            Brace formatted string, e.g., '<pf><int>..<int><suf>'.
        root : str
            Root folder for the dataset.

        Returns
        -------
        Tuple[str, ...]
            Tuple of joined paths from brace expansion.
        """
        match = re.search(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', input_string)
        prefix, start, end, suffix = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4) # type: ignore
        num_digits = len(match.group(2)) # type: ignore
        format_str = f"{prefix}{{:0{num_digits}d}}{suffix}"
        return tuple([os.path.join(root, format_str.format(i)) for i in range(start, end + 1)])
    
    @staticmethod
    def _fixext(extensions: Sequence[str]) -> Sequence[str]:
        out = [f'.{e}' if not e.startswith('.') else e for e in extensions]
        return tuple(out)
            
    def map_tuple(self, *maps:Tuple[Callable,...]) -> QuixDataset:
        """Applies given mappings to individual extensions of dataset items.

        For `maps = [f1, f2]` and `extensions = ['jpg', 'cls'], this will return
        the tuple `(f1(<sample>.jpg), f2(<sample>.cls))`.

        Parameters
        ----------
        maps : Tuple[Callable, ...]
            Tuple of callables for mapping individual extensions.

        Returns
        -------
        QuixDataset
            The dataset itself with added transformations.
        """
        assert len(maps) == len(self.use_extensions)
        # Removed some assertions for now, will be moved to the init of the MapTuple class
        self.transforms.append(MapTuple(maps))
        return self
                
    def map_all(self, mapping:Callable) -> QuixDataset:
        '''Takes a mapping and applies it to all extensions.
        
        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `(f(<sample>.jpg), f(<sample>.cls))`.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to all extensions.

        Returns
        -------
        QuixDataset
            The dataset itself with added transformations.
        '''
        # assert len(inspect.signature(mapping).parameters) == 1
        # Removed some assertions for now, will be moved to the init of the MapAll class
        self.transforms.append(MapAll(mapping))
        return self
        
    def map(self, mapping:Callable) -> QuixDataset:
        '''Takes a mapping and applies it to the tuple of extensions.
        
        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `f(<sample>.jpg), f(<sample>.cls)`.

        Parameters
        ----------
        mapping : Callable
            Callable to be applied to the tuple of extensions.

        Returns
        -------
        QuixDataset
            The dataset itself with added transformations.
        '''
        # assert len(inspect.signature(mapping).parameters) == len(self.use_extensions)
        self.transforms.append(mapping)
        return self # type: ignore
    
    def set_decoders(
        self, override:Optional[Iterable[Tuple[str, Callable]]]
    ) -> Tuple[Callable,...]:
        """Sets the decoders for the dataset extensions.

        Parameters
        ----------
        override : Optional[Iterable[Tuple[str, Callable]]]
            Iterable of tuples mapping file extensions to custom decoders.

        Returns
        -------
        Tuple[Callable, ...]
            Tuple of decoders for the dataset extensions.
        """
        decoders:List[Any] = [None]*len(self.use_extensions)
        
        # Prefer override extensions
        if override is not None:
            for ext, dec in override:
                assert isinstance(dec, Callable), f'Decoder for {ext} not callable!'
                ext = self._fixext((ext,))[0]
                if ext in self.use_extensions:
                    idx = self.use_extensions.index(ext)
                    decoders[idx] = dec
        
        # Use default decoders
        for idx, ext in enumerate(self.use_extensions):
            head, wext = os.path.splitext(ext)
            wext = head if wext == '' else wext
            assert wext in DEFAULT_DECODERS, f"{wext=} not in default decoders."
            if decoders[idx] is None:
                decoders[idx] = DEFAULT_DECODERS[wext]
        
        return tuple(decoders)
            
    def generate_shard_index(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Generates an index of files and their offsets within each tar shard.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary mapping each file in the tar shards to its offset.
        """
        shard_index = {}
        
        for file in self.shard_list:
            assert os.path.isfile(file)

            shard_name = file.split("/")[-1]
            if self.debug:
                print(f'Extracting file names, extensions, and offsets for {shard_name}')
            
            with tarfile.open(file, 'r') as tar:
                shard_index[shard_name] = {}
                while True:
                    try:
                        member = tar.next()
                        if member is None:
                            break
                        if member.isfile():
                            name, extension = member.name.split('.')
                            extension = f'.{extension}'
                            offset = member.offset
                            if extension not in shard_index[shard_name]:
                                shard_index[shard_name][extension] = {name:offset}
                            else:
                                shard_index[shard_name][extension][name] = offset
                            
                    except StopIteration:
                        break
        
        return shard_index
    
    def generate_offset_index(
        self, 
        shard_index:Dict[str, Dict[str, Dict[str, int]]], 
        deterministic_offset_order:bool
    ) -> torch.Tensor:
        """
        Generates an index of offsets for each dataset item as a shared tensor.

        Parameters
        ----------
        shard_index : Dict[str, Dict[str, Dict[str, int]]]
            The shard index dictionary.
        deterministic_offset_order : bool
            Flag for setting deterministic offset ordering.

        Returns
        -------
        torch.Tensor
            Tensor containing shard key indices and offsets for each sample.
        """
        offsets = []

        for key in shard_index:
            # Generate list of samples which share the relevant extensions
            sets = []
            for extension in self.use_extensions:
                if extension in shard_index[key]:
                    sets.append(set(shard_index[key][extension]))
            
            if not sets:  # if no sets were appended, skip this shard
                continue
            
            # Take the intersection, giving list of samples with all required extensions.
            member_list = list(set.intersection(*sets))

            # Generate list containing shardkey, sample name, and offsets
            full_list = []
            for m in member_list:
                cur_offsets = []

                for extension in self.use_extensions:
                    if extension in shard_index[key] and m in shard_index[key][extension]:
                        cur_offsets.append(shard_index[key][extension][m])
                    else:
                        break
                else:
                    full_list.append((self.shard_keys.index(key), *cur_offsets))

            offsets += full_list

        # Order offsets deterministically if flag is set
        if deterministic_offset_order:
            offsets = sorted(
                offsets, 
                key=self._shard_offset_key
            )

        # Convert offsets to a tensor with shared memory.
        offsets = torch.tensor(offsets).share_memory_()

        return offsets


class QuixUnionDataset(Dataset):
    """Dataset class for joining multiple QuixDataset objects.

    The QuixUnionDataset allows for simultaneous sampling from all provided datasets
    in a round-robin fashion. All datasets should be of the same length, or they will
    be adjusted to the length of the longest dataset via modulo operation.

    Attributes
    ----------
    datasets : tuple of QuixDataset
        The datasets to be concatenated.
    maxlen : int
        The length of the longest dataset, which will be the effective length of the
        QuixUnionDataset.

    Parameters
    ----------
    datasets : QuixDataset
        Variable number of QuixDataset instances to be concatenated.

    Raises
    ------
    AssertionError
        If any provided dataset is not an instance of QuixDataset.

    Examples
    --------
    >>> dataset1 = QuixDataset(...)
    >>> dataset2 = QuixDataset(...)
    >>> combined_dataset = QuixUnionDataset(dataset1, dataset2)
    >>> sample = combined_dataset[5]  # Sample from both datasets at index 5

    Methods
    -------
    __len__()
        Returns the length of the longest dataset.
    __getitem__(key)
        Retrieves a combined tuple of items from all datasets at the given index.
    __repr__()
        Returns a string representation of the QuixUnionDataset instance.
    shufflecontext()
        A context manager for temporarily shuffling the order of offset indices in each dataset.
    """
    def __init__(self, *datasets:QuixDataset):
        """Initializes the QuixUnionDataset.

        Parameters
        ----------
        datasets : QuixDataset
            The datasets to be concatenated.

        Raises
        ------
        AssertionError
            If any provided dataset is not an instance of QuixDataset.
        """
        assert len(datasets) > 0
        assert all([isinstance(d, QuixDataset) for d in datasets])
        self.datasets = datasets
        self.maxlen = max([len(d) for d in datasets]) # type:ignore
        # Set modulo length for all smaller datasets
        for d in datasets:
            if len(d) < self.maxlen:
                d.mod_length = self.maxlen

    def __len__(self):
        return self.maxlen

    def __getitem__(self, key):
        return tuple([k for d in self.datasets for k in d[key]]) # type: ignore

    def __repr__(self):
        dstring = "\n".join([d.__repr__() for d in self.datasets])
        return f'{self.__class__.__name__}(\n{dstring}\n)'

    @contextmanager
    def shufflecontext(self):
        """Context manager for temporarily shuffling the order of offset indices in each dataset.

        This method shuffles the order of the offset indices for each underlying `QuixDataset`.
        The original offsets are restored once the context manager exits.

        Yields
        ------
        None
            Yields no value but modifies the offset index of each dataset in place.
        """
        context_managers = [ds.shufflecontext() for ds in self.datasets]

        try:
            for cm in context_managers:
                cm.__enter__()
            yield

        finally:
            for cm in reversed(context_managers):
                cm.__exit__(None, None, None)