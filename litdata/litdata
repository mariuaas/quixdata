import re
import json
import tarfile
import os
import inspect

from io import BytesIO
from PIL import Image
from typing import (
    Tuple, Dict, Iterable, List, Callable, Union, TypeVar, Optional
)

'''
litdata is a compact dataset handler for local indexed tar sharded datasets.

litdata contains the LITDataset class which handles locally stored datasets 
using .tar shards, e.g. the WebDataset (WDS) format. Designed to simplify 
some of the implementation choices used in WDS for online hosting, such as 
missing length and buffered sampling, which can be an issue for locally 
hosted datasets, and allows the dataset to be shuffled / batched / etc. 
using native DataLoader classes. This also improves shuffling behaviour for 
tasks which require high stochasticity, such as contrastive learning.

Instead of sequential iteration over shards, which is not necessary if files
are hosted on a local filesystem with high speeds, this format creates an
index of the byte offsets of all files in the shards, which can be serialized
for faster subsequent initialisation. These indices are then concatenated
by taking the union over elements which match the supplied extensions.
    
Author: Marius Aasan <mariuaas@ifi.uio.no>
'''


_LITDataset_t = TypeVar('_LITDataset_t', bound='LITDataset')

class LITMapTuple:

    def __init__(self, maps):
        assert all([isinstance(m, Callable) for m in maps])
        #assert all([len(inspect.signature(m).parameters) == 1 for m in maps]) # Ignore this assertion for now
        self.maps = maps
    
    def fnapply(self, fn, x):
        return fn(x)

    def __call__(self, y):
        return list(map(self.fnapply, self.maps, y))
    

class LITMapAll:
    def __init__(self, mapping):
        assert isinstance(mapping, Callable)
        #assert len(inspect.signature(mapping).parameters) == 1 # Ignore this assertion for now
        self.mapping = mapping

    def __call__(self, y):
        return list(map(self.mapping, y))


class LITDataset(Dataset):
    
    '''Local Indexed Tar Dataset.
    
    A simple dataset class to handle locally stored datasets using tar shards. 
    The goal is to simplify some of the implementation choices used in the
    WebDataset format which is designed for streaming from online hosting, 
    such as missing length and buffered sampling. It allows the dataset to be 
    shuffled / batched / etc. using native DataLoader classes. 
    
    With default datasets and batching done by DataLoader, applying post-batched 
    transformations / mappings is done by supplying a wrapper around `collate_fn`
    and passing this to the DataLoader.
    
    NOTE: This class will return the selected extensions in the order
          they are passed in use_extensions. For image extensions, the 
          loader will return a PIL Image object. 

    TODO: Implement decoders for QOI, RLE, Polygons, Bboxes.
    TODO: Add option to pass `idx` as extension to retrieve sample indices? 
    TODO: Add option to pass `name` as extension to retrieve sample names? 
    '''
    
    def __init__(
        self, 
        dataset:str, 
        loc:str, 
        train:bool=True, 
        override_extensions:Optional[Tuple[str,...]]=None,
        override_decoders:Optional[Iterable[Tuple[str, Callable]]]=None,
        config_fname:str='config.json',
        train_idx_fname:str='train.idx.json',
        val_idx_fname:str='val.idx.json',
        serialize:bool=False,
        debug:bool=False
    ):
        '''Initializes the LITDataset
        
        NOTE: This class will return the selected extensions in the order
              they are passed in use_extensions. For image extensions, the loader 
              will return a PIL Image object. Work on support for other extensions is
              ongoing.
                      
        Args:
            dataset (str): Name of the webdataset. Should be a subfolder of loc.
            loc (str): Locations of the webdataset.
            train (bool, optional): Whether to use training or validation. Default is True.
            override_extensions (Tuple[str], optional): Extension overrides.
            override_decoders (Iterable[Tuple[str, Callable]], optional): Decoder overrides.
            config_fname (str, optional): The config file name to use, default to 'config.json'.
            train_idx_fname (str, optional): The train index file name to use, default to 'train.idx.json'.
            val_idx_fname (str, optional): The validation file name to use, default to 'val.idx.json'.
            serialize (bool, optional): If True, serializes the offsets in the dataset folder.
            debug (bool, optional): If True, prints to stdout during extraction of filenames.
        '''
        # Set initial parameters
        self.debug = debug
        self.fold = 'train' if train else 'val'
        self.root = os.path.join(loc, dataset)
        self.config_fname = config_fname
        self.index_fname = train_idx_fname if train else val_idx_fname
                
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
                self.shard_index = json.load(infile)
                
        # If serialized indices does not exist, generate from scratch.
        else:
            self.shard_index = self.generate_shard_index()        
            if serialize:
                with open(idxpath, 'w') as outfile:
                    json.dump(self.shard_index, outfile)
        
        # Generate offset index.
        self.offset_index = self.generate_offset_index()
        
        # Initialize transforms
        self.transforms = []
        
    @property
    def num_classes(self):
        return self.cfg['metadata']['num_classes']
    
    @property
    def num_train(self):
        return self.cfg['metadata']['num_train']

    @property
    def num_val(self):
        return self.cfg['metadata']['num_val']
                
    def __len__(self):
        return len(self.offset_index)
        
    def __getitem__(self, key):
        # Load keys, incl. shard name and extension offsets.
        key_tuple = self.offset_index[key]
        shard_name, offsets = key_tuple[0], key_tuple[2:]
        
        # Apply offsets in ascending order
        os_argsort = [i for i, _ in sorted(enumerate(offsets), key=lambda x: x[1])]
        os_sorted = sorted(offsets)
        
        # Generate list for storing outputs
        out = [None]*len(offsets)
        
        # Load shard and retrieve files with extensions at offsets
        shard_path = os.path.join(self.root, shard_name)
        with tarfile.open(shard_path, 'r') as tar:
            for extidx, offset in zip(os_argsort, os_sorted):
                tar.fileobj.seek(offset)                        # type: ignore
                member = tarfile.TarInfo.fromtarfile(tar)
                data = tar.extractfile(member).read()           # type: ignore
                out[extidx] = self.decoders[extidx](data)
        
        # Apply transformations
        if len(self.transforms) > 0:
            for t in self.transforms:
                out = t(out)
        
        if isinstance(out, Iterable):
            return tuple(out)

        return out
    
    def __repr__(self):
        fold = self.fold
        num_classes = self.num_classes
        num_train = self.num_train
        num_val = self.num_val
        length = len(self)
        use_extensions = self.use_extensions
        return (f'{self.__class__.__name__}(\n    ' +
            f'fold={fold},\n    ' +
            f'num_classes={num_classes},\n    ' +
            f'length={length},\n    ' +
            f'use_extensions={use_extensions},\n    ' +
            f'num_train={num_train},\n    ' + 
            f'num_val={num_val},\n' +
        ')')
    
    @staticmethod
    def braceexpand(input_string:str, root:str) -> Tuple[str]:
        '''Performs brace expansion for a string given root.
        
        Args:
            input_string (str): Brace formatted string, eg., <pf><int>..<int><suf>.
            root (str): Root folder.
            
        Returns:
            Tuple[str]: Tuple of joined paths from brace expansion.
        '''
        match = re.search(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', input_string)
        prefix, start, end, suffix = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4) # type: ignore
        num_digits = len(match.group(2)) # type: ignore
        format_str = f"{prefix}{{:0{num_digits}d}}{suffix}"
        return tuple([os.path.join(root, format_str.format(i)) for i in range(start, end + 1)])
    
    @staticmethod
    def _fixext(extensions: Tuple[str]) -> Tuple[str]:
        out = [f'.{e}' if not e.startswith('.') else e for e in extensions]
        return tuple(out)
    
    @staticmethod
    def _pil_decoder(data:bytes):
        return Image.open(BytesIO(data))

    @staticmethod
    def _cls_decoder(data:bytes):
        return int(data)
    
    @staticmethod
    def _non_decoder(data:bytes):
        return data
        
    def map_tuple(self, *maps:Tuple[Callable,...]) -> _LITDataset_t:
        '''Takes a set of mappings and applies them to individual extensions.
        
        For `maps = [f1, f2]` and `extensions = ['jpg', 'cls'], this will return
        the tuple `(f1(<sample>.jpg), f2(<sample>.cls))`.
        
        Args:
            maps (Tuple[Callable]): The mappings to apply in order of the extensions.

        Returns:
            LITDataset: Returns the dataset with added transformations.
        '''  
        assert len(maps) == len(self.use_extensions)
        # Removed some assertions for now, will be moved to the init of the LITMapTuple class
        self.transforms.append(LITMapTuple(maps))
        return self # type: ignore
                
    def map_all(self, mapping:Callable) -> _LITDataset_t:
        '''Takes a mapping and applies it to all extensions.
        
        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `(f(<sample>.jpg), f(<sample>.cls))`.

        Args:
            mapping (Callable): The mapping to apply to.

        Returns:
            LITDataset: Returns the dataset with added transformations.
        '''
        assert len(inspect.signature(mapping).parameters) == 1
        # Removed some assertions for now, will be moved to the init of the LITMapAll class
        self.transforms.append(LITMapAll(mapping))
        return self # type: ignore
        
    def map(self, mapping:Callable) -> _LITDataset_t:
        '''Takes a mapping and applies it to the tuple of extensions.
        
        For `mapping = f` and `extensions = ['jpg', 'cls'], this will return
        the tuple `f(<sample>.jpg, f(<sample>.cls)`.

        Args:
            mapping (Callable): The mapping to apply to.
            
        Returns:
            LITDataset: Returns the dataset with added transformations.
        '''
        assert len(inspect.signature(mapping).parameters) == len(self.use_extensions)
        self.transforms.append(mapping)
        return self # type: ignore
    
    def set_decoders(
        self, override:Optional[Iterable[Tuple[str, Callable]]]
    ) -> Tuple[Callable,...]:
        '''Set the decoders for the extensions.
        
        Args:
            override (Iterable[Tuple[str, Callable]]): Override decoders.
            
        Returns:
            Tuple[Callable]: Tuple of decoders.
        '''
        decoders = [None]*len(self.use_extensions)
        
        # Prefer override extensions
        if override is not None:
            for ext, dec in override:
                assert isinstance(dec, Callable), f'Decoder for {ext} not callable!'
                ext = self._fixext((ext,))[0]
                if ext in self.use_extensions:
                    idx = self.use_extensions.index(ext)
                    decoders[idx] = dec                     # type: ignore
        
        # Use default decoders
        for idx, ext in enumerate(self.use_extensions):
            if decoders[idx] is None:
                if ext in Image.registered_extensions():
                    decoders[idx] = self._pil_decoder       # type: ignore
                elif ext == '.cls':
                    decoders[idx] = self._cls_decoder       # type: ignore
                else:
                    decoders[idx] = self._non_decoder       # type: ignore
        
        return tuple(decoders)                              # type: ignore
            
    def generate_shard_index(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        '''Generates shard index from shard list.
        
        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Dict of Shard[Extension][Sample] = Offset.
        '''
        
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
    
    def generate_offset_index(self) -> List[Tuple[Union[str, int]]]:
        '''Generates offset indices from shard list.
        
        Returns:
            List[Tuple[Union[str, int]]]: List with shardname, filename, and offsets.
        '''
        offsets = []

        for key in self.shard_index:            
            # Generate list of samples which share the relevant extensions
            sets = []
            for i, extension in enumerate(self.use_extensions):
                sets.append(set(self.shard_index[key][extension]))
            
            # Take the union, giving list of samples with particular set of extensions.
            member_list = list(set.union(*sets))

            # Generate list containing shardkey, sample name, and offsets
            full_list = []            
            for m in member_list:
                cur_offsets = []
                
                for extension in self.use_extensions:
                    cur_offsets.append(self.shard_index[key][extension][m])
                
                full_list.append((key, m, *cur_offsets))
            
            offsets += full_list
        
        return offsets
    