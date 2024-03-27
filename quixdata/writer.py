'''
Quix Data Writing
=================

This module provides a set of classes designed for efficient creation and management 
of large datasets stored in a sharded tar format. These utilities are particularly 
useful for handling datasets in environments where direct access to filesystems is available,
optimizing for speed and flexibility in dataset manipulation.

The module contains three main classes:

- `IndexedTarWriter`: A class for writing objects to a tar file, supporting custom encoding
  methods and maintaining an index of byte offsets for efficient retrieval.

- `IndexedShardWriter`: Manages writing data to multiple tar files (shards), creating new 
  shards based on specified size or file count limits.

- `QuixWriter`: A high-level utility for creating and managing Quix datasets, orchestrating
  the storage pattern, writing data, and managing metadata and indices in sharded tar files.

The classes are designed to work together, providing a streamlined process for dataset 
creation, from individual data items to complete datasets organized into shards.

Classes
-------
IndexedTarWriter
    Writes objects to a tar file with indexed contents for efficient retrieval.
IndexedShardWriter
    Manages writing data to multiple tar files, handling shard creation and organization.
QuixWriter
    High-level utility for creating and managing datasets in the Quix framework.

Example Usage
-------------
Creating a new Quix dataset and writing data to it:

```python
from quix_tar_writing_utils import QuixWriter

# Initialize a QuixWriter for the dataset
quix_writer = QuixWriter(dataset_name='example_dataset', loc='/path/to/dataset')

# Write data to the training and validation sets
quix_writer.train.write({'__key__': 'sample1', 'image.jpg': image_data, 'label.txt': label_data})
quix_writer.val.write({'__key__': 'sample2', 'image.jpg': image_data, 'label.txt': label_data})

# Finalize the dataset
quix_writer.close()

Author
------
Marius Aasan <mariuaas@ifi.uio.no>

'''
import tarfile
import os
import time
import getpass
import re
import json

from io import BytesIO
from collections import OrderedDict
from typing import Optional, Tuple, Iterable, Callable, IO, Union
from .encoders import DEFAULT_ENCODERS

StrOrStream = Union[str, IO]
_validcomp = ['gz', 'bz2', 'xz']


class IndexedTarWriter:
    """A writer for creating tar files with indexed contents.

    This class is designed to write objects to a tar file, supporting optional compression
    and custom encoding methods for different file types. It maintains an index of byte offsets
    for efficient retrieval of files from the tar archive.

    Parameters
    ----------
    fileobj : StrOrStream
        The file name or file object to write the tar file to.
    username : Optional[str], optional
        The user name to use in the tar file metadata, by default the current user.
    groupname : str, optional
        The group name to use in the tar file metadata, by default 'defaultgroup'.
    mode : int, optional
        The file permission mode to use in the tar file metadata, by default 292.
    compression : Optional[str], optional
        The compression method ('gz', 'bz2', 'xz', or None), by default None.
    override_encoders : Optional[Iterable[Tuple[str, Callable]]], optional
        Custom encoders for specific file extensions, by default None.

    Attributes
    ----------
    encoders : dict
        A dictionary of file encoders, keyed by file extension.
    username : str
        Username to include in the tar file metadata.
    groupname : str
        Group name to include in the tar file metadata.
    mode : int
        File permission mode to include in the tar file metadata.
    index : OrderedDict
        An ordered dictionary mapping file extensions to byte offsets.

    Methods
    -------
    write(objdict: dict) -> int
        Writes objects to the tar file based on the provided dictionary.
    close()
        Closes the tar file and associated resources.
    """

    def __init__(
        self,
        fileobj:StrOrStream,
        username:Optional[str]=None,
        groupname:str='defaultgroup',
        mode:int=292,
        compression:Optional[str] = None,
        override_encoders:Optional[Iterable[Tuple[str, Callable]]] = None
    ):
        '''
        Initializes an IndexedTarWriter instance.

        Args:
            fileobj (StrOrStream): The file name or file object to write the tar file to.
            username (str, optional): The user name to write in the tar file metadata. Default is 'defaultuser'.
            groupname (str, optional): The group name to write in the tar file metadata. Default is 'defaultgroup'.
            mode (int, optional): The file permission mode to write in the tar file metadata. Default is 292.
            compression (str, optional): The compression method to use when writing the tar file. Must be one of 
                'gz', 'bz2', 'xz', or None for no compression. Default is None.
            override_encoders (Iterable[Tuple[str, Callable]], optional): An iterable of tuples where the first element 
                is a file extension (including the leading dot) and the second element is a callable that takes a Python 
                object and returns a bytes-like object. These encoders will be added to the set of default encoders, 
                replacing any default encoders with the same file extension.
        '''
        if compression is not None:
            assert compression in _validcomp
            tarmode = f'w:{compression}'
        else:
            tarmode = 'w'

        if isinstance(fileobj, str):
            folder, filename = os.path.split(fileobj)
            assert os.path.isdir(folder), f'Invalid file/folder: {folder}/{filename}.'
            self.fileobj = open(os.path.join(folder, filename), 'wb')
            self._closeflag = True
        else:
            assert hasattr(fileobj, 'tell'), "Provided file object must support seek/tell"
            self.fileobj = fileobj
            self._closeflag = False

        self.encoders = {**DEFAULT_ENCODERS}
        if override_encoders is not None:
            for k,v in override_encoders:
                assert callable(v)
                if not k.startswith('.'):
                    k = f'.{k}'
                self.encoders[k] = v

        self.username = username if username is not None else getpass.getuser()
        self.groupname = groupname
        self.mode = mode
        self.index = OrderedDict()

        self._tarfile = tarfile.open(
            fileobj=self.fileobj,
            mode=tarmode
        )
    
    def write(self, objdict:dict):
        """Writes objects to the tar file based on the provided dictionary.

        Each item in the dictionary is encoded to bytes using the appropriate encoder
        for its file extension and then written to the tar file as a separate file.

        Parameters
        ----------
        objdict : dict
            A dictionary where the keys are file names and the values are Python objects.
            Must contain a '__key__' key for naming the files.

        Returns
        -------
        int
            The total size of all data written to the tar file from this dictionary, in bytes.
        """
        assert '__key__' in objdict
        assert len(objdict) > 1
        key = objdict['__key__']
        totalsize = 0
        for k, v in objdict.items():
            if k == '__key__':
                continue

            # Get extension. TODO: Default encoder could ideally handle this better?
            name, ext = os.path.splitext(k)
            ext = f'.{name}' if ext == '' else ext
            byteval = self.encoders[ext](v)
            assert isinstance(byteval, (bytes, bytearray, memoryview))

            # Generate tarinfo
            tarinfo = tarfile.TarInfo(key + '.' + k)
            tarinfo.mtime = int(time.time())
            tarinfo.size = len(byteval)
            tarinfo.uname = self.username
            tarinfo.gname = self.groupname

            # Store indices
            if f'.{k}' not in self.index:
                self.index[f'.{k}'] = OrderedDict()
            self.index[f'.{k}'][key] = self.fileobj.tell()
            self._tarfile.addfile(tarinfo, BytesIO(byteval))
            totalsize += tarinfo.size

        return totalsize
    
    def close(self):
        '''Close the tar file.

        If the file object was created by this IndexedTarWriter instance, it is also closed.
        '''
        self._tarfile.close()
        if self._closeflag:
            self.fileobj.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __enter__(self):
        return self


class IndexedShardWriter:
    """Manages writing data to multiple shard files, each being a tar file.

    This class handles the creation of new shards when the current shard reaches
    its maximum size or file count, ensuring efficient organization of large datasets.

    Parameters
    ----------
    base_pattern : str
        The pattern for naming shard files, with a placeholder for the shard index.
    start : int, optional
        The starting index for shard files, by default 0.
    shard_maxfiles : int, optional
        The maximum number of files per shard, by default 100000.
    shard_maxsize : float, optional
        The maximum size of each shard in bytes, by default 3e9.
    **kwargs : dict
        Additional keyword arguments for IndexedTarWriter.

    Attributes
    ----------
    base_pattern : str
        The base pattern for shard file names.
    shard_maxfiles : int
        Maximum number of files per shard.
    shard_maxsize : float
        Maximum size of each shard in bytes.
    next_shard_index : int
        Index for the next shard file.
    cur_tarfile : IndexedTarWriter
        The current tar file being written to.
    cur_shardname : str
        The name of the current shard file.
    cur_totalsize : int
        The current size of the shard being written to.
    cur_count : int
        The current number of files in the shard.
    total_count : int
        The total number of files written across all shards.
    index : dict
        A dictionary mapping shard names to offset indices.

    Methods
    -------
    new_shard()
        Finalizes the current shard and starts a new one.
    finalize_shard()
        Finalizes the current shard and stores its offset indices.
    close()
        Closes the current shard and all associated resources.
    write(objdict: dict)
        Writes a dictionary of objects to the current shard.
    """
    def __init__(
        self,
        base_pattern:str,
        start=0,
        shard_maxfiles:int=100000,
        shard_maxsize:float=3e9,
        **kwargs
    ):
        '''Initializes the ShardWriter with the given settings.

        Args:
            base_pattern (str): The base pattern for the file names of the shards.
                The pattern should include one '%d' or '%i' placeholder, which will be replaced by
                the shard's index to generate each shard's filename.
            start (int, optional): The starting index for the shards. Defaults to 0.
            shard_maxfiles (int, optional): The maximum number of files that can be written to a 
                single shard. When this limit is reached, a new shard will be started. Defaults to 100000.
            shard_maxsize (float, optional): The maximum size (in bytes) that a single shard can be.
                When this size is exceeded, a new shard will be started. Defaults to 3e9.
            **kwargs: Keyword arguments for writing the individual shards.
        '''
        self.base_pattern = base_pattern
        self.shard_maxfiles = shard_maxfiles
        self.shard_maxsize = shard_maxsize
        self.kwargs = kwargs

        self._start = start
        self.next_shard_index = start        
        self.cur_tarfile = None
        self.cur_shardname = None
        self.cur_totalsize = 0
        self.cur_count = 0
        self.total_count = 0
        self.index = {}
        self.new_shard()

    def new_shard(self):
        """Finalizes the current shard and starts a new one.

        Closes the current tar file and opens a new one for the next shard, 
        updating the shard name and resetting counters.
        """
        self.finalize_shard()
        self.cur_shardname = self.base_pattern % self.next_shard_index
        self.next_shard_index += 1
        self.cur_tarfile = IndexedTarWriter(self.cur_shardname, **self.kwargs)
        self.cur_count = 0
        self.cur_totalsize = 0

    def finalize_shard(self):
        """Finalizes the current shard and stores the offset indices.

        Closes the current tar file and updates the index with the byte offsets
        of the files contained within the shard.
        """
        if self.cur_tarfile is not None:
            self.cur_tarfile.close()
            assert self.cur_shardname is not None
            _, key = os.path.split(self.cur_shardname)
            self.index[key] = self.cur_tarfile.index
            self.cur_tarfile = None

    def close(self):
        '''Close the shardwriter.
        '''
        self.finalize_shard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, objdict:dict):
        '''Writes a dictionary of objects to the current shard.

        The `objdict` dictionary must contain a '__key__' key, whose corresponding value 
        will be used as the prefix for the names of all files written from this dictionary.
        Each other item in the dictionary will be encoded to bytes using the encoder
        for its file extension and then written to the tar file as a separate file.

        Parameters
        ----------
        objdict : dict
            A dictionary where the keys are file names and the values are Python objects.
            Must contain a '__key__' key for naming the files.
        '''
        if (
            (self.cur_tarfile is None) or 
            (self.cur_count >= self.shard_maxfiles) or 
            (self.cur_totalsize >= self.shard_maxsize)
        ):
            self.new_shard()

        assert self.cur_tarfile is not None
        size = self.cur_tarfile.write(objdict)
        self.cur_count += 1
        self.total_count += 1
        self.cur_totalsize += size

    def __repr__(self):
        repr_str = (
            f"ShardWriter(\n"
            f"\tCurrent Shard: {self.cur_shardname}\n"
            f"\tCurrent Shard Size: {self.cur_totalsize / 1e9:.2f} GB\n"
            f"\tCurrent Shard File Count: {self.cur_count}\n"
            f"\tTotal File Count: {self.total_count}\n"
            f")"
        )
        return repr_str


class QuixWriter:
    """Utility class for creating and managing Quix datasets in sharded tar format.

    Provides functionalities for setting up datasets, configuring their storage patterns,
    and writing data into sharded tar files. It also manages dataset metadata and indices.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    loc : str
        The directory where the dataset will be stored.
    pattern : str, optional
        The pattern for naming tar files, by default '{}_%04d.tar'.
    trainstr : str, optional
        The string identifier for the training set in file names, by default 'train'.
    valstr : str, optional
        The string identifier for the validation set in file names, by default 'val'.
    additional_metadata : dict, optional
        Additional metadata for the dataset, by default {}.
    config_fname : str, optional
        The name of the configuration file, by default 'config.json'.
    train_idx_fname : str, optional
        The name of the training index file, by default 'train.idx.json'.
    val_idx_fname : str, optional
        The name of the validation index file, by default 'val.idx.json'.
    **kwargs : dict
        Additional keyword arguments for IndexedShardWriter.

    Attributes
    ----------
    train : IndexedShardWriter
        Shard writer for the training set.
    val : IndexedShardWriter
        Shard writer for the validation set.
    cfg : dict
        Configuration details of the dataset.

    Methods
    -------
    close()
        Finalizes the dataset and writes configuration and index files.
    """
    def __init__(
        self,
        dataset_name:str,
        loc:str,
        pattern:str='{}_%04d.tar',
        trainstr:str='train',
        valstr:str='val',
        additional_metadata:dict={},
        config_fname:str='config.json',
        train_idx_fname:str='train.idx.json',
        val_idx_fname:str='val.idx.json',
        **kwargs
    ):
        '''
        Initializes a QuixWriter with the provided dataset name, location, naming 
        pattern, and additional arguments.

        Args:
            dataset_name (str): Name of the dataset.
            loc (str): Directory where the dataset will be stored.
            pattern (str, optional): A pattern for naming the tar files. 
                Default is '{}_%04d.tar'.
            trainstr (str, optional): String to represent the training set in file names. 
                Default is 'train'.
            valstr (str, optional): String to represent the validation set in file names. 
                Default is 'val'.
            additional_metadata (dict, optional): Additional metadata for the dataset. 
                Default is an empty dict.
            config_fname (str, optional): Name of the configuration file. 
                Default is 'config.json'.
            train_idx_fname (str, optional): Name of the index file for the training set. 
                Default is 'train.idx.json'.
            val_idx_fname (str, optional): Name of the index file for the validation set. 
                Default is 'val.idx.json'.
            **kwargs: Additional keyword arguments for the IndexedShardWriter.
        '''
        assert os.path.isdir(loc)

        self.dataset_name = dataset_name
        self.loc = loc
        self.fullpath = os.path.join(loc, dataset_name)
        if not os.path.isdir(self.fullpath):
            os.mkdir(self.fullpath)

        self.train_pattern = pattern.format(trainstr)
        self.val_pattern = pattern.format(valstr)
        self.train = IndexedShardWriter(os.path.join(self.fullpath, self.train_pattern), **kwargs)
        self.val = IndexedShardWriter(os.path.join(self.fullpath, self.val_pattern), **kwargs)
        self.config_fname = config_fname
        self.train_idx_fname = train_idx_fname
        self.val_idx_fname = val_idx_fname
        
        self.cfg = {
            'train': f'/{self._procpat(self.train_pattern)}', # Add first and last when finalizing.
            'val': f'/{self._procpat(self.val_pattern)}', # Add first and last when finalizing.
            'extensions': [], # Add extensions when finalizing.
            'metadata': additional_metadata, # Add remaining metadata when finalizing.
        }
    
    def _procpat(self, instr:str):
        rematch = re.search(r"%(\d*)(d|i)", instr)
        assert rematch
        fillfmt = '%' + rematch.group(1) + rematch.group(2)
        new_str = re.sub(
            rematch.group(), 
            f'{{{fillfmt}..{fillfmt}}}', 
            instr)
        return new_str

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def close(self):
        '''Closes the QuixWriter.

        Finalizes the dataset by closing the training and validation ShardWriters, 
        gathering the file extensions, updating the configuration, and writing the 
        configuration and index files to disk.
        '''
        self.train.close()
        self.val.close()

        extensions = list(set.union(
            set([k for v in self.train.index.values() for k in v.keys()]),
            set([k for v in self.val.index.values() for k in v.keys()])
        ))
        extensions = [ext[1:] if ext.startswith('.') else ext for ext in extensions]

        # Finalize config file
        self.cfg['train'] = self.cfg['train'] % (
            self.train._start, 
            self.train.next_shard_index - 1
        )
        self.cfg['val'] = self.cfg['val'] % (
            self.val._start, 
            self.val.next_shard_index - 1
        )
        self.cfg['extensions'] = extensions
        self.cfg['metadata']['num_train'] = self.train.total_count
        self.cfg['metadata']['num_val'] = self.val.total_count

        # Construct paths and write config and index files
        cfgpath = os.path.join(self.fullpath, self.config_fname)
        tidxpath = os.path.join(self.fullpath, self.train_idx_fname)
        vidxpath = os.path.join(self.fullpath, self.val_idx_fname)

        with open(cfgpath, 'w') as cfgfile:
            json.dump(self.cfg, cfgfile)

        with open(tidxpath, 'w') as tidxfile:
            json.dump(self.train.index, tidxfile)

        with open(vidxpath, 'w') as vidxfile:
            json.dump(self.val.index, vidxfile)

