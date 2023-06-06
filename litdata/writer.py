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
        '''
        Write objects to the tar file.

        The `objdict` dictionary must contain a '__key__' key, whose corresponding value 
        will be used as the prefix for the names of all files written from this dictionary.
        Each other item in the dictionary will be encoded to bytes using the encoder
        for its file extension and then written to the tar file as a separate file.

        Args:
            objdict (dict): A dictionary where the keys are file names and the values are Python objects.

        Returns:
            int: The total size of all data written to the tar file from this dictionary, in bytes.
        '''
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
        '''
        Close the tar file.

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
    '''Writes data to multiple shard files, each of which is a tar file.
    
    This class manages writing data to multiple tar files, creating a new shard
    when the current one becomes too large or contains too many files.

    Attributes:
        base_pattern (str): The base pattern for the file names of the shards.
        shard_maxfiles (int): Maximum number of files per shard.
        shard_maxsize (float): Maximum size of each shard in bytes.
        next_shard_index (int): Index to use for naming the next shard.
        cur_tarfile (IndexedTarWriter): Current tar file being written to.
        cur_shardname (str): Name of the current shard.
        cur_totalsize (int): Current size of the shard being written to.
        cur_count (int): Current number of files in the shard.
        total_count (int): Total number of files written across all shards.
        index (dict): A dictionary mapping from shard names to indices.
    '''
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
        '''Finalizes the current shard and starts a new one.
        '''
        self.finalize_shard()
        self.cur_shardname = self.base_pattern % self.next_shard_index
        self.next_shard_index += 1
        self.cur_tarfile = IndexedTarWriter(self.cur_shardname, **self.kwargs)
        self.cur_count = 0
        self.cur_totalsize = 0

    def finalize_shard(self):
        '''Finalizes current shard and stores the offset indices.
        '''
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

        Args:
            objdict (dict): A dictionary where the keys are file names and the values are Python objects.
        
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


class LITWriter:

    '''A utility class for writing datasets for use in the LIT framework.

    This class provides utilities for creating LIT datasets and writing them
    into sharded tar files. This includes the configuration of the datasets, setting 
    their names and locations, and managing their storage pattern. The LITWriter also 
    manages the writing of data to the tar files and the updating of metadata and index files.

    Attributes:
        dataset_name (str): Name of the dataset.
        loc (str): Directory where the dataset will be stored.
        fullpath (str): Full path to the directory where the dataset will be stored.
        train_pattern (str): Pattern used to name the tar files for the training set.
        val_pattern (str): Pattern used to name the tar files for the validation set.
        train (IndexedShardWriter): Shard writer for the training set.
        val (IndexedShardWriter): Shard writer for the validation set.
        config_fname (str): Name of the configuration file.
        train_idx_fname (str): Name of the index file for the training set.
        val_idx_fname (str): Name of the index file for the validation set.
        cfg (dict): A dictionary containing the dataset's configuration details.
    '''

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
        Initializes a LITWriter with the provided dataset name, location, naming 
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
        '''Closes the LITWriter.

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

