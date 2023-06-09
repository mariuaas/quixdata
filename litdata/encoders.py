import os
import numpy as np
import json
import scipy.io as spio
import pickle

from io import BytesIO
from typing import List, Dict, Union, Set, Optional
from PIL import Image


class EncoderDecoder:

    '''Base EncoderDecoder class for LITDataset.

    Abstract decoder class to build encoders for different data modalities.
    To implement a subclass, it must have:
        - `_encode` and `_decode` methods.
        - A `supported_extensions` attribute.
    
    Then, encoders or decoders can be instantiated using the classmethods 
    `encoder` or `decoder`, or by initializing the class with the corresponding
    mode, either `encode` or `decode`.
    '''

    def __init__(self, mode:str):
        '''Initializes the encoder / decoder. 
        
        Args:
            mode (str): The mode, either `encode` or `decode`.
        '''
        self.supported_extensions:Optional[Union[List,Dict,Set]] = None
        assert mode in ['encode', 'decode']
        self._mode = mode
        self._bit8 = 2**8
        self._bit16 = 2**16
        self._bit24 = 2**24

    def supports(self, filepath:str):
        '''Returns True if the provided path is supported.

        NOTE: This does not check if the file exists.

        Args:
            filepath (str): A file path. 
        
        Returns:
            bool: Whether the file extension is supported by the encoder/decoder.
        '''
        assert self.supported_extensions is not None
        name, extension = os.path.splitext(filepath)
        return extension in self.supported_extensions

    def _encode(self, obj):
        raise NotImplementedError('EncodeDecoder is abstract!')
    
    def _decode(self, data:bytes):
        raise NotImplementedError('EncoderDecoder is abstract!')
    
    def __call__(self, x):
        if self._mode == 'encode':
            return self._encode(x)
        elif self._mode == 'decode':
            return self._decode(x)
        else:
            raise ValueError(f'Invalid mode: {self._mode}')
    
    @classmethod
    def encoder(cls):
        '''Initializes an encoder.
        '''
        return cls('encode')

    @classmethod
    def decoder(cls):
        '''Initializes a decoder.
        '''
        return cls('decode')


class PIL(EncoderDecoder):

    '''PIL Image EncoderDecoder class for LITDataset.

    Default image encoder / decoder for images.
    NOTE: Image extensions are generally considered raw image data. For masks 
          and other alternate image modalities, we recommend using either the provided
          segmentation encoders, or a custom implementation.
    '''
    def __init__(self, mode:str):
        '''Initializes the encoder / decoder. 
        
        Args:
            mode (str): The mode, either `encode` or `decode`.
        '''
        super().__init__(mode)
        self.supported_extensions = Image.registered_extensions()

    def _decode(self, data:bytes):
        return Image.open(BytesIO(data))
    
    def _encode(self, img:Image.Image):
        assert img.format is not None, f'PIL image must have a set format!'
        byte_arr = BytesIO()
        img.save(byte_arr, format=img.format)
        return byte_arr.getvalue()


class CLS(EncoderDecoder):

    '''Class label EncoderDecoder for LITDataset.

    A simple label encoder for encoding class labels for single label
    classification tasks.
    '''

    def __init__(self, mode:str):
        '''Initializes the encoder / decoder. 
        
        Args:
            mode (str): The mode, either `encode` or `decode`.
        '''
        super().__init__(mode)
        self.supported_extensions = {'.cls': 'Classification label'}

    def _decode(self, data:bytes):
        return int(data)
    
    def _encode(self, label:int):
        return f'{label}'.encode()


class RLE(EncoderDecoder):

    '''Simple run length EncoderDecoder for LITDataset.

    An implementation of a simple run length encoding, generally for
    largely homogeneous data, e.g. segmentation masks. The RLE supports
    arbitrary dimensions, which is useful for higher dimensional segmentation
    than images for 2D image data.

    NOTE: This is not the same RLE encoding used in the COCO API. 

    NOTE: The encoder takes numpy ndarrays of arbitrary shape as input.
          The output file consists of a header which contains:
            - A single byte encoding the from/to dtypes for compression. An optimal
              dtype is inferred during the encoding process.
            - The shape and table lengths in bytes (8 bytes each).
            - The shape of the array.
            - The lookup table for the values of the array.
          After the header, the rest of the file contains the RLE data.

    NOTE: The RLE encoding typically takes 1.5x-6x more space than a PNG encoded
          segmentation, but supports up to 64-bit resolution masks. However, for 2D images,
          the SEG8, SEG16, SEG24, or SEG32 encoder/decoders are much more space effective for 
          storing 2D data.

          HOWEVER; it is worth mentioning that the RLE encoding is significantly faster than
          PNG or QOI formats for encoding / decoding segmentation masks. Up to about 8x-15x faster.
          If storage is not an issue, this is a factor to consider.

          All in all, you have a worst case of a 6x space increase, but a 15x speed increase,
          so for large batch sizes, the RLE should be somewhat competitive.
    '''

    def __init__(self, mode:str):
        '''Initializes the encoder / decoder. 
        
        Args:
            mode (str): The mode, either `encode` or `decode`.
        '''
        super().__init__(mode)
        self.supported_extensions = {
            '.rle': 'Run length encoding',
        }

    @staticmethod
    def pack_dtypes(olddt:np.dtype, newdt:np.dtype):
        '''Packs dtypes into single byte.

        Args:
            olddt (np.dtype): The previous dtype. Can be signed/unsigned.
            newdt (np.dtype): The new dtype. Always unsigned.
        
        Returns:
            bytes: A single byte encoding the old/new datatypes.
        '''
        old = int(np.log2(np.iinfo(olddt).bits)) - 3
        new = int(np.log2(np.iinfo(newdt).bits)) - 3
        sign = 1 if np.issubdtype(olddt, np.signedinteger) else 0
        old = (old << 1) | sign
        return (old << 4 | new).to_bytes(1, 'big')
    
    @staticmethod
    def unpack_dtypes(dtbyte:bytes):
        '''Unpacks dtypes from single byte.

        Args:
            dtbyte (bytes): A byte object encoding the old/new dtypes. 

        Returns:
            tuple[np.dtype, np.dtype]: The old and new dtypes, respectively.
        '''
        dtint = int.from_bytes(dtbyte, 'big')
        old, new = dtint >> 4, dtint & 15
        sign = old & 1
        old = (old >> 1)
        oldbit = 2**(old+3)
        newbit = 2**(new+3)
        olddt = np.dtype(('int' if sign else 'uint') + str(oldbit))
        newdt = np.dtype('uint' + str(newbit))
        return olddt, newdt

    @staticmethod    
    def unique_consecutive_with_counts(arr):
        '''Finds unique consecutive elements and their counts for RLE.

        Args:
            arr (np.ndarray): A flattened array to encode with RLE.
            dt (np.dtype): The dtype of the array.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing unique cons. values and counts.
        '''
        assert arr.ndim == 1
        mask = np.empty(arr.shape, dtype=np.bool_)
        mask[0] = True
        mask[1:] = arr[1:] != arr[:-1]
        unique_consecutive = arr[mask]
        counts = np.diff(np.append(np.where(mask), len(arr)))
        return unique_consecutive, counts    

    def _decode(self, data:bytes):
        # Unpack dtypes
        olddt, newdt = self.unpack_dtypes(data[0:1])
        
        # Unpack lengths and get endpoints
        shapelen = int.from_bytes(data[1:9], 'big')
        tablelen = int.from_bytes(data[9:17], 'big')
        shape_end = 17+shapelen
        table_end = shape_end + tablelen
        
        # Get shape, table and RLE bytes
        shapebytes = data[17:shape_end]
        tablebytes = data[shape_end:table_end]
        rlebytes = data[table_end:]
        
        # Get indices and counts
        unq, cnt = np.frombuffer(rlebytes, dtype=newdt).reshape(2, -1)
        table = np.frombuffer(tablebytes, dtype=olddt)
        
        # Reconstruct array
        vals = table[unq]
        shape = np.frombuffer(shapebytes, dtype=np.uint64)

        return np.repeat(vals, cnt).reshape(*shape)
    
    def _encode(self, arr:np.ndarray):
        # Get dtype, shape, and size info
        olddt = arr.dtype
        shape = arr.shape
        
        # Compute RLE
        unq, cnt = self.unique_consecutive_with_counts(arr.flatten())
        table = np.unique(unq)
        maxunq = len(table)
        
        # Remap and determine new dtype
        mapunq = np.searchsorted(table, unq)
        newdt = np.min_scalar_type(max(maxunq, cnt.max()))
            
        # Construct rlebytes
        rle = np.stack((mapunq, cnt), axis=0)
        rlebytes = rle.astype(newdt).tobytes()
        
        # Construct header
        dtbyte = self.pack_dtypes(olddt, newdt)
            
        # Construct shape and table
        shapebytes = np.array(shape, dtype=np.uint64).tobytes()
        tablebytes = table.tobytes()
        
        # Determine lengths
        shapelen = len(shapebytes).to_bytes(8, 'big')
        tablelen = len(tablebytes).to_bytes(8, 'big')
        
        # Concatenate
        return dtbyte + shapelen + tablelen + shapebytes + tablebytes + rlebytes
    

class SEG8(EncoderDecoder):

    '''8-bit segmentation mask EncoderDecoder for LITDataset.

    Uses PNG encoding to compress a segmentation map of maximum 256 classes.

    NOTE: Why PNG instead of QOI, if it yields better compression speeds?
          Well, QOI on segmentation masks seem to be more or less similar
          in performance to RLE, but much slower. Here, PNG actually performs
          a lot better.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.seg8': '8 bit segmentation mask (PNG)',
        }
    
    def _encode(self, arr:np.ndarray):
        assert arr.dtype == np.uint8
        img = Image.fromarray(arr, 'L')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('L')
        return np.array(img)


class SEG16(EncoderDecoder):

    '''16-bit segmentation mask EncoderDecoder for LITDataset.

    Uses PNG encoding to compress a segmentation map of maximum 65536 classes.

    NOTE: Why PNG instead of QOI, if it yields better compression speeds?
          Well, QOI on segmentation masks seem to be more or less similar
          in performance to RLE, but much slower. Here, PNG actually performs
          a lot better.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.seg16': '16 bit segmentation mask (PNG)',
        }
    
    def _encode(self, arr:np.ndarray):
        assert arr.dtype == np.uint16
        img = Image.fromarray(arr, 'I;16')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('I;16')
        return np.array(img)
    

class SEG24(EncoderDecoder):

    '''24-bit segmentation mask EncoderDecoder for LITDataset.

    Uses PNG encoding to compress a segmentation map of maximum 16777216 classes.

    NOTE: Why PNG instead of QOI, if it yields better compression speeds?
          Well, QOI on segmentation masks seem to be more or less similar
          in performance to RLE, but much slower. Here, PNG actually performs
          a lot better.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.seg24': '24 bit segmentation mask (PNG|RGB)',
        }

    def _torgb(self, arr:np.ndarray) -> np.ndarray:
        b = (arr % self._bit8).astype(np.uint8)
        g = ((arr // self._bit8) % self._bit8).astype(np.uint8)
        r = (arr // self._bit16).astype(np.uint8)
        return np.stack((r,g,b), 2)

    def _fromrgb(self, arr:np.ndarray) -> np.ndarray:
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        return r * self._bit16 + g * self._bit8 + b

    def _encode(self, arr:np.ndarray):
        assert arr.dtype == np.uint32
        img = Image.fromarray(self._torgb(arr), 'RGB')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('RGB')
        return self._fromrgb(np.array(img))


class SEG32(EncoderDecoder):

    '''32-bit segmentation mask EncoderDecoder for LITDataset.

    Uses PNG encoding to compress a segmentation map of maximum 4294967296 classes.

    NOTE: You are very unlikely to need this level of resolution for segmentation tasks.

    NOTE: Why PNG instead of QOI, if it yields better compression speeds?
          Well, QOI on segmentation masks seem to be more or less similar
          in performance to RLE, but much slower. Here, PNG actually performs
          a lot better.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.seg32': '32 bit segmentation mask (PNG|RGBA)',
        }

    def _torgba(self, arr:np.ndarray) -> np.ndarray:
        a = (arr % self._bit8).astype(np.uint8)
        b = ((arr // self._bit8) % self._bit8).astype(np.uint8)
        g = ((arr // self._bit16) % self._bit16).astype(np.uint8)
        r = (arr // self._bit24).astype(np.uint8)
        return np.stack((r,g,b,a), 2)

    def _fromrgba(self, arr:np.ndarray) -> np.ndarray:
        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        return r * self._bit24 + g * self._bit16 + b * self._bit8 + a

    def _encode(self, arr:np.ndarray):
        assert arr.dtype == np.uint32
        img = Image.fromarray(self._torgba(arr), 'RGBA')
        byte_arr = BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        img = Image.open(byte_arr).convert('RGBA')
        return self._fromrgba(np.array(img))
    

class JSON(EncoderDecoder):
    '''JSON encoding / decoding for text, trees, string objects, etc.

    NOTE: Uses UTF-8. For other encodings, implement a custom EncoderDecoder.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.json': 'JSON file.',
        }

    def _encode(self, obj:Union[List, Dict]):
        return json.dumps(obj).encode()

    def _decode(self, data:bytes):
        return json.loads(data)


class NPY(EncoderDecoder):

    '''Numpy array encoding/decoding. 

    NOTE: Uses standard numpy.save and load without compression.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.npy': 'Numpy array file.',
        }
    
    def _encode(self, arr:np.ndarray):
        byte_arr = BytesIO()
        np.save(byte_arr, arr)
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        return np.load(byte_arr)


class MAT(EncoderDecoder):

    '''Matlab file array encoding/decoding.

    NOTE: Supports Matlab v>5. For v<=4, implement a custom EncoderDecoder.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.mat': 'Matlab file.',
        }
    
    def _encode(self, mdict:dict):
        byte_arr = BytesIO()
        spio.savemat(byte_arr, mdict)
        return byte_arr.getvalue()
    
    def _decode(self, data:bytes):
        byte_arr = BytesIO(data)
        return spio.loadmat(byte_arr)
    

class PKL(EncoderDecoder):

    '''Pickle encoding/decoding for Python objects.

    NOTE: Uses protocol 5. For other versions, implement a custom EncoderDecoder.
    '''

    def __init__(self, mode:str):
        super().__init__(mode)
        self.supported_extensions = {
            '.pkl': 'Python pickled object file.',
        }
    
    def _encode(self, obj):
        return pickle.dumps(obj)
    
    def _decode(self, data:bytes):
        return pickle.loads(data)



# Define default encoders

_default_decoder_list = [
    PIL.decoder(),
    CLS.decoder(),
    RLE.decoder(),
    SEG8.decoder(),
    SEG16.decoder(),
    SEG24.decoder(),
    SEG32.decoder(),
    JSON.decoder(),
    NPY.decoder(),
    MAT.decoder(),
    PKL.decoder(),
]

_default_encoder_list = [
    PIL.encoder(),
    CLS.encoder(),
    RLE.encoder(),
    SEG8.encoder(),
    SEG16.encoder(),
    SEG24.encoder(),
    SEG32.encoder(),
    JSON.encoder(),
    NPY.encoder(),
    MAT.encoder(),
    PKL.encoder(),
]

DEFAULT_DECODERS = {
    ext:dec for dec in _default_decoder_list for ext in dec.supported_extensions    
}

DEFAULT_ENCODERS = {
    ext:dec for dec in _default_encoder_list for ext in dec.supported_extensions    
}