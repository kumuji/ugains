from typing import Any, List

import multiprocessing as mp
import pickle
from . import comm

import numpy as np

import torch


class NumpySharedData:
    """
    A class for wrapping a list of objects and sharing them across data workers.

    This class can be used to prevent memory explosion when using lists of objects
    in PyTorch DataLoaders. It serializes each object into a string using pickle,
    and stores the start address of each serialized string in memory.

    This class is intended to be used as a workaround for the issue described in
    https://github.com/pytorch/pytorch/issues/13246, which can occur when trying
    to use a list of dictionaries as input to a PyTorch DataLoader.

    Args:
        lst (List[Any]): A list of dictionaries to be wrapped.

    Attributes:
        _lst (np.ndarray): A serialized and concatenated version of the input list of dictionaries.
            1D[uint8] list.
        _addr (List[int]): A list of first addresses for each serialized dictionary in the `_lst`.
            The list of addresses starts from an address of element 1,
            as element 0 always have address 0.
    """

    def __init__(self, lst: List[Any]):
        self._lst = [self._serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)

    def __len__(self) -> int:
        """
        Returns:
            int: The length of the wrapped list.
        """
        return len(self._addr)

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx (int): The index of the dictionary to retrieve.

        Returns:
            Dict: The dictionary at the specified index.
        """
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        raw_bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(raw_bytes)

    @staticmethod
    def _serialize(data: Any) -> np.ndarray:
        """
        Serializes an object using pickle and returns it as a numpy array.

        Args:
            data (Any): An object to serialize.

        Returns:
            np.ndarray: The serialized object as a numpy array.
        """
        buffer = pickle.dumps(data, protocol=-1)
        return np.frombuffer(buffer, dtype=np.uint8)


class TorchSharedData(NumpySharedData):
    """
    This version takes the numpy implementation, but it uses fully shared memmory.
    """

    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class SharedData(TorchSharedData):
    def __init__(self, lst: list):
        if comm.get_local_rank() == 0:
            super().__init__(lst)
        if comm.get_local_size() == 1:
            # Just one GPU on this machine. Do nothing.
            return
        if comm.get_local_rank() == 0:
            # Move to shared memory, obtain a handle.
            serialized = bytes(
                mp.reduction.ForkingPickler.dumps((self._addr, self._lst))
            )
            # Broadcast the handle of shared memory to other GPU workers.
            comm.all_gather(serialized)
        else:
            serialized = comm.all_gather(None)[comm.get_rank() - comm.get_local_rank()]
            # Materialize a tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(serialized)
            print(
                f"Worker {comm.get_rank()} obtains a dataset of length="
                f"{len(self)} from its local leader."
            )
