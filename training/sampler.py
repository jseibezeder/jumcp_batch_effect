import random
import math

from torch.utils.data.sampler import Sampler
import numpy as np
import torch
from collections.abc import Iterator
from typing import Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset

def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]


class GroupSampler:
    r"""
        Samples batches of data from predefined groups.
    """

    def __init__(self, dataset, meta_batch_size, support_size,
                 drop_last=None, uniform_over_groups=True):

        self.dataset = dataset
        self.indices = range(len(dataset))

        self.group_ids = dataset.group_ids
        self.groups = dataset.groups
        self.num_groups = dataset.n_groups

        self.meta_batch_size = meta_batch_size
        self.support_size = support_size
        self.batch_size = meta_batch_size * support_size
        self.drop_last = drop_last
        self.dataset_size = len(self.dataset)
        self.num_batches = len(self.dataset) // self.batch_size

        self.groups_with_ids = {}
        self.actual_groups = []

        # group_count will have one entry per group
        # with the size of the group
        self.group_count = []
        for group_id in self.groups:
            ids = np.nonzero(self.group_ids == group_id)[0]
            self.group_count.append(len(ids))
            self.groups_with_ids[group_id] = ids

        self.group_count = np.array(self.group_count)
        self.group_prob = self.group_count / np.sum(self.group_count)
        self.uniform_over_groups = uniform_over_groups

    def __iter__(self):

        n_batches = len(self.dataset) // self.batch_size
        if self.uniform_over_groups:
            sampled_groups = np.random.choice(self.groups, size=(n_batches, self.meta_batch_size))
        else:
            # Sample groups according to the size of the group
            sampled_groups = np.random.choice(self.groups, size=(n_batches, self.meta_batch_size), p=self.group_prob)

        group_sizes = np.zeros(sampled_groups.shape)

        for batch_id in range(self.num_batches):

            sampled_ids = [np.random.choice(self.groups_with_ids[sampled_groups[batch_id, sub_batch]],
                                size=self.support_size,
                                replace=True,
                                p=None)
                                for sub_batch in range(self.meta_batch_size)]



            # Flatten
            sampled_ids = np.concatenate(sampled_ids)

            yield sampled_ids

        self.sub_distributions = None

    def __len__(self):
        return len(self.dataset) // self.batch_size

_T_co = TypeVar("_T_co", covariant=True)


#distributed sampler from pytorch
class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedGroupSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = False,
        meta_batch_size: int = 2, 
        support_size: int = 16,
        uniform_over_groups: bool = True,
        seed: int = 1234,
        distributed: bool = True
    ) -> None:

        if distributed:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
                )
        else:
            num_replicas=1
            rank=0
        

        self.dataset = dataset
        self.num_replicas = num_replicas            #how many devices there are
        self.rank = rank                            #current device
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        
        self.meta_batch_size = meta_batch_size
        self.support_size = support_size
        self.batch_size = meta_batch_size * support_size
        self.dataset_size = len(self.dataset)
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last or (len(self.dataset)/self.batch_size) % self.num_replicas == 0:  # type: ignore[arg-type]
            self.num_batches = int((len(self.dataset)/self.batch_size) / self.num_replicas)
        else:
            self.num_batches = int((len(self.dataset)/self.batch_size) / self.num_replicas) + int((4-((len(self.dataset)/self.batch_size) % self.num_replicas)))

        self.total_batches = self.num_batches * self.num_replicas

        #things relevant for Grouped Sampler
        self.group_ids = dataset.group_ids
        self.groups = dataset.groups
        self.num_groups = dataset.n_groups

        self.groups_with_ids = {}
        self.actual_groups = []

        # group_count will have one entry per group
        # with the size of the group
        self.group_count = []
        for group_id in self.groups:
            ids = np.nonzero(self.group_ids == group_id)[0]
            self.group_count.append(len(ids))
            self.groups_with_ids[group_id] = ids

        self.group_count = np.array(self.group_count)
        self.group_prob = self.group_count / np.sum(self.group_count)
        self.uniform_over_groups = uniform_over_groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.uniform_over_groups:
            sampled_groups = rng.choice(self.groups, size=(self.total_batches, self.meta_batch_size))
        else:
            # Sample groups according to the size of the group
            sampled_groups = rng.choice(self.groups, size=(self.total_batches, self.meta_batch_size), p=self.group_prob)

        all_grouped_batches = np.zeros((self.total_batches, self.batch_size))

        for batch_id in range(self.total_batches):

            sampled_ids = [rng.choice(self.groups_with_ids[sampled_groups[batch_id, sub_batch]],
                                size=self.support_size,
                                replace=True,
                                p=None)
                                for sub_batch in range(self.meta_batch_size)]



            # Flatten
            sampled_ids = np.concatenate(sampled_ids)
            all_grouped_batches[batch_id] = sampled_ids

        all_grouped_batches = np.array(all_grouped_batches)

        gpu_batches = all_grouped_batches[self.rank:: self.num_replicas]
        for batch in gpu_batches:
            yield batch.astype(int)

        self.sub_distributions = None

    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch