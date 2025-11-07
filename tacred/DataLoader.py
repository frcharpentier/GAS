from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union
import numpy as np
import random
 
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


class tacredStratifiedGraphSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,
        dataset,
        avg_num,
        tolerance = 0.1,
        mode = 'node',
        ):
        
        if mode != "node":
            raise NotImplementedError
        self.dataset = dataset
        self.nbGraphs    = len(dataset)
        self.avg_num = avg_num
        self.mode = mode
        self.tolerance = tolerance
        self.effectifs = self.dataset.get_num_nodes()
        num_nodes = sum(self.effectifs)
        self.num_batches = int(0.5 + (num_nodes / avg_num))
        self.index, self.cumul = self.dataset.get_strates()
        

    def __len__(self):
        return self.num_batches
    

    def draw_batch_with_replacement(self, effectifs, mini, maxi):
        cur_sample = ()
        #eff_sample = ()
        cumul_eff = 0
        N = len(self.cumul) -1
        while True:
            i = random.randint(0, N-1)
            j = i+1
            elt = random.choice(self.index[self.cumul[i]:self.cumul[j]])
            eff = effectifs[elt]
            if eff > maxi:
                continue
            elif cumul_eff + eff <= maxi:
                cur_sample += (elt,)
                #eff_sample += (eff,)
                cumul_eff += eff
            else:
                if cumul_eff >= mini:
                    yield cur_sample
                # Tout recommencer
                cur_sample = ()
                #eff_sample = ()
                cumul_eff = 0

    def __iter__(self):
        maxi = int(0.5 + (1+self.tolerance)*self.avg_num)
        mini = int(0.5 + (1-self.tolerance)*self.avg_num)
        N = 0
        iterat = self.draw_batch_with_replacement(self.effectifs, mini, maxi)
        for _ in range(self.num_batches):
            spl = next(iterat)
            yield spl

        

class tacredGraphSampler(torch.utils.data.sampler.Sampler):
    r"""Classe qui sélectionne des indices pour regrouper les graphes
    en batches dont le nombre total de sommets vaut avg_num.
    Une tolérance en plus ou en moins est autorisée (10% par défaut.)

    Args:
        dataset (Dataset): Dataset to sample from.
        avg_num (int): Size of mini-batch to aim for in number of nodes or
            edges.
        tolerance (float): La tolérance. Un nombre réel. (0,1 signifie 10%.)
        mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
            batch size. (default: :obj:`"node"`)
        shuffle: Une fois les graphes associés en batches, ce paramètre détermine
        s’il faut mélanger ou non l’ordre de ces batches, à chaque époque.
    """
    def __init__(self,
        dataset,
        avg_num,
        tolerance = 0.1,
        mode = 'node',
        shuffle=False,
        ):
        
        if mode != "node":
            raise NotImplementedError
        self.dataset = dataset
        self.nbGraphs    = len(dataset)
        self.avg_num = avg_num
        self.mode = mode
        self.tolerance = tolerance
        self.shuffle = shuffle
        
        self.brasser()
        

    def construire_strates(self):
        self.strates = [None for _ in self.dataset.dico_relations]
        for R in range(len(self.strates)):
            self.strates[R] = np.array([i for i,X in enumerate(self.dataset) if X.y1 == R])
        



    @staticmethod
    def regroupement_aleatoire(elements, effectifs, mini, maxi):
        # fonction qui regroupe les éléments de façon que leurs effectifs
        # cumulés soient compris entre mini et maxi.
        # renvoie une liste de groupes, et une liste d’éléments non regroupés,
        # pour recommencer.
        #assert len(elements) == len(effectifs)
        assert mini <= maxi
        indices = torch.randperm(len(effectifs)).tolist()
        samples = []
        mineurs = []
        eff_mineurs = []
        cur_sample = ()
        eff_sample = ()
        
        cumul_eff = 0
        NNN = len(effectifs)
        
        for i in indices:  #tqdm(indices, desc="regroupement"):
            eff = effectifs[i]
            elt = elements[i]
            if eff > maxi:
                continue
            elif cumul_eff + eff <= maxi:
                cur_sample += (elt,)
                eff_sample += (eff,)
                cumul_eff += eff
            else:
                if cumul_eff < mini:
                    mineurs.append(cur_sample)
                    eff_mineurs.extend(eff_sample)
                else:
                    samples.append(cur_sample)
                
                cur_sample = (elt,)
                eff_sample = (eff,)
                cumul_eff = eff
        if len(cur_sample) > 0:
            if cumul_eff < mini:
                mineurs.append(cur_sample)
                eff_mineurs.extend(eff_sample)
            else:
                samples.append(cur_sample)

        return samples, mineurs, eff_mineurs
    
   

    def brasser(self):
        samples = []
        effectifs = self.dataset.get_num_nodes()
        N = len(effectifs)
        elements = [x for x in range(N)]
        maxi = int(0.5 + (1+self.tolerance)*self.avg_num)
        mini = int(0.5 + (1-self.tolerance)*self.avg_num)
        while True:
            samples0, mineurs, eff_mineurs = tacredGraphSampler.regroupement_aleatoire(
                elements, effectifs, mini, maxi)
            samples.extend(samples0)
            if len(eff_mineurs) == 0:
                break
            if N == len(eff_mineurs):
                samples.extend(mineurs)
                break
            elements = []
            for mn in mineurs:
                elements.extend(mn)
            effectifs = eff_mineurs
            N = len(effectifs)
        self.samples = samples
        self.L = len(samples)
        

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.L).tolist()
        else:
            indices = range(self.L)
        for i in indices:
            yield list(self.samples[i])

    def __len__(self):
        return self.L



class tacredCollater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            B = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            cumul = B.ptr[B.msk_batch]
            B.msk = B.msk + cumul
            return B
        

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class tacredDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = ["msk"],
        exclude_keys: Optional[List[str]] = None,
        forceGPU = False,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)
        if not "msk" in follow_batch:
            follow_batch.append("msk")
        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys


        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=tacredCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )

class tacredGPUDataLoader(tacredDataLoader):
    def __iter__(self):
        for b in super().__iter__():
            b = b.to(torch.device("cuda"))
            yield b

    
