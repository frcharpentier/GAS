import os
import os.path as osp
from pathlib import Path
import numpy as np
import json
import struct

import torch
from torch.utils.data import Dataset as torchDataset
from torch_geometric.data import Dataset as geoDataset, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DynamicBatchSampler
import torch_geometric.transforms as TRF
from graphe_adjoint import TRANSFORMER_ATTENTION, faire_graphe_adjoint
from collections import OrderedDict, defaultdict, namedtuple
from tqdm import tqdm
from inspect import isfunction
from liste_tous_roles import cataloguer_roles, liste_roles, sourcer_fichier_txt
import random
import hashlib


class FusionElimination(TRF.BaseTransform):
    # ef : effectif
    # li : liste des classes associées à un numéro
    # al : alias associé à un numéro (nom considéré comme canonique)
    # no : numéro d’une classe

    Ntup = namedtuple("Ntup", ["no", "al", "ef", "li"])

    def __init__(self, nom_json=None, index=None, noms_classes=None, effectifs=None, alias=None, dico_ARGn=None):
        if index is None:
            assert nom_json is not None
            #nom_json = osp.join(repertoire, 'processed', 'liste_roles.json')
            with open(nom_json, "r", encoding="UTF-8") as F:
                jason = json.load(F)
            dico_roles = OrderedDict([tuple(t) for t in jason["dico_roles"]])
            if dico_ARGn is None:
                dico_ARGn = OrderedDict([tuple(t) for t in jason["dico_ARGn"]])
            index=[i for i, _ in enumerate(dico_roles)]
            if noms_classes is None:
                noms_classes = [[k] for k in dico_roles]
            if effectifs is None:
                #effectifs = [v[0] for _,v in dico_roles.items()]
                effectifs = [v for _,v in dico_roles.items()]
            if alias is None:
                alias = [k for k in dico_roles]

        assert type(index) in (list, tuple)
        self.nb_classes = 1+max(index)
        assert all(idx in index for idx in range(self.nb_classes))

        self.index = torch.tensor([0 if i < 0 else i for i in index], dtype=torch.int8)
        if any(idx < 0 for idx in index):
            self.eliminVF = True
            self.agarder = torch.tensor([idx >=0 for idx in index], dtype=torch.bool)
        else:
            self.agarder = None
            self.eliminVF = False

        
        if noms_classes:
            self.noms_classes = noms_classes
        else:
            self.noms_classes = None
        if effectifs:
            self.effectifs = effectifs
        else:
            self.effectifs = None
        if alias:
            self.alias = alias
        else:
            self.alias = None
        if dico_ARGn:
            self.dico_ARGn = dico_ARGn
        else:
            self.dico_ARGn = None
            
    def __getitem__(self, clef):
        # syntaxe : machin["al#12"] : pour obtenir l’alias de la 12e classe
        # machin["ef@PATIENT"] : pour obtenir l’effectif de la classe nommée "PATIENT"
        # machin["li@AGENT"] : pour obtenir la liste de toutes les classes fusionnées avec la classe nommée AGENT
        # machin["no@THEME"] : pour obtenir le numéro de la classe nommée THÈME.
        assert type(clef) is str
        assert len(clef) > 3 and clef[:2] in ("al", "ef", "li", "no") and clef[2] in "#@"
        if clef.startswith("al"):
            table = self.alias
        elif clef.startswith("ef"):
            table = self.effectifs
        elif clef.startswith("li"):
            table = self.noms_classes
        if clef[2] == "#":
            i = int(clef[3:])
        elif clef[2] == "@":
            klef = clef[3:]
            if klef in self.alias:
                i = self.alias.index(klef)
            else:
                for i, li in enumerate(self.noms_classes):
                    if klef in li:
                        break
                else:
                    raise KeyError
        if clef.startswith("no"):
            return i
        else:
            return table[i]
    

    def __setitem__(self, clef, valeur):
        # Uniquement pour définir un alias ou un effectif.
        assert type(clef) is str
        assert len(clef) > 3 and clef[:2] in ("al", "ef") and clef[2] in "#@"
        if clef.startswith("al"):
            table = self.alias
        elif clef.startswith("ef"):
            table = self.effectifs
        if clef[2] == "#":
            i = int(clef[3:])
        elif clef[2] == "@":
            klef = clef[3:]
            if klef in self.alias:
                i = self.alias.index(klef)
            else:
                for i, li in enumerate(self.noms_classes):
                    if klef in li:
                        break
                else:
                    raise KeyError
        table[i] = valeur

    def eliminer(self, *args):
        N = 0
        index = []
        alias = []
        noms_classes = []
        effectifs = []
        djvus = dict()
        if isfunction(args[0]):
            f = args[0]
            for i, idx in enumerate(self.index.numpy()):
                if self.eliminVF and (not self.agarder[i]):
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                tup = FusionElimination.Ntup(no=i, al=al, ef=ef, li=li)
                if not f(tup):
                    #garder
                    if not idx in djvus:
                        djvus[idx] = N
                        index.append(N)
                        alias.append(al)
                        noms_classes.append(li)
                        effectifs.append(ef)
                        N += 1
                    else:
                        index.append(djvus[idx])
                else:
                    #Éliminer
                    index.append(-1)
                    
        else:
            for i, idx in enumerate(self.index.numpy()):
                if self.eliminVF and (not self.agarder[i]):
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if not al in args:
                    #garder
                    if not idx in djvus:
                        djvus[idx] = N
                        index.append(N)
                        alias.append(al)
                        noms_classes.append(li)
                        effectifs.append(ef)
                        N += 1
                    else:
                        index.append(djvus[idx])
                else:
                    #Éliminer
                    index.append(-1)
                    
        
        return FusionElimination(index=index, noms_classes=noms_classes, effectifs=effectifs, alias=alias, dico_ARGn=self.dico_ARGn)


    def garder(self, *args):
        N = 0
        index = []
        alias = []
        noms_classes = []
        effectifs = []
        djvus = dict()
        if isfunction(args[0]):
            f = args[0]
            for i, idx in enumerate(self.index.numpy()):
                if self.eliminVF and (not self.agarder[i]):
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if f(FusionElimination.Ntup(no=i, al=al, ef=ef, li=li)):
                    #garder
                    if not idx in djvus:
                        djvus[idx] = N
                        index.append(N)
                        alias.append(al)
                        noms_classes.append(li)
                        effectifs.append(ef)
                        N += 1
                    else:
                        index.append(djvus[idx])
                else:
                    #Éliminer
                    index.append(-1)
                    
        else:
            for i, idx in enumerate(self.index.numpy()):
                if self.eliminVF and (not self.agarder[i]):
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if al in args:
                    #garder
                    if not idx in djvus:
                        djvus[idx] = N
                        index.append(N)
                        alias.append(al)
                        noms_classes.append(li)
                        effectifs.append(ef)
                        N += 1
                    else:
                        index.append(djvus[idx])
                else:
                    #Éliminer
                    index.append(-1)
                    
        return FusionElimination(index=index, noms_classes=noms_classes, effectifs=effectifs, alias=alias, dico_ARGn=self.dico_ARGn)
                    

    def fusionner(self, *args):
        if isfunction(args[0]):
            f = args[0]
            dico0 = {}
            for i, idx in enumerate(self.index.numpy()):
                if self.eliminVF and (not self.agarder[i]):
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                truc = f(FusionElimination.Ntup(no=i, al=al, ef=ef, li=li))
                if truc in dico0:
                    if al != truc:
                        dico0[truc].add(al)
                elif truc == al:
                    dico0[truc] = set()
                else:
                    dico0[truc] = set([al])
            listeargus = [ list(v)+ [k] for k,v in dico0.items()]
            

        else:
            listeargus = args
        dico = {}
        index = []
        alias = []
        noms_classes = []
        effectifs = []
        N = 0
        for i, idx in enumerate(self.index.numpy()):
            if self.eliminVF and (not self.agarder[i]):
                index.append(-1)
                continue
            al = self.alias[idx]
            ef = self.effectifs[idx]
            li = self.noms_classes[idx]
            if al in dico:
                idy = dico[al]
                index.append(idy)
                noms_classes[idy].extend(self.noms_classes[idx])
                effectifs[idy] += self.effectifs[idx]
            else:
                for lis in listeargus:
                    if al in lis:
                        for elt in lis:
                            dico[elt] = N
                        alias.append(lis[-1])
                        break
                index.append(N)
                noms_classes.append([elt for elt in self.noms_classes[idx]])
                effectifs.append(self.effectifs[idx])
                N += 1
        
        return FusionElimination(index=index, noms_classes=noms_classes, effectifs=effectifs, alias=alias, dico_ARGn=self.dico_ARGn)  

    def forward(self, data):
        # data contient : y1=roles, y2=sens, ARGn,
        # msk1=msk_roles, msk2 = msk_sens,
        # msk_iso et msk_ARGn.
        roles = data.y1.to(dtype=torch.int)
        data.y1 = self.index[roles]
        if self.eliminVF:
            data.msk1 = data.msk1 & self.agarder[roles]
        return data

class PermutEdgeDataset(torchDataset):
    def __init__(self, edgeDS, permut):
        # permut est une liste indexée par le numéro de la classe dans le dataset de départ
        # et dont la valeur à chaque indice donne le nouveau numéro de la classe
        # permut peut aussi être la liste des classes dans l’ordre souhaité
        self.edgeDS = edgeDS
        nb_classes = len(edgeDS.liste_rolARG)
        assert type(permut) in (list, tuple)
        permut = list(permut)
        
        if all(type(i) == str for i in permut):
            assert len(permut) == nb_classes
            assert all(c in edgeDS.liste_rolARG for c in permut)
            assert all(c in permut for c in edgeDS.liste_rolARG)
            prmt = [None] * nb_classes
            for i, C in enumerate(permut):
                j = edgeDS.liste_rolARG.index(C)
                prmt[j] = i
            permut = prmt
         
        assert all(type(i) == int for i in permut)
        mini = min(permut)
        N = 1 +max(permut)
        assert len(permut) == N - mini
        assert all(i in permut for i in range(mini, N))
        # Toutes les lignes ci-dessus servent à vérifier que permut encode une permutation
        assert mini >= 0
        assert N <= nb_classes
        # Ces lignes pour vérifier que permut encode une permutation d’un sous ensemble de {0,...,classe_max}

        if mini > 0:
            permut = list(range(0, mini)) + permut
        if N < nb_classes:
            permut = permut + list(range(N, nb_classes))
        
        self.permut = torch.tensor(permut)
        liste_rolARG = [None] * nb_classes
        freqARGn = [None] * nb_classes
        for i in range(nb_classes):
            j = permut[i]
            liste_rolARG[j] = edgeDS.liste_rolARG[i]
            freqARGn[j] = edgeDS.freqARGn[i].item()
        self.liste_rolARG = liste_rolARG
        self.freqARGn = torch.tensor(freqARGn)

    @staticmethod
    def renum_freq(edgeDS, ordre):
        assert ordre in ["croissant", "décroissant", "decroissant"]
        if ordre == "decroissant":
            ordre = "décroissant"
        if ordre == "décroissant":
            reverse = True
        else:
            reverse = False
        invpermut = [i for i, t in sorted(enumerate(edgeDS.freqARGn), key=lambda x: x[1], reverse=reverse)]
        permut = [None] * len(invpermut)
        for i,j in enumerate(invpermut):
            permut[j] = i

        return PermutEdgeDataset(edgeDS, permut)


    def __len__(self):
        return self.edgeDS.Nadj
    
    def __getitem__(self, idx):
        return {
            "X": self.edgeDS.X[idx].to(dtype=torch.float32),
            "sens": self.edgeDS.sens[idx],
            "ARGn": self.permut[self.edgeDS.ARGn[idx]].to(dtype=torch.long)
        }




class EdgeDataset(torchDataset):
    def __init__(self, aligDS, repertoire, masques = "1 2 iso"): #, transform=None):
        self.gros_fichier = aligDS.processed_paths[0]
        self.filtre = aligDS.filtre
        
        Path(repertoire).mkdir(parents=True, exist_ok=True)
        self.fichier_edge = osp.join(repertoire, "edge_data.bin")
        self.fichier_edge_labels = osp.join(repertoire, "edge_labels.bin")
        self.fichier_ARGn_labels = osp.join(repertoire, "edge_ARGn.bin")
        self.fichier_sens = osp.join(repertoire, "edge_dir.bin")
        self.fichier_digests = osp.join(repertoire, "digests.txt")
        self.liste_roles = None
        masques = masques.split()
        assert 1 <= len(masques) <= 3
        assert all(x in ("1", "2", "iso") for x in masques)
        self.masques = masques
        self.debug_idSNT = aligDS.debug_idSNT
        if self.debug_idSNT:
            self.liste_idSNT = aligDS.liste_idSNT
        self.calculer_ou_lire(aligDS)
        self.calc_freq()

    @staticmethod
    def calculer_digests_filtre(filtre):
        HH = hashlib.new("md5")
        HH.update(repr([k for k in filtre.alias]).encode("UTF-8"))
        HH.update(repr([k for k in filtre.dico_ARGn]).encode("UTF-8"))
        HH.update(repr([ef for ef in filtre.effectifs]).encode("UTF-8"))
        return HH.hexdigest()

    def calculer_ou_lire(self, aligDS):
        calculer = True
        HH = EdgeDataset.calculer_digests_filtre(self.filtre)
        with open(aligDS.processed_paths[3], "r", encoding="UTF-8") as F: #fichier "digests.txt"
            ref_digests = json.load(F)
        if osp.exists(self.fichier_digests):
            if osp.exists(self.fichier_edge) and osp.exists(self.fichier_edge_labels) and osp.exists(self.fichier_ARGn_labels):
                with open(self.fichier_digests, "r", encoding="UTF-8") as F:
                    digests = json.load(F)
                verification = True
                if digests["gros_fichier"] != ref_digests["gros_fichier"]:
                    verification = False
                if verification:
                    if HH != digests["filtre_roles"]:
                        verification = False
                if verification:
                    if not "masques" in digests:
                        verification = False
                if verification:
                    if not all(x in self.masques for x in digests["masques"]):
                        verification = False
                    if not all(x in digests["masques"] for x in self.masques):
                        verification = False
                if verification: # vérifier les digests
                    calculer = False
        if calculer:
            dico = {"gros_fichier": ref_digests["gros_fichier"],
                    "filtre_roles": HH,
                    "masques": self.masques}
            with open(self.fichier_digests, "w", encoding="UTF-8") as F:
                json.dump(dico, F)
            self.process(aligDS)
            self.read_files()
        else:
            if self.debug_idSNT:
                self.etablir_table_debug(aligDS)
            self.read_files()

    def lire_liste_roles(self):
        if self.liste_roles is None: 
            self.dico_ARGn = self.filtre.dico_ARGn
            self.liste_roles = [k for k in self.filtre.alias]
            self.liste_ARGn = [k for k in self.dico_ARGn]
            self.dico_roles = OrderedDict([(k,v) for k,v in zip(self.filtre.alias, self.filtre.effectifs)])
            self.liste_rolARG = self.liste_roles + self.liste_ARGn # On met les roles de type ":ARGn" à la fin.

            with open(self.gros_fichier, "rb") as F:
                ligne = F.readline().decode("ascii").strip()
            self.dimension = int(ligne)

    def calc_freq(self):
        assert self.roles.shape == (self.Nadj,)
        uns = torch.ones((self.Nadj,), dtype=torch.int)
        cumRoles = torch.zeros((len(self.liste_roles),), dtype=torch.int)
        indices = self.roles.to(dtype = torch.long)
        cumRoles.scatter_add_(0, indices, uns)

        assert self.roles.shape == (self.Nadj,)
        cumARGn = torch.zeros((len(self.liste_rolARG),), dtype=torch.int)
        indices = self.ARGn.to(dtype = torch.long)
        cumARGn.scatter_add_(0, indices, uns)

        assert self.roles.shape == (self.Nadj,)
        cumSens = torch.tensor([0, 0], dtype=torch.int)
        indices = self.sens.to(dtype = torch.long)
        cumSens.scatter_add_(0, indices, uns)

        self.freqRoles = cumRoles / self.Nadj
        self.freqARGn  = cumARGn / self.Nadj
        self.freqSens  = cumSens / self.Nadj 



    def etablir_table_debug(self, aligDS):
        print("Établissement de la table de debug")
        self.table_debug = [0]
        NN = 0
        msk_idx = [["1", "2", "iso"].index(m) for m in self.masques]
        msk_idx.sort()
        for data in tqdm(aligDS):
            masques = (data.msk1, data.msk2, data.msk_iso)
            msk = masques[msk_idx[0]]
            for mmm in msk_idx[1:]:
                msk = msk & masques[mmm]
            idx = torch.nonzero(msk).view(-1)
            (N,) = idx.shape
            NN += N
            self.table_debug.append(NN)


    def get_range_debug(self, idx):
        if not self.debug_idSNT:
            raise NotImplementedError
        if type(idx) == str:
            idx = self.liste_idSNT.index(idx)
        return self.table_debug[idx], self.table_debug[idx+1]



    def process(self, aligDS):
        print("Construction des fichiers")
        self.lire_liste_roles()
        FX = open(self.fichier_edge, "wb")             #fichier edge_data.bin
        if self.debug_idSNT:
            self.table_debug = [0]
            NN = 0
        Froles = open(self.fichier_edge_labels, "wb")  #fichier edge_labels.bin
        FARGn = open(self.fichier_ARGn_labels, "wb")   #fichier edge_ARGn.bin
        Fsens = open(self.fichier_sens, "wb")          #fichier edge_dir.bin
        msk_idx = [["1", "2", "iso"].index(m) for m in self.masques]
        msk_idx.sort()
        try:
            for data in tqdm(aligDS):
                masques = (data.msk1, data.msk2, data.msk_iso)
                msk = masques[msk_idx[0]]
                for mmm in msk_idx[1:]:
                    msk = msk & masques[mmm]
                idx = torch.nonzero(msk).view(-1)
                (N,) = idx.shape
                if self.debug_idSNT:
                    NN += N
                    self.table_debug.append(NN)
                if N == 0:
                    continue
                X = data.x[idx].contiguous()
                shape = X.shape
                assert (shape[1], shape[2]) == (self.dimension,2)
                assert X.dtype == torch.bfloat16
                octets = X.view(dtype=torch.int16).numpy().reshape(-1).tobytes()
                assert len(octets) == shape[0] * self.dimension * 2 * 2
                FX.write(octets)
                #self.X = X

                roles = data.y1[idx].contiguous() #role
                assert roles.dtype == torch.int8
                Froles.write(roles.numpy().reshape(-1).tobytes())
                #self.roles = roles

                sens = data.y2[idx].contiguous()
                #self.sens = sens
                sens = sens.to(dtype=torch.int8) #sens
                Fsens.write(sens.numpy().reshape(-1).tobytes())

                msk_argus = data.msk_ARGn[idx].contiguous()
                assert msk_argus.dtype == torch.bool
                ARGn = data.ARGn[idx] + len(self.liste_roles)
                # décaler tous les numéros ARGn pour qu’ils prennent place à la fin de la liste des rôles normaux
                assert ARGn.dtype == torch.int8
                ARGn = (ARGn * msk_argus) + (roles * (~msk_argus))
                assert ARGn.dtype == torch.int8
                FARGn.write(ARGn.numpy().reshape(-1).tobytes())
                #self.ARGn = ARGn
        except:
            raise
        finally:
            FX.close()
            Froles.close()
            FARGn.close()
            Fsens.close()

            
            

    def read_files(self):
        self.lire_liste_roles()
        with open(self.fichier_edge, "rb") as F:
            grfSig = np.fromfile(F, dtype="int16").reshape((-1, self.dimension, 2))
        self.X = torch.as_tensor(grfSig).view(dtype=torch.bfloat16)
        (Nadj, dim, deux) = self.X.shape
        assert dim == self.dimension
        assert deux == 2
        with open(self.fichier_edge_labels, "rb") as F:
            self.roles = torch.as_tensor(np.fromfile(F, dtype="int8"))
        assert self.roles.dtype == torch.int8
        assert self.roles.shape == (Nadj,)
        with open(self.fichier_ARGn_labels, "rb") as F:
            self.ARGn = torch.as_tensor(np.fromfile(F, dtype="int8"))
        assert self.ARGn.dtype == torch.int8
        assert self.ARGn.shape == (Nadj,)
        with open(self.fichier_sens, "rb") as F:
            sens = torch.as_tensor(np.fromfile(F, dtype="int8"))
        assert sens.dtype == torch.int8
        assert sens.shape == (Nadj,)
        #self.sens = sens.to(dtype=torch.bfloat16)
        self.sens = sens.to(dtype=torch.float32)
        self.Nadj = Nadj


    def __len__(self):
        return self.Nadj

    def __getitem__(self, idx):
        return {
            "X": self.X[idx].to(dtype=torch.float32),
            "roles": self.roles[idx].to(dtype=torch.long),
            "sens": self.sens[idx],
            "ARGn": self.ARGn[idx].to(dtype=torch.long)
        }

    def redresser_X(self, direction, reshape=True):
        # Direction est un tenseur qui ne contient que des zéros et des uns.
        if not direction.dtype == torch.long:
            direction = direction.to(dtype = torch.long)
        indices = torch.column_stack((direction, 1-direction)).view(-1,1,2)
        #assert all(indices[i, 0, 0] == T[i] for i in range(self.Nadj))
        #assert all(indices[i, 0, 1] == 1-T[i] for i in range(self.Nadj))
        X = torch.take_along_dim(self.X, indices, dim=2)
        if reshape:
            X = X.transpose(1,2).reshape(-1, 2*self.dimension)
        if False and self.debug_idSNT:
            print("Vérification de la permutation de X :")
            for i in tqdm(range(self.Nadj)):
                d = direction[i]
                if d == 0:
                    assert all(X[i,k].item() == self.X[i,k,0].item() for k in range(self.dimension))
                    assert all(X[i,k + self.dimension].item() == self.X[i,k,1].item() for k in range(self.dimension))
                else:
                    assert all(X[i,k].item() == self.X[i,k,1].item() for k in range(self.dimension))
                    assert all(X[i,k + self.dimension].item() == self.X[i,k,0].item() for k in range(self.dimension))
            print("Vérification terminée") 
        self.X = X.contiguous()

    def permuter_X1_X2(self, direction):
        # On inverse l’ordre des deux vecteurs colonnes de X, ainsi que l’étiquette "sens"
        # L’inversion est cohérente.
        typ = self.sens.dtype
        sens = ((self.sens != direction)*1).to(dtype=typ)
        self.sens = sens
        self.redresser_X(direction, reshape=False)
    

class EdgeDatasetMono(EdgeDataset):
    # Dataset d’arêtes, où les descripteurs CS et SC sont concaténés dans
    # un vecteur de double dimension, dans un ordre déterminé par le sens
    # de l’arête
    def __init__(self, aligDS, repertoire):
        super().__init__(aligDS, repertoire)
        self.redresser_X(self.sens)

class EdgeDatasetMonoEnvers(EdgeDataset):
    # Dataset d’arêtes, où les descripteurs CS et SC sont concaténés dans
    # un vecteur de double dimension, dans un ordre déterminé par le sens
    # de l’arête, le sens inverse du dataset ci-dessus
    def __init__(self, aligDS, repertoire):
        super().__init__(aligDS, repertoire)
        self.redresser_X(1-self.sens)
        
class EdgeDatasetRdmDir(EdgeDataset):
    # Dataset d’arêtes, où les descripteurs CS et SC sont concaténés dans
    # un vecteur de double dimension, dans un ordre aléatoire
    def __init__(self, aligDS, repertoire):
        super().__init__(aligDS, repertoire)
        self.permutation_aleatoire()

    def permutation_aleatoire(self):
        self.redresser_X(torch.randint(0,2,(self.Nadj,), dtype=torch.long))

    def redirection_aleatoire(self):
        X = self.X.reshape(-1, 2, self.dimension).transpose(1,2).contiguous()
        self.X = X
        self.redresser_X(torch.randint(0,2,(self.Nadj,), dtype=torch.long))

                            




class AligDataset(geoDataset):
    def __init__(self, root, alignment_file, transform=None, pre_transform=None, pre_filter=None, split=False, QscalK=False, debug_idSNT=False, device="cpu"):
        if not split:
            self.split = False
        else:
            assert split in ["test", "dev", "train"]
            self.split = split
            if not alignment_file.endswith("_%s.txt"%split):
                if alignment_file.endswith(".txt"):
                    alignment_file = alignment_file[:-4]
                alignment_file += "_%s.txt"%split
        self.alignment_file = alignment_file
        self.digests = dict()
        self.offsets = None
        self.FileHandle = None
        self.liste_roles = None
        self.liste_ARGn = None
        self.QscalK = QscalK
        self.debug_idSNT = debug_idSNT
        self.device = device
        self.filtre = transform # Éventuellement remplacé plus tard si cette valeur est None.
        super().__init__(root, transform, pre_transform, pre_filter)
        if self.transform is None:
            self.filtre = FusionElimination(nom_json=self.processed_paths[2])
        self.check_consistency()
        


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return []
        return ['gros_fichier.bin', 'pointeurs.bin', 'liste_roles.json', 'digests.txt']
    
    def check_consistency(self):
        self.ouvrir_digests()
        try:
            if not "attention_type" in self.digests:
                assert self.QscalK == True
            elif self.QscalK:
                assert self.digests["attention_type"] == "QscalK"
            else:
                assert self.digests["attention_type"] == "softmax"
            assert "fichier_source" in self.digests

            with open(self.alignment_file, "rb") as F:
                digest = hashlib.file_digest(F, "md5")
            digest_source = digest.hexdigest()
            assert digest_source == self.digests["fichier_source"]
        
        except:
            message = "CAUTION ! The torch geometric dataset located in the folder %s "
            message += "was not build from the %s alignment file, "
            message += "or was not built with the QscalK=%s parameter"
            message = message%(self.root, self.alignment_file, ("True" if self.QscalK else "False"))
            raise Exception(message)


    def ecrire_liste_roles(self):
        if self.split:
            suffixe = "_%s.txt"%self.split
            assert self.alignment_file.endswith(suffixe)
            fichier = self.alignment_file[:-len(suffixe)]
            fichier_train = fichier + "_train.txt"
            fichier_dev   = fichier + "_dev.txt" 
            fichier_test  = fichier + "_test.txt"
            dico = cataloguer_roles(sourcer_fichier_txt(fichier_train))
            dico = cataloguer_roles(sourcer_fichier_txt(fichier_dev), dico)
            dico = cataloguer_roles(sourcer_fichier_txt(fichier_test), dico)
            dico_roles, dico_ARGn = liste_roles(dico=dico)
        else:
            dico_roles, dico_ARGn = liste_roles(nom_fichier=self.alignment_file)
        self.dico_roles = dico_roles
        self.dico_ARGn = dico_ARGn
        self.liste_roles = [k for k in self.dico_roles]
        self.liste_ARGn = [k for k in self.dico_ARGn]
        jason = dict()
        jason["dico_roles"] = [(k,v) for k,v in dico_roles.items()]
        jason["dico_ARGn"] = [(k,v) for k,v in dico_ARGn.items()]
        with open(self.processed_paths[2], "w", encoding="UTF-8") as F: #fichier "liste_roles.json"
            json.dump(jason, F)
        with open(self.processed_paths[2], "rb") as F: #fichier "liste_roles.json"
            HH = hashlib.file_digest(F, "md5")
        self.digests["liste_roles"] = HH.hexdigest()
        
    def compter_graphes(self):
        if self.debug_idSNT:
            self.liste_idSNT = []
        total_graphes = 0
        lbl_id = "# ::id "
        with open(self.alignment_file, "r", encoding="utf-8") as F:
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith(lbl_id):
                    total_graphes += 1
                    if self.debug_idSNT:
                        idSNT = ligne.split()[2]
                        self.liste_idSNT.append(idSNT)
        return total_graphes

    def process(self):
        idx = 0
        offsets = []
        print("Entrée fonction process")
        lbl_id = "# ::id "
        lbl_modname = "# ::model_name "
        
        self.ecrire_liste_roles()
        total_graphes = self.compter_graphes()

        with open(self.alignment_file, "rb") as F:
            digest = hashlib.file_digest(F, "md5")
        self.digests["fichier_source"] = digest.hexdigest()

        etat = 0
        model_name=None
        nb_graphes = 0
        attn = TRANSFORMER_ATTENTION(QscalK=self.QscalK, device=self.device)
        pbar = tqdm(total = total_graphes)

        # Test du format des floats
        T = torch.ones((1,), dtype=torch.bfloat16)*1000/3
        buf = T.to(dtype=torch.float32).numpy().tobytes()[2:4]
        assert buf == bytes([0xa7, 0x43])

        HH = hashlib.new("md5")
        with open(self.processed_paths[0], "wb") as FF: # fichier "gros_fichier.bin"
            with open(self.alignment_file, "r", encoding="utf-8") as F:
                depart = True
                for ligne in F:
                    ligne = ligne.strip()
                    if depart and ligne.startswith(lbl_modname):
                        model_name = ligne[len(lbl_modname):]
                        if "://" in model_name:
                            assert (model_name.startswith("minbert://")
                                    or model_name.startswith("mingpt://")
                                    or model_name.startswith("huggingface://")
                                    or model_name.startswith("hf://"))
                        elif model_name.startswith("bert") or model_name.startswith("roberta"):
                            model_name = "minbert://"+ model_name
                        elif model_name.startswith("gpt"):
                            model_name = "mingpt://" + model_name
                        else:
                            model_name = "huggingface://" + model_name
                            #assert self.QscalK == False, "Seule l’attention après softmax est disponible pour les modèles Huggingface"
                        #else:
                        #    model_name = "XXXX" # ERROR
                        attn.select_modele(model_name)
                    elif etat == 0:
                        if ligne.startswith(lbl_id):
                            ligne = ligne[len(lbl_id):]
                            idSNT = ligne.split()[0]
                            etat = 1
                    elif etat == 1:
                        if ligne.startswith('{"tokens": '):
                            jsn = json.loads(ligne)
                            attn.compute_attn_tensor(jsn["tokens"])
                            if self.QscalK:
                                data_attn = attn.data_QK.astype(np.float32)
                            else:
                                data_attn = attn.data_att.astype(np.float32)
                            nbtokens = len(jsn["tokens"])
                            Nadj = nbtokens*(nbtokens-1)//2
                            grfAdj = faire_graphe_adjoint(
                                len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
                                data_attn, self.liste_roles, outputARGn=True
                            )

                            sh1, dimension, deux = grfAdj["grfSig"].shape
                            assert deux == 2
                            #assert dim == dimension
                            assert sh1 == Nadj
                            assert (Nadj,) == grfAdj["roles"].shape
                            assert (Nadj,) == grfAdj["sens"].shape
                            assert (Nadj,) == grfAdj["msk_roles"].shape
                            assert (Nadj,) == grfAdj["msk_sens"].shape
                            assert (Nadj,) == grfAdj["msk_tkisoles"].shape
                            assert (Nadj,) == grfAdj["argus_num"].shape
                            assert (Nadj,) == grfAdj["msk_ARGn"].shape

                            msk_ARGn = grfAdj["msk_ARGn"] & (grfAdj["argus_num"] < 8)
                            # On estime qu’il n’existe pas d’arcs étiqueté par :ARG8 ou :ARGn (n>8)
                            argus_num = grfAdj["argus_num"] * msk_ARGn
                            
                            grfSig = torch.as_tensor(grfAdj["grfSig"]).to(dtype=torch.bfloat16)
                            # Conversion du tableau numpy en un tenseur de brainfloats
                            grfSig = grfSig.view(dtype=torch.int16).numpy()
                            # On "caste" point à point les brainfloats en entier à 16 bits, pour refaire un tableau numpy

                            
                            if nb_graphes == 0:
                                #Écrire la dimension des embeddings
                                octs = b"%d"%dimension + b"\n"
                                HH.update(octs)
                                FF.write(octs)

                                #Écrire le dtype du tableau edge_index
                                self.dtyp_edge_idx = grfAdj["edge_idx"].dtype
                                dtyp_edge_idx = repr(self.dtyp_edge_idx).encode("ascii")
                                octs = dtyp_edge_idx + b"\n"
                                HH.update(octs)
                                FF.write(octs)

                                #Écrire le dtype du tableau des roles
                                self.dtyp_roles = grfAdj["roles"].dtype
                                dtyp_roles = repr(self.dtyp_roles).encode("ascii")
                                octs = dtyp_roles + b"\n"
                                HH.update(octs)
                                FF.write(octs)
                                
                            offsets.append(FF.tell())    

                            octs = struct.pack("l", nbtokens)
                            HH.update(octs)
                            FF.write(octs)
                            #FF.write(bufgrfSig)
                             
                            octs = grfSig.reshape(-1).tobytes()
                            HH.update(octs)
                            FF.write(octs)

                            deux, sh = grfAdj["edge_idx"].shape
                            assert deux == 2
                            assert sh%2 == 0
                            assert sh == Nadj * (2*nbtokens-4)
                            sh = sh // 2
                            edge_idx = grfAdj["edge_idx"][:,:sh]
                            octs = struct.pack("l", sh)
                            HH.update(octs)
                            FF.write(octs)

                            octs = edge_idx.reshape(-1).tobytes()
                            HH.update(octs)
                            FF.write(octs)

                            octs = grfAdj["roles"].tobytes()
                            HH.update(octs)
                            FF.write(octs)

                            bools = np.zeros((Nadj,), dtype="uint8")
                            ones = np.ones((Nadj,), dtype="uint8")
                            bools = bools | ((ones * grfAdj["msk_sens"]))
                            bools = bools | ((ones * grfAdj["msk_roles"]) << 1)
                            bools = bools | ((ones * grfAdj["msk_tkisoles"]) << 2)
                            bools = bools | ((ones * msk_ARGn) << 3)
                            bools = bools | ((argus_num & 0x07) << 4)
                            bools = bools | ((ones * (grfAdj["sens"] == 1)) << 7)
                            
                            
                            
                            octs = bools.tobytes()
                            HH.update(octs)
                            FF.write(octs)

                            
                            nb_graphes += 1
                            pbar.update(1)
                            #if nb_graphes > 200:
                            #    break

                            etat  = 0
                    depart = False
        self.digests["gros_fichier"] = HH.hexdigest()
        offsets = np.array(offsets, dtype = np.int64)
        with open(self.processed_paths[1], "wb") as FF: # fichier "pointeurs.bin"
            np.save(FF, offsets)
        
        with open(self.processed_paths[1], "rb") as FF: # fichier "pointeurs.bin"
            HH = hashlib.file_digest(FF, "md5")
        self.digests["pointeurs"] = HH.hexdigest()
        self.offsets = offsets
        if self.QscalK:
            self.digests["attention_type"] = "QscalK"
        else:
            self.digests["attention_type"] = "softmax"
        with open(self.processed_paths[3], "w", encoding="UTF-8") as F: #fichier "digests.txt"
            json.dump(self.digests, F)
        pbar.close()

    def ouvrir_digests(self):
        if len(self.digests) == 0:
            with open(self.processed_paths[3], "r", encoding="UTF-8") as F: #fichier "digests.txt"
                self.digests = json.load(F)
            

    def ouvrir_offsets(self):
        if self.offsets is None:
            with open(self.processed_paths[1], "rb") as FF: # fichier "pointeurs.bin"
                self.offsets = np.load(FF)
            if self.debug_idSNT:
                self.compter_graphes()

    def lire_liste_roles(self):
        if self.liste_roles is None:
            with open(self.processed_paths[2], "r", encoding="UTF-8") as F: #fichier "liste_roles.json"
                jason = json.load(F)
            self.dico_roles = OrderedDict([tuple(t) for t in jason["dico_roles"]])
            self.dico_ARGn = OrderedDict([tuple(t) for t in jason["dico_ARGn"]])
            self.liste_roles = [k for k in self.dico_roles]
            self.liste_ARGn = [k for k in self.dico_ARGn]
            


    def ouvrir_gros_fichier(self):
        from numpy import dtype
        if self.FileHandle is None:
            self.FileHandle = open(self.processed_paths[0], "rb") # fichier "gros_fichier.bin"

            ligne = self.FileHandle.readline().decode("ascii").strip()
            self.dimension = int(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_edge_idx = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_roles = eval(ligne)

            self.sizeL = len(struct.pack("l", 0))
        

    def len(self):
        self.ouvrir_digests()
        self.ouvrir_offsets()
        self.lire_liste_roles()
        return self.offsets.shape[0]

    def get(self, idx):
        self.ouvrir_digests()
        self.ouvrir_offsets()
        self.lire_liste_roles()
        self.ouvrir_gros_fichier() 
        self.FileHandle.seek(self.offsets[idx])
        
        XXX = self.FileHandle.read(self.sizeL)
        (nbtokens,) = struct.unpack("l", XXX)
        Nadj = nbtokens * (nbtokens-1) // 2
        

        cnt = Nadj * self.dimension * 2
        grfSig = np.fromfile(self.FileHandle, dtype="int16", count=cnt).reshape((Nadj, self.dimension, 2))

        EEE = self.FileHandle.read(self.sizeL)
        (E1,) = struct.unpack("l", EEE)
        assert E1 == Nadj * (nbtokens -2)

        edge_idx = np.fromfile(self.FileHandle, dtype=self.dtyp_edge_idx, count=2*E1).reshape((2,E1))
        edge_idx = np.concatenate((edge_idx, edge_idx[(1,0),:]), axis=1)

        roles = np.fromfile(self.FileHandle, dtype=self.dtyp_roles, count=Nadj)

        bools = np.fromfile(self.FileHandle, dtype="uint8", count=Nadj)

        #grfSig = torch.frombuffer(grfSig, dtype=torch.bfloat16).reshape(Nadj, self.dimension, 2)
        grfSig = torch.as_tensor(grfSig).view(dtype=torch.bfloat16)
        edge_idx = torch.as_tensor(edge_idx)
        roles = torch.as_tensor(roles)
        sens = torch.as_tensor((bools & 128) > 0).to(dtype=torch.bfloat16)
        msk_iso = torch.as_tensor((bools & 4) > 0)
        msk_roles = torch.as_tensor((bools & 2) > 0)
        msk_sens = torch.as_tensor((bools & 1) > 0)
        msk_ARGn = torch.as_tensor((bools & 8) > 0)
        ARGn = torch.as_tensor((bools >> 4) & 0x07).view(dtype=torch.int8)

        data = Data(x=grfSig, edge_index=edge_idx,
                    y1=roles, y2=sens, ARGn=ARGn,
                    msk1=msk_roles, msk2 = msk_sens,
                    msk_iso = msk_iso,
                    msk_ARGn = msk_ARGn)
        return data
    
    def get_num_nodes(self):
        self.ouvrir_digests()
        self.ouvrir_offsets()
        self.lire_liste_roles()
        self.ouvrir_gros_fichier()
        resu = []
        for offset in self.offsets:  #tqdm(self.offsets, desc="num_nodes"):
            self.FileHandle.seek(offset)
        
            XXX = self.FileHandle.read(self.sizeL)
            (nbtokens,) = struct.unpack("l", XXX)
            Nadj = nbtokens * (nbtokens-1) // 2
            resu.append(Nadj)
        return resu



class BalancedGraphSampler(torch.utils.data.sampler.Sampler):
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
        shuffle=False
        ):
        
        if mode != "node":
            raise NotImplementedError
        self.dataset = dataset
        self.avg_num = avg_num
        self.mode = mode
        self.tolerance = tolerance
        self.shuffle = shuffle

        self.brasser()
        


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
        #itr = iter(indices)
        
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
            samples0, mineurs, eff_mineurs = BalancedGraphSampler.regroupement_aleatoire(
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
        

def essai_chrono():
    nom_fichier = "./AMR_et_graphes_phrases_explct.txt"
    etat = 0
    lbl_id = "# ::id "
    lbl_modname = "# ::model_name "

    total_graphes = 0
    with open(nom_fichier, "r", encoding="utf-8") as F:
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith(lbl_id):
                    total_graphes += 1
    
    pbar = tqdm(total = total_graphes)
    with open(nom_fichier, "r", encoding="utf-8") as F:
        for ligne in F:
            ligne = ligne.strip()
            if etat == 0:
                if ligne.startswith(lbl_id):
                    ligne = ligne[len(lbl_id):]
                    idSNT = ligne.split()[0]
                    etat = 1
            elif etat == 1:
                if ligne.startswith('{"tokens": '):
                    jsn = json.loads(ligne)
                    tokens = jsn["tokens"]
                    ntokens = len(tokens)
                    Nadj = ntokens*(ntokens-1)//2
                    degAdj = 2*ntokens - 4

                    # Construction de la matrice d’adjacence au format "edge_index" :
                    edge_idx = np.zeros((2, Nadj*degAdj), dtype=np.int32)
                    numerote = lambda s,c : (2*ntokens-1-s)*s//2+c-s-1
                    # Fonction pour obtenir le numéro d’un sommet adjoint (s,c) (avec s<c)

                    
                    with open("/dev/null", "wb") as FF:
                        octets = edge_idx.reshape(-1).tobytes()
                        FF.write(octets)
                    pbar.update(1)
                    etat = 0

def test_dataset():
    nom_fichier = "./AMR_et_graphes_phrases_explct.txt"
    lbl_id = "# ::id "
    lbl_modname = "# ::model_name "
    attn = TRANSFORMER_ATTENTION()
    liste_phrases = []
    total_graphes = 0
    with open(nom_fichier, "r", encoding="utf-8") as F:
        for ligne in F:
            ligne = ligne.strip()
            if ligne.startswith(lbl_id):
                total_graphes += 1
    pbar = tqdm(total = total_graphes)
    etat = 0
    with open(nom_fichier, "r", encoding="utf-8") as F:
        depart = True
        for ligne in F:
            ligne = ligne.strip()
            if depart and ligne.startswith(lbl_modname):
                model_name = ligne[len(lbl_modname):]
                attn.select_modele("minbert://"+ model_name)
            elif etat == 0:
                if ligne.startswith(lbl_id):
                    ligne = ligne[len(lbl_id):]
                    idSNT = ligne.split()[0]
                    etat = 1
            elif etat == 1:
                if ligne.startswith('{"tokens": '):
                    jsn = json.loads(ligne)
                    liste_phrases.append(jsn)
                    pbar.update(1)
                    etat = 0

    ds = AligDataset("./icidataset", "./AMR_et_graphes_phrases_explct.txt", QscalK=False)
    liste_roles = [k for k in ds.dico_roles]
    #idx = 0
    for NBESSAI in range(20):
        idx = random.randint(0, total_graphes-1)
        #idx += 1
        jsn = liste_phrases[idx]

        attn.compute_attn_tensor(jsn["tokens"])
        data_attn = attn.data_att.astype(np.float32)
        nbtokens = len(jsn["tokens"])
        print("Essai no %d (%d tokens)"%(NBESSAI, nbtokens))
        Nadj = nbtokens*(nbtokens-1)//2
        resu = faire_graphe_adjoint(
            len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
            data_attn, liste_roles
        )
        grfSig = resu["grfSig"]
        roles = resu["roles"]
        msk_roles = resu["msk_roles"]
        sens = resu["sens"]
        msk_sens = resu["msk_sens"]

        data = ds[idx]
        RESULTATS = []
        COMP = (data.y1 == roles)
        RESULTATS.append(all(x for x in COMP.reshape(-1).numpy().tolist()))
        COMP = (data.y2 == sens)
        RESULTATS.append(all(x for x in COMP.reshape(-1).numpy().tolist()))
        COMP = (data.msk1 == msk_roles)
        RESULTATS.append(all(x for x in COMP.reshape(-1).numpy().tolist()))
        COMP = (data.msk2 == msk_sens)
        RESULTATS.append(all(x for x in COMP.reshape(-1).numpy().tolist()))

        T = torch.as_tensor(grfSig)
        err = (T - data.x)
        err = err / T
        err.abs_()
        COMP = (err <= 2**(-7))
        COMP2 = T < 2**(-126)
        TEST = all(x for x in (COMP|COMP2).reshape(-1).numpy().tolist())
        if not TEST:
            maxi = max(err.reshape(-1).numpy().tolist())
            #for ind in range(Nadj):
            #    maxi = max(err[ind,:,:].reshape(-1).numpy().tolist())
            #    if maxi > 2**(-7):
            #        print(ind, maxi)
            #        pass
        RESULTATS.append(TEST)

        RESULTAT = all(RESULTATS)
        if RESULTAT:
            print("    Test positif.")
        else:
            print("    ÉCHEC TEST !!", RESULTATS, maxi)

        #print(RESU0, RESU1, RESU2, RESU3)



if __name__ == "__main__":
    #print(0)
    #essai_chrono()
    #test_dataset()
    #ds_train = AligDataset("./dataset_attn_train", "./AMR_et_graphes_phrases_explct_train.txt")
    #ds_dev   = AligDataset("./dataset_attn_dev", "./AMR_et_graphes_phrases_explct_dev.txt")
    #ds_test  = AligDataset("./dataset_attn_test", "./AMR_et_graphes_phrases_explct_test.txt")

    #ds_train = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    #ds_dev   = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct", QscalK=True, split="dev") #, debug_idSNT=True)
    #ds_test  = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct", QscalK=True, split="test")

    #ds_train = AligDataset("./deberta_att_train", "./AMR_grph_DebertaV2_xxlarge", QscalK=False, split="train", device="cuda")
    #ds_dev   = AligDataset("./deberta_att_dev", "./AMR_grph_DebertaV2_xxlarge", QscalK=False, split="dev", device="cuda") #, debug_idSNT=True)
    #ds_test  = AligDataset("./deberta_att_test", "./AMR_grph_DebertaV2_xxlarge", QscalK=False, split="test", device="cuda")

    #ds_train = AligDataset("./llama_QK_train", "./AMR_grph_LLAMA32", QscalK=True, split="train", device="cuda")
    #ds_dev   = AligDataset("./llama_QK_dev", "./AMR_grph_LLAMA32", QscalK=True, split="dev", device="cuda") #, debug_idSNT=True)
    #ds_test  = AligDataset("./llama_QK_test", "./AMR_grph_LLAMA32", QscalK=True, split="test", device="cuda")


    #ds_test2  = AligDataset("./dataset_QK_test2", "./AMR_et_graphes_phrases_explct", QscalK=True, split="test")

    if False:
        dsed0 = EdgeDataset(ds_dev, "./edges_QK_dev")
        
        filtre0 = ds_dev.filtre
        filtre1 = filtre0.eliminer(lambda x: x.al.startswith(":prep-"))
        filtre1 = filtre1.eliminer(":beneficiary")
        ds_dev1 = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct", QscalK=True, split="dev", transform=filtre1, debug_idSNT=True)
        data5 = ds_dev[5]
        data777 = ds_dev1[777]
        dsed1 = EdgeDataset(ds_dev1, "./edges_f_QK_dev")
        dsmono1 = EdgeDatasetMono(ds_dev1, "./edges_f_QK_dev")
        
        print(data5)

        #print(ds_train[2])
        #print(ds_train.raw_paths)
        #print(ds_dev.processed_paths)
        #print(len(ds_test))
    