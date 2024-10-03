import os.path as osp
import numpy as np
import json
import struct

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
import torch_geometric.transforms as TRF
from graphe_adjoint import TRANSFORMER_ATTENTION, faire_graphe_adjoint
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from inspect import isfunction
from liste_tous_roles import dico_roles
import random




class FusionElimination(TRF.BaseTransform):
    def __init__(self, index=None, noms_classes=None, effectifs=None, alias=None):
        if index is None:
            index=[i for i, _ in enumerate(dico_roles)]
            if noms_classes is None:
                noms_classes = [[k] for k in dico_roles]
            if effectifs is None:
                effectifs = [v[0] for _,v in dico_roles.items()]
            if alias is None:
                alias = [k for k in dico_roles]

        assert type(index) in (list, tuple)
        self.nb_classes = 1+max(index)
        assert all(idx in index for idx in range(self.nb_classes))

        self.index = index
        if any(idx < 0 for idx in index):
            self.eliminer = True
            self.agarder = [idx >=0 for idx in index]
        else:
            self.agarder = None
            self.eliminer = False

        
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
        if isfunction(args[0]):
            f = args[0]
            for i, idx in enumerate(self.index):
                if idx == -1:
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                dico = {"no": i, "al": al, "ef":ef, "li":li}
                if not f(dico):
                    #garder
                    index.append(N)
                    alias.append(al)
                    noms_classes.append(li)
                    effectifs.append(ef)
                    N += 1
                else:
                    #Éliminer
                    index.append(-1)
                    
        else:
            for i, idx in enumerate(self.index):
                if idx == -1:
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if not al in args:
                    #garder
                    index.append(N)
                    alias.append(al)
                    noms_classes.append(li)
                    effectifs.append(ef)
                    N += 1
                else:
                    #Éliminer
                    index.append(-1)
                    
        #return FusionElimination(self.noms_classes, index, self.effectifs, self.alias)
        return FusionElimination(index, noms_classes, effectifs, alias)


    def garder(self, *args):
        N = 0
        index = []
        alias = []
        noms_classes = []
        effectifs = []
        if isfunction(args[0]):
            f = args[0]
            for i, idx in enumerate(self.index):
                if idx < 0:
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if f({"no": i, "al": al, "ef":ef, "li":li}):
                    #garder
                    index.append(N)
                    alias.append(al)
                    noms_classes.append(li)
                    effectifs.append(ef)
                    N += 1
                else:
                    #Éliminer
                    index.append(-1)
                    
        else:
            for i, idx in enumerate(self.index):
                if idx < 0:
                    index.append(-1)
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                if al in args:
                    #garder
                    index.append(N)
                    alias.append(al)
                    noms_classes.append(li)
                    effectifs.append(ef)
                    N += 1
                else:
                    #Éliminer
                    index.append(-1)
                    
        #return FusionElimination(self.noms_classes, index, self.effectifs, self.alias)
        return FusionElimination(index, noms_classes, effectifs, alias)
                    

    def fusionner(self, *args):
        if isfunction(args[0]):
            f = args[0]
            dico0 = {}
            for i, idx in enumerate(self.index):
                if idx < 0:
                    continue
                al = self.alias[idx]
                ef = self.effectifs[idx]
                li = self.noms_classes[idx]
                truc = f({"no": i, "al": al, "ef":ef, "li":li})
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
        for i, idx in enumerate(self.index):
            if idx == -1:
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
        #resu = FusionElimination(self.noms_classes, index, self.effectifs)
        #resu.alias = [T[-1] for T in listeargus]
        return FusionElimination(index, noms_classes, effectifs, alias)  

    def forward(self, data):
        data.y1 = self.index[data.y1]
        if self.eliminer:
            data.msk_y1 = data.msk_y1 & self.agarder[data.y1]
        return data





class AligDataset(Dataset):
    def __init__(self, root, nom_fichier, transform=None, pre_transform=None, pre_filter=None):
        self.nom_fichier = nom_fichier
        self.offsets = None
        self.FileHandle = None
        self.liste_roles = [k for k in dico_roles]
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return []
        return ['gros_fichier.bin', 'pointeurs.bin']

    def process(self):
        idx = 0
        offsets = []
        print("Entrée fonction process")
        lbl_id = "# ::id "
        lbl_modname = "# ::model_name "
        total_graphes = 0
        with open(self.nom_fichier, "r", encoding="utf-8") as F:
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith(lbl_id):
                    total_graphes += 1
        etat = 0
        model_name=None
        nb_graphes = 0
        attn = TRANSFORMER_ATTENTION()
        pbar = tqdm(total = total_graphes)

        # Test du format des floats
        T = torch.ones((1,), dtype=torch.bfloat16)*1000/3
        buf = T.to(dtype=torch.float32).numpy().tobytes()[2:4]
        assert buf == bytes([0xa7, 0x43])

        with open(self.processed_paths[0], "wb") as FF:
            with open(self.nom_fichier, "r", encoding="utf-8") as F:
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
                            attn.compute_attn_tensor(jsn["tokens"])
                            data_attn = attn.data_att.astype(np.float32)
                            nbtokens = len(jsn["tokens"])
                            Nadj = nbtokens*(nbtokens-1)//2
                            idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens, msk_iso = faire_graphe_adjoint(
                                len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
                                data_attn, self.liste_roles
                            )
                            sh1, dimension, deux = grfSig.shape
                            assert deux == 2
                            #assert dim == dimension
                            assert sh1 == Nadj
                            assert (Nadj,) == roles.shape
                            assert (Nadj,) == sens.shape
                            assert (Nadj,) == msk_roles.shape
                            assert (Nadj,) == msk_sens.shape
                            assert (Nadj,) == msk_iso.shape
                            
                            grfSig = torch.as_tensor(grfSig).to(dtype=torch.bfloat16)
                            # Conversion du tableau numpy en un tenseur de brainfloats
                            grfSig = grfSig.view(dtype=torch.int16).numpy()
                            # On "caste" point à point les brainfloats en entier à 16 bits, pour refaire un tableau numpy

                            
                            if nb_graphes == 0:
                                #Écrire la dimension des embeddings
                                FF.write(b"%d"%dimension + b"\n")

                                #Écrire le dtype du tableau edge_index
                                self.dtyp_edge_idx = edge_idx.dtype
                                dtyp_edge_idx = repr(edge_idx.dtype).encode("ascii")
                                FF.write(dtyp_edge_idx + b"\n")

                                #Écrire le dtype du tableau des roles
                                self.dtyp_roles = roles.dtype
                                dtyp_roles = repr(roles.dtype).encode("ascii")
                                FF.write(dtyp_roles + b"\n")
                                
                            offsets.append(FF.tell())    

                            FF.write(struct.pack("l", nbtokens))
                            #FF.write(bufgrfSig)
                             
                            FF.write(grfSig.reshape(-1).tobytes())

                            deux, sh = edge_idx.shape
                            assert deux == 2
                            assert sh%2 == 0
                            assert sh == Nadj * (2*nbtokens-4)
                            sh = sh // 2
                            edge_idx = edge_idx[:,:sh]
                            FF.write(struct.pack("l", sh))

                            FF.write(edge_idx.reshape(-1).tobytes())

                            FF.write(roles.tobytes())

                            bools = np.zeros((Nadj,), dtype="uint8")
                            ones = np.ones((Nadj,), dtype="uint8")
                            bools = bools | ((ones * (sens == 1)) << 7)
                            bools = bools | ((ones * msk_iso) << 2)
                            bools = bools | ((ones * msk_roles) << 1)
                            bools = bools | ((ones * msk_sens))

                            FF.write(bools.tobytes())

                            
                            nb_graphes += 1
                            pbar.update(1)
                            #if nb_graphes > 200:
                            #    break

                            etat  = 0
                    depart = False
        offsets = np.array(offsets, dtype = np.int64)
        with open(self.processed_paths[1], "wb") as FF:
            np.save(FF, offsets)
        self.offsets = offsets
        pbar.close()

    def ouvrir_offsets(self):
        if self.offsets is None:
            with open(self.processed_paths[1], "rb") as FF:
                self.offsets = np.load(FF)


    def ouvrir_gros_fichier(self):
        from numpy import dtype
        if self.FileHandle is None:
            self.FileHandle = open(self.processed_paths[0], "rb")

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
        self.ouvrir_offsets()
        return self.offsets.shape[0]

    def get(self, idx):
        self.ouvrir_offsets()
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
        msk_roles = torch.as_tensor((bools & 2) > 0)
        msk_sens = torch.as_tensor((bools & 1) > 0)

        data = Data(x=grfSig, edge_index=edge_idx,
                    y1=roles, y2=sens,
                    msk1=msk_roles, msk2 = msk_sens)
        return data


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

    liste_roles = [k for k in dico_roles]
    ds = AligDataset("./icidataset", "./AMR_et_graphes_phrases_explct.txt")
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
        idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens = faire_graphe_adjoint(
            len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
            data_attn, liste_roles
        )
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
    # filtre = FusionElimination()
    # filtre2 = filtre.garder(lambda x: x["ef"] > 1000)

    # def clef(x):
    #     if x["al"].startswith(":>"):
    #         return ":" + x["al"][2:].lower()
    #     else:
    #         return x["al"].lower()
        
    # filtre3 = filtre2.fusionner(clef)

    # print(0)
    #essai_chrono()
    #test_dataset()
    ds = AligDataset("./icidataset", "./AMR_et_graphes_phrases_explct.txt")

    print(ds[2])
    print(ds.raw_paths)
    print(ds.processed_paths)
    print(len(ds))
    