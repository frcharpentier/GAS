import os.path as osp
import numpy as np
import json
import struct

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
from graphe_adjoint import TRANSFORMER_ATTENTION, faire_graphe_adjoint
from collections import OrderedDict, defaultdict
from enchainables import MAILLON
from tqdm import tqdm


@MAILLON
def sourcer_fichier_txt(SOURCE, nom_fichier):
    lbl_modname = "# ::model_name "
    lbl_id = "# ::id "
    etat = 0
    model_name=None
    with open(nom_fichier, "r", encoding="utf-8") as F:
        depart = True
        for ligne in F:
            ligne = ligne.strip()
            if depart and ligne.startswith(lbl_modname):
                model_name = ligne[len(lbl_modname):]
            elif etat == 0:
                if ligne.startswith(lbl_id):
                    ligne = ligne[len(lbl_id):]
                    idSNT = ligne.split()[0]
                    etat = 1
            elif etat == 1:
                if ligne.startswith('{"tokens": '):
                    jsn = json.loads(ligne)
                    jsn["idSNT"] = idSNT
                    jsn["model_name"] = model_name
                    yield jsn
                    etat  = 0
            depart = False


def cataloguer_roles(source):
    dico = defaultdict(lambda: 0)
    for jsn in source:
        for s, r, c in jsn["aretes"]:
            if not r.startswith("?"):
                dico[r] += 1
    return dico

def liste_roles(nom_fichier):
    chaine = sourcer_fichier_txt(nom_fichier)
    dico = cataloguer_roles(chaine)
    cles = [k for k in dico]
    classer = lambda x: x[2:].lower() if x.startswith(":>") else (x[1:].lower() if x.startswith(":") else x.lower())
    cles.sort(key = classer)
    for k in cles:
        print("%s: (%d,)"%(repr(k), dico[k]))
    #print(dict(dico))

#Le dico suivant a été fait en exécutant la fonction liste_roles
dico_roles = OrderedDict( [ (':accompanier', (569,0)),
    (':age', (1135,1)),
    (':>AGENT', (125945,2)),
    (':>ARG0', (2,None)),
    (':>ARG2', (3,None)),
    (':>ASSET', (1283,3)),
    (':>ATTRIBUTE', (14329,4)),
    (':beneficiary', (1371,5)),
    (':>BENEFICIARY', (7998,5)),
    (':>CAUSE', (1573,6)),
    (':>CO-AGENT', (2030,7)),
    (':>CO-PATIENT', (513,8)),
    (':>CO-THEME', (9280,9)),
    (':>CONCESSION', (1140,10)),
    (':concession', (1713,10)),
    (':condition', (3670,11)),
    (':>CONDITION', (1562,11)),
    (':conj-as-if', (24,12)),
    (':consist-of', (1386,13)),
    (':day', (1,None)),
    (':dayperiod', (38,None)),
    (':decade', (1,None)),
    (':degree', (5623,14)),
    (':>DESTINATION', (6178,15)),
    (':destination', (575,15)),
    (':direction', (1188,16)),
    (':duration', (2922,17)),
    (':example', (2914,18)),
    (':>EXPERIENCER', (15572,19)),
    (':extent', (455,20)),
    (':>EXTENT', (1877,20)),
    (':frequency', (1380,21)),
    (':>FREQUENCY', (23,21)),
    (':>GOAL', (7218,22)),
    (':>IDIOM', (744,23)),
    (':>INSTRUMENT', (2714,24)),
    (':instrument', (787,24)),
    (':li', (852,None)),
    (':location', (21913,25)),
    (':>LOCATION', (5104,25)),
    (':manner', (8095,26)),
    (':>MANNER', (101,26)),
    (':>MATERIAL', (322,27)),
    (':medium', (2055,28)),
    (':mod', (121338,29)),
    (':>MOD', (413,29)),
    (':mode', (1455,30)),
    (':month', (34,31)),
    (':name', (4569,32)),
    (':op1', (18,None)),
    (':ord', (1254,33)),
    (':part', (6322,34)),
    (':>PART', (141,34)),
    (':path', (401,35)),
    (':>PATIENT', (29132,36)),
    (':polarity', (18501,None)),
    (':>POLARITY', (96,None)),
    (':polite', (203,None)),
    (':poss', (16093,37)),
    (':prep-against', (152,None)),
    (':prep-along-with', (9,None)),
    (':prep-amid', (5,None)),
    (':prep-among', (29,None)),
    (':prep-as', (178,None)),
    (':prep-at', (8,None)),
    (':prep-by', (21,None)),
    (':prep-for', (114,None)),
    (':prep-from', (31,None)),
    (':prep-in', (224,None)),
    (':prep-in-addition-to', (7,None)),
    (':prep-into', (24,None)),
    (':prep-on', (137,None)),
    (':prep-on-behalf-of', (65,None)),
    (':prep-out-of', (1,None)),
    (':prep-to', (140,None)),
    (':prep-toward', (8,None)),
    (':prep-under', (138,None)),
    (':prep-with', (295,None)),
    (':prep-without', (41,None)),
    (':>PRODUCT', (2754,38)),
    (':purpose', (5417,39)),
    (':>PURPOSE', (487,39)),
    (':quant', (13804,40)),
    (':range', (54,41)),
    (':>RECIPIENT', (9893,42)),
    (':>RESULT', (14792,42)),
    (':scale', (101,43)),
    (':>SOURCE', (5986,44)),
    (':source', (2604,44)),
    (':>STIMULUS', (11294,45)),
    (':subevent', (350,46)),
    (':subset', (3,47)),
    (':>THEME', (121792,48)),
    (':time', (30960,49)),
    (':>TIME', (419,49)),
    (':timezone', (326,None)),
    (':topic', (5436,50)),
    (':>TOPIC', (14119,50)),
    (':unit', (107,None)),
    (':>VALUE', (2168,51)),
    (':value', (340,51)),
    (':weekday', (26,None)),
    (':year', (9,None)),
    ('{and_or}', (2358,52)),
    ('{and}', (140798,53)),
    ('{groupe}', (1117630,54)),
    ('{idem}', (41628,55)),
    ('{inter}', (1214,56)),
    ('{or}', (11339,57)),
    ('{syntax}', (10258,58))  ]
)

def traiter_dico_roles():
    dico = defaultdict(lambda :([], 0))
    for k, (n, id) in dico_roles.items():
        lis, cumul = dico[id]
        lis.append(k)
        cumul += n
        dico[id] = (lis, cumul)
    return dico

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
                            #jsn["idSNT"] = idSNT
                            #jsn["model_name"] = model_name
                            #print(idSNT)
                            attn.compute_attn_tensor(jsn["tokens"])
                            idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens = faire_graphe_adjoint(
                                len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
                                attn.data_att, self.liste_roles
                            )
                            if nb_graphes == 0:
                                #Écrire les dtypes des tableaux
                                self.dtyp_grfSig = grfSig.dtype
                                dtyp_grfSig = repr(grfSig.dtype).encode("ascii")
                                FF.write(dtyp_grfSig + b"\n")

                                self.dtyp_edge_idx = edge_idx.dtype
                                dtyp_edge_idx = repr(edge_idx.dtype).encode("ascii")
                                FF.write(dtyp_edge_idx + b"\n")

                                self.dtyp_roles = roles.dtype
                                dtyp_roles = repr(roles.dtype).encode("ascii")
                                FF.write(dtyp_roles + b"\n")

                                self.dtyp_sens = sens.dtype
                                dtyp_sens = repr(sens.dtype).encode("ascii")
                                FF.write(dtyp_sens + b"\n")

                                self.dtyp_msk_roles = msk_roles.dtype
                                dtyp_msk_roles = repr(msk_roles.dtype).encode("ascii")
                                FF.write(dtyp_msk_roles + b"\n")

                                self.dtyp_msk_sens = msk_sens.dtype
                                dtyp_msk_sens = repr(msk_sens.dtype).encode("ascii")
                                FF.write(dtyp_msk_sens + b"\n")
                                
                            offsets.append(FF.tell())    

                            FF.write(struct.pack("lll", *grfSig.shape))
                            octets = grfSig.reshape(-1).tobytes()
                            FF.write(octets)

                            FF.write(struct.pack("ll", *edge_idx.shape))
                            octets = edge_idx.reshape(-1).tobytes()
                            FF.write(octets)

                            FF.write(roles.tobytes())
                            FF.write(sens.tobytes())
                            FF.write(msk_roles.tobytes())
                            FF.write(msk_sens.tobytes())
                            
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
            assert ligne.startswith("dtype(")
            self.dtyp_grfSig = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_edge_idx = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_roles = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_sens = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_msk_roles = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_msk_sens = eval(ligne)

            self.sizeL = len(struct.pack("l", 0))
        

    def len(self):
        self.ouvrir_offsets()
        return self.offsets.shape[0]

    def get(self, idx):
        self.ouvrir_offsets()
        self.ouvrir_gros_fichier() 
        self.FileHandle.seek(self.offsets[idx])
        
        X1X2X3 = self.FileHandle.read(3*self.sizeL)
        X1, X2, X3 = struct.unpack("lll", X1X2X3)

        buf = self.FileHandle.read(X1*X2*X3*(self.dtyp_grfSig.itemsize))
        grfSig = np.frombuffer(buf, dtype=self.dtyp_grfSig).reshape((X1,X2,X3))

        E1E2 = self.FileHandle.read(2*self.sizeL)
        E1, E2 = struct.unpack("ll", E1E2)

        buf = self.FileHandle.read(E1*E2*(self.dtyp_edge_idx.itemsize))
        edge_idx = np.frombuffer(buf, dtype=self.dtyp_edge_idx).reshape((E1, E2))

        buf = self.FileHandle.read(X1*(self.dtyp_roles.itemsize))
        roles = np.frombuffer(buf, dtype=self.dtyp_roles)

        buf = self.FileHandle.read(X1*(self.dtyp_sens.itemsize))
        sens = np.frombuffer(buf, dtype=self.dtyp_sens)

        buf = self.FileHandle.read(X1*(self.dtyp_msk_roles.itemsize))
        msk_roles = np.frombuffer(buf, dtype=self.dtyp_msk_roles)

        buf = self.FileHandle.read(X1*(self.dtyp_msk_sens.itemsize))
        msk_sens = np.frombuffer(buf, dtype=self.dtyp_msk_sens)

        grfSig = torch.as_tensor(grfSig)
        edge_idx = torch.as_tensor(edge_idx)
        roles = torch.as_tensor(roles)
        sens = torch.as_tensor(sens)
        msk_roles = torch.as_tensor(msk_roles)
        msk_sens = torch.as_tensor(msk_sens)

        data = Data(x=grfSig, edge_index=edge_idx,
                    y1=roles, y2=sens,
                    msk1=msk_roles, msk2 = msk_sens)
        return data


if __name__ == "__main__":
    ds = AligDataset("./icidataset", "./AMR_et_graphes_phrases_explct.txt")

    print(ds[2])
    print(ds.raw_paths)
    print(ds.processed_paths)
    print(len(ds))
    