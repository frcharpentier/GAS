import os.path as osp
import numpy as np
import json
import struct

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
from graphe_adjoint import TRANSFORMER_ATTENTION, faire_graphe_adjoint


class AligDataset(Dataset):
    def __init__(self, root, nom_fichier, transform=None, pre_transform=None, pre_filter=None):
        self.nom_fichier = nom_fichier
        self.offsets = None
        self.FileHandle = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return []
        return ['gros_fichier.bin', 'pointeurs.bin']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    #path = download_url(url, self.raw_dir)
    #    print("On télécharge, mais on ne fait rien.")

    def process(self):
        idx = 0
        offsets = []
        print("Entrée fonction process")
        lbl_id = "# ::id "
        lbl_modname = "# ::model_name "
        etat = 0
        model_name=None
        nb_graphes = 0
        attn = TRANSFORMER_ATTENTION()
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
                            print(idSNT)
                            attn.compute_attn_tensor(jsn["tokens"])
                            idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens = faire_graphe_adjoint(
                                len(jsn["tokens"]), jsn["sommets"], jsn["aretes"], attn.data_att
                            )
                            if nb_graphes == 0:
                                #Écrire les dtypes des tableaux
                                self.dtyp_grfSig = grfSig.dtype
                                dtyp_grfSig = repr(grfSig.dtype).encode("ascii")
                                #print(dtyp_grfSig, file=FF)
                                FF.write(dtyp_grfSig + b"\n")

                                self.dtyp_edge_idx = edge_idx.dtype
                                dtyp_edge_idx = repr(edge_idx.dtype).encode("ascii")
                                #print(dtyp_edge_idx, file=FF)
                                FF.write(dtyp_edge_idx + b"\n")

                                #self.dtyp_roles = roles.dtype
                                #dtyp_roles = repr(roles.dtype).encode("ascii")
                                #print(dtyp_roles, file=FF)
                                #FF.write(dtyp_roles + b"\n")

                                self.dtyp_sens = sens.dtype
                                dtyp_sens = repr(sens.dtype).encode("ascii")
                                #print(dtyp_sens, file=FF)
                                FF.write(dtyp_sens + b"\n")

                                self.dtyp_msk_roles = msk_roles.dtype
                                #dtyp_msk_roles = repr(msk_roles.dtype).encode("ascii")
                                #print(dtyp_msk_roles, file=FF)
                                #FF.write(dtyp_msk_roles + b"\n")

                                self.dtyp_msk_sens = msk_sens.dtype
                                dtyp_msk_sens = repr(msk_sens.dtype).encode("ascii")
                                #print(dtyp_msk_sens, file=FF)
                                FF.write(dtyp_msk_sens + b"\n")
                                
                            offsets.append(FF.tell())    

                            FF.write(struct.pack("lll", *grfSig.shape))
                            octets = grfSig.reshape(-1).tobytes()
                            FF.write(octets)

                            FF.write(struct.pack("ll", *edge_idx.shape))
                            octets = edge_idx.reshape(-1).tobytes()
                            FF.write(octets)

                            #FF.write(struct.pack("l", *sens.shape))
                            FF.write(sens.tobytes())

                            #FF.write(struct.pack("l", *msk_sens.shape))
                            FF.write(msk_sens.tobytes())
                            
                            nb_graphes += 1
                            if nb_graphes > 200:
                                break

                            etat  = 0
                    depart = False
        offsets = np.array(offsets, dtype = np.int64)
        with open(self.processed_paths[1], "wb") as FF:
            np.save(FF, offsets)
        self.offsets = offsets

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

            #ligne = self.FileHandle.readline().decode("ascii").strip()
            #assert ligne.startswith(b"dtype(")
            #self.dtyp_roles = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_sens = eval(ligne)

            #ligne = self.FileHandle.readline().decode("ascii").strip()
            #assert ligne.startswith(b"dtype(")
            #self.dtyp_msk_roles = eval(ligne)

            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_msk_sens = eval(ligne)

            self.sizeL = len(struct.pack("l", 0))
        

    def len(self):
        #return len(self.processed_file_names)
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

        #SENS = self.FileHandle.read(self.sizeL)
        #(S1,) = struct.unpack("l", SENS)

        #MASK = self.FileHandle.read(self.sizeL)
        #(M1,) = struct.unpack("l", MASK)

        

        

        buf = self.FileHandle.read(X1*(self.dtyp_sens.itemsize))
        sens = np.frombuffer(buf, dtype=self.dtyp_sens)

        buf = self.FileHandle.read(X1*(self.dtyp_msk_sens.itemsize))
        msk_sens = np.frombuffer(buf, dtype=self.dtyp_msk_sens)

        grfSig = torch.as_tensor(grfSig)
        edge_idx = torch.as_tensor(edge_idx)
        sens = torch.as_tensor(sens)
        msk_sens = torch.as_tensor(msk_sens)

        data = Data(x=grfSig, edge_index=edge_idx, y=sens, msk=msk_sens)
        return data


if __name__ == "__main__":
    ds = AligDataset("./icidataset", "./AMR_et_graphes_phrases_explct.txt")

    print(ds[2])
    print(ds.raw_paths)
    print(ds.processed_paths)
    print(len(ds))
    