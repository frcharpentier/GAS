import os
from collections import namedtuple, defaultdict, OrderedDict
import torch
from transformers import AutoTokenizer
from torch_geometric.data import Data as geoGraphData, Dataset as geoDataset
import hashlib
import json
import json_stream
import struct
from tqdm.auto import tqdm
import numpy as np

from tacred.make_dataset import(
    sourcer_tacred_alig,
    cataloguer_relations,
    liste_relations,
    #TacredDataset,
    count_json_stream,
    iter_json,
)
from tacred.graphe_adjoint import faire_graphe_adjoint


def transfo_to_filenames(transfo, QscalK=True):
    implemented = {"roberta": "minbert://roberta-base",
                   "robertaBase": "minbert://roberta-base",
                   "robertaLarge": "minbert://roberta-large",
                   "GPT2": "mingpt://gpt2",
                   "deberta": "huggingface://microsoft/deberta-v2-xxlarge",
                   "LLAMA32": "huggingface://meta-llama/Llama-3.2-3B",
                   "Llama3B": "huggingface://meta-llama/Llama-3.2-3B",
                   "Llama3Bi": "huggingface://meta-llama/Llama-3.2-3B-Instruct",
                   "Llama1B": "huggingface://meta-llama/Llama-3.2-1B",
                   "Llama1Bi": "huggingface://meta-llama/Llama-3.2-1B-Instruct",
                   "Llama8B": "huggingface://meta-llama/Llama-3.1-8B",
                   "Llama8Bi": "huggingface://meta-llama/Llama-3.1-8B-Instruct",
                   "mdrnBertBase" : "huggingface://answerdotai/ModernBERT-base",
                   "mdrnBertLarge" : "huggingface://answerdotai/ModernBERT-large",
                   "spring" : "huggingface://facebook/bart-large"}
    assert transfo in implemented
    label_QK = "QK" if QscalK else "att"
    rep_ds_grph = "./ds_graph_tacred_LLM_GNN_%s_%s_"%(transfo, label_QK)
    alig_file   = "./alig_tacred_%s"%(transfo)
    id_model = implemented[transfo]

    return alig_file, rep_ds_grph, id_model


class tacredTokens2GNNDataset(geoDataset):
    lbl_modname = "# ::model_name "
    #datatyp = namedtuple("datatyp", ("tokens", "ntoks", "edge_index", "msk", "y1", "y2"))

    def __init__(self, root, alignment_file,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split=False,
                 QscalK=False,
                 debug_idSNT=False,
                 device="cuda",
                 tokenHF = None,
                 ):
        if not split:
            self.split = False
        else:
            assert split in ["test", "dev", "train"]
            self.split = split
            if not alignment_file.endswith("_%s.json"%split):
                if alignment_file.endswith(".json"):
                    alignment_file = alignment_file[:-4]
                alignment_file += "_%s.json"%split
        self.alignment_file = alignment_file
        self.digests = dict()
        self.offsets = None
        self.FileHandle = None # FileHandle est le handle sur le fichier "gros_fichier.bin"
        self.pad_tkid = None   # id du token de padding

        self.liste_relations = None
        
        self.QscalK = QscalK
        self.debug_idSNT = debug_idSNT
        self.device = device
        self.tokenHF = tokenHF
        self.filtre = transform # Éventuellement remplacé plus tard si cette valeur est None.
        
        super().__init__(root, transform, pre_transform, pre_filter) # HACK pour appeler la méthode init de la classe grand-mère, et non la classe mère.
        #if self.transform is None:
        #    self.filtre = FusionElimination(nom_json=self.processed_paths[2])
        self.check_consistency()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        #return []
        return ['bigFile.bin', 'offsets.bin', 'relations_list.json', 'digests.txt']
    
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
            suffixe = "_%s.json"%self.split
            assert self.alignment_file.endswith(suffixe)
            fichier = self.alignment_file[:-len(suffixe)]
            fichier_train = fichier + "_train.json"
            fichier_dev   = fichier + "_dev.json" 
            fichier_test  = fichier + "_test.json"
            dico = cataloguer_relations(sourcer_tacred_alig(fichier_train))
            dico = cataloguer_relations(sourcer_tacred_alig(fichier_dev), dico)
            dico = cataloguer_relations(sourcer_tacred_alig(fichier_test), dico)
        else:
            dico = cataloguer_relations(sourcer_tacred_alig(self.alignment_file))
        cles = [k for k in dico]
        cles.sort(key = lambda x: x.lower())
        dico_relations = OrderedDict([(k, dico[k]) for k in cles])
        self.dico_relations = dico_relations
        
        self.liste_relations = [k for k in self.dico_relations]
        
        jason = dict()
        jason["dico_relations"] = [(k,v) for k,v in dico_relations.items()]
        
        with open(self.processed_paths[2], "w", encoding="UTF-8") as F: #fichier "relations_list.json"
            json.dump(jason, F)
        with open(self.processed_paths[2], "rb") as F: #fichier "relations_list.json"
            HH = hashlib.file_digest(F, "md5")
        self.digests["liste_relations"] = HH.hexdigest()

    def compter_graphes(self):
        with open(self.alignment_file, "r", encoding="utf-8") as F:
            NN = 0
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith("{") or ligne.startswith("[{"):
                    NN += 1
        if self.debug_idSNT:
            self.liste_idSNT = []
            with open(self.alignment_file, "r", encoding="utf-8") as F:
                ligne = F.readline().strip()
                assert ligne.startswith(tacredTokens2GNNDataset.lbl_modname)
                model_name = ligne[len(tacredTokens2GNNDataset.lbl_modname):]
                data = json_stream.load(F)
                try:
                    for N in tqdm(range(NN)):
                        jsn = json_stream.to_standard_types(data[N])
                        self.liste_idSNT.append(jsn["id"])
                except IndexError:
                    pass
        else:
            return NN
        
    def process(self):
        idx = 0
        offsets = []
        print("Entrée fonction process")
                
        print("Examining list of relations")
        self.ecrire_liste_roles()
        print("Done.")
        total_graphes = self.compter_graphes()

        with open(self.alignment_file, "rb") as F:
            digest = hashlib.file_digest(F, "md5")
        self.digests["fichier_source"] = digest.hexdigest()

        model_name=None
        
        HH = hashlib.new("md5")
        nb_graphes = 0
        iter0 = sourcer_tacred_alig(self.alignment_file, total_graphes)

        with open(self.processed_paths[0], "wb") as FF: # fichier "bigFile.bin"
            for jsn in iter0:
                tokens = jsn["tokens"]
                nbtokens = len(tokens)
                Nadj = nbtokens*(nbtokens-1)//2

                grfAdj = faire_graphe_adjoint(
                    len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
                    None, self.liste_relations
                )
                
                tokens = np.array(tokens)
                edge_idx = grfAdj["edge_idx"].astype(np.int32)

                if nb_graphes == 0:
                    #Écrire le dtype des tokens
                    model_name = jsn["model_name"]
                    assert "://" in model_name
                    _, model_name = model_name.split("://", maxsplit=1)
                    tknzr = AutoTokenizer.from_pretrained(model_name, token=self.tokenHF)
                    self.pad_tkid = tknzr.pad_token_id
                    self.digests["pad_tkid"] = self.pad_tkid

                    self.dtyp_tokens = tokens.dtype
                    dtyp_tokens = repr(self.dtyp_tokens).encode("ascii")
                    octs = dtyp_tokens + b"\n"
                    HH.update(octs)
                    FF.write(octs)

                    #Écrire le dtype du tableau edge_index
                    self.dtyp_edge_idx = edge_idx.dtype
                    dtyp_edge_idx = repr(self.dtyp_edge_idx).encode("ascii")
                    octs = dtyp_edge_idx + b"\n"
                    HH.update(octs)
                    FF.write(octs)

                    #Écrire le dtype du tableau de masques
                    self.dtyp_msk = grfAdj["utiles"].dtype
                    dtyp_msk = repr(self.dtyp_msk).encode("ascii")
                    octs = dtyp_msk + b"\n"
                    HH.update(octs)
                    FF.write(octs)

                    #Écrire le dtype du tableau de la relation
                    self.dtyp_relation = grfAdj["relation"].dtype
                    dtyp_relation = repr(self.dtyp_relation).encode("ascii")
                    octs = dtyp_relation + b"\n"
                    HH.update(octs)
                    FF.write(octs)

                    #Écrire le dtype du tableau de la direction
                    self.dtyp_dir = grfAdj["direction"].dtype
                    dtyp_dir = repr(self.dtyp_dir).encode("ascii")
                    octs = dtyp_dir + b"\n"
                    HH.update(octs)
                    FF.write(octs)
                    
                offsets.append(FF.tell())   

                #Écrire le nombre de tokens du graphe direct
                octs = struct.pack("l", nbtokens)
                HH.update(octs)
                FF.write(octs)
                                    
                #Écrire le tenseur des tokens
                octs = tokens.reshape(-1).tobytes()
                HH.update(octs)
                FF.write(octs)

                #Écrire la longueur du tableau de masques
                octs = struct.pack("l", grfAdj["utiles"].shape[0])
                HH.update(octs)
                FF.write(octs)

                #Écrire le signal masque
                octs = grfAdj["utiles"].reshape(-1).tobytes()
                HH.update(octs)
                FF.write(octs)

                deux, sh = edge_idx.shape
                assert deux == 2
                assert sh%2 == 0
                assert sh == Nadj * (2*nbtokens-4)
                I = edge_idx[0] < edge_idx[1]
                edg_id = edge_idx[:,I]
                deux, sh2 = edg_id.shape
                assert deux == 2
                assert 2*sh2 == sh
                
                #Écrire shape de edge_index
                octs = struct.pack("l", sh2)
                HH.update(octs)
                FF.write(octs)

                #Écrire edge_index
                octs = edg_id.reshape(-1).tobytes()
                HH.update(octs)
                FF.write(octs)

                #Écrire la relation 
                octs = grfAdj["relation"].tobytes()
                HH.update(octs)
                FF.write(octs)

                #Écrire la direction 
                octs = grfAdj["direction"].tobytes()
                HH.update(octs)
                FF.write(octs)
                
                nb_graphes += 1
                #if nb_graphes > 200:
                #    break
                

            
        self.digests["bigFile"] = HH.hexdigest()
        offsets = np.array(offsets, dtype = np.int64)
        with open(self.processed_paths[1], "wb") as FF: # fichier "pointeurs.bin"
            np.save(FF, offsets)
        
        with open(self.processed_paths[1], "rb") as FF: # fichier "pointeurs.bin"
            HH = hashlib.file_digest(FF, "md5")
        self.digests["offsets"] = HH.hexdigest()
        self.offsets = offsets
        if self.QscalK:
            self.digests["attention_type"] = "QscalK"
        else:
            self.digests["attention_type"] = "softmax"
        with open(self.processed_paths[3], "w", encoding="UTF-8") as F: #fichier "digests.txt"
            json.dump(self.digests, F)
        
    def ouvrir_digests(self):
        if len(self.digests) == 0:
            with open(self.processed_paths[3], "r", encoding="UTF-8") as F: #fichier "digests.txt"
                self.digests = json.load(F)
                self.pad_tkid = self.digests["pad_tkid"]

    def ouvrir_offsets(self):
        if self.offsets is None:
            with open(self.processed_paths[1], "rb") as FF: # fichier "offsets.bin"
                self.offsets = np.load(FF)
            if self.debug_idSNT:
                self.compter_graphes()

    def lire_liste_roles(self):
        if self.liste_relations is None:
            with open(self.processed_paths[2], "r", encoding="UTF-8") as F: #fichier "relations_list.json"
                jason = json.load(F)
            self.dico_relations = OrderedDict([tuple(t) for t in jason["dico_relations"]])
            self.liste_relations = [k for k in self.dico_relations]

    def ouvrir_gros_fichier(self):
        from numpy import dtype
        if self.FileHandle is None:
            self.FileHandle = open(self.processed_paths[0], "rb") # fichier "bigFile.bin"

            # Lire le dtype des tokens
            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_tokens = eval(ligne)

            # Lire le dtype du tableau edge_index
            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_edge_idx = eval(ligne)

            # Lire le dtype du tableau de masques
            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_msk = eval(ligne)

            # Lire le dtype du tableau de la relation
            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_relation = eval(ligne)

            # Lire le dtype du tableau de la direction
            ligne = self.FileHandle.readline().decode("ascii").strip()
            assert ligne.startswith("dtype(")
            self.dtyp_dir = eval(ligne)

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
        
        #Lire le nombre de tokens du graphe direct
        XXX = self.FileHandle.read(self.sizeL)
        (nbtokens,) = struct.unpack("l", XXX)
        Nadj = nbtokens * (nbtokens-1) // 2
        
        #Lire le tenseur des tokens
        tokens = np.fromfile(self.FileHandle, dtype=self.dtyp_tokens, count=nbtokens).reshape(-1)

        #Lire la longueur du tableau des masques
        XXX = self.FileHandle.read(self.sizeL)
        (n_msk,) = struct.unpack("l", XXX)

        #Lire le tableau des masques
        msk_tnsr = np.fromfile(self.FileHandle, dtype=self.dtyp_msk, count=n_msk).reshape(-1)

        #Lire format de edge_index
        EEE = self.FileHandle.read(self.sizeL)
        (E1,) = struct.unpack("l", EEE)
        assert E1 == Nadj * (nbtokens -2)

        #Lire edge_index
        edge_idx = np.fromfile(self.FileHandle, dtype=self.dtyp_edge_idx, count=2*E1).reshape((2,E1))
        edge_idx = np.concatenate((edge_idx, edge_idx[(1,0),:]), axis=1)

        #Lire la relation
        relation = np.fromfile(self.FileHandle, dtype=self.dtyp_relation, count=1).reshape(-1)

        #Lire la direction
        direction = np.fromfile(self.FileHandle, dtype=self.dtyp_dir, count=1).reshape(-1)
        

        #Conversion en tenseurs
        tokens = torch.as_tensor(tokens)
        edge_idx = torch.as_tensor(edge_idx)
        msk_tnsr = torch.as_tensor(msk_tnsr)
        relation = torch.as_tensor(relation)
        direction = torch.as_tensor(direction)
        
        if self.debug_idSNT:
            data = geoGraphData(num_nodes = Nadj,
                input_ids=tokens, ntoks = nbtokens,
                edge_index=edge_idx, msk=msk_tnsr,
                y1=relation, y2=direction,
                idSNT=self.liste_idSNT[idx])
        else:
            data = geoGraphData(num_nodes = Nadj,
                input_ids=tokens, ntoks = nbtokens,
                edge_index=edge_idx, msk=msk_tnsr,
                y1=relation,  y2=direction)
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