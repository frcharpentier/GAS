import os
import re
from alig.algebre_relationnelle import RELATION
import json_stream
import json
import struct
from tqdm.auto import tqdm
import torch
from torch_geometric.data import Data as geoGraphData, Dataset as geoDataset
import numpy as np
import itertools
from tacred.fichier_zero import rebuild_sentence
from alig.outils_alignement import ALIGNEUR
from collections import defaultdict, OrderedDict
from DtSets.make_dataset import AligDataset
from DtSets.graphe_adjoint import TRANSFORMER_ATTENTION
from tacred.graphe_adjoint import faire_graphe_adjoint
import hashlib



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
    rep_ds_grph = "./ds_graph_tacred_%s_%s_"%(transfo, label_QK)
    alig_file   = "./alig_tacred_%s"%(transfo)
    id_model = implemented[transfo]

    return alig_file, rep_ds_grph, id_model


def sourcer_tacred_alig(nom_fichier, NN=None):
    lbl_modname = "# ::model_name "
    etat = 0
    model_name=None
    if NN is None:
        with open(nom_fichier, "r", encoding="utf-8") as F:
            NN = 0
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith("{") or ligne.startswith("[{"):
                    NN += 1
    with open(nom_fichier, "r", encoding="utf-8") as F:
        ligne = F.readline().strip()
        assert ligne.startswith(lbl_modname)
        model_name = ligne[len(lbl_modname):]
        data = json_stream.load(F)
        try:
            for N in tqdm(range(NN)):
                jsn = json_stream.to_standard_types(data[N])
                jsn["model_name"] = model_name
                yield jsn
        except IndexError:
            pass

def cataloguer_relations(source, dico=None):
    if dico is None:
        dico = defaultdict(lambda: 0)
    assert isinstance(dico, defaultdict)
    for jsn in source:
        #for s, r, c in jsn["aretes"]:
        #    dico[r] += 1
        s, r, c = jsn["aretes"][0]
        dico[r] += 1
    return dico

def liste_relations(dico):
    assert isinstance(dico, dict)
    cles = [k for k in dico]
    classer = lambda x: x.lower()
    cles.sort(key = classer)
    dico_relations = OrderedDict([(k, dico[k]) for k in cles])
    
    return dico_relations


class TacredDataset(geoDataset):
    def __init__(self, root, alignment_file,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split=False,
                 QscalK=False,
                 debug_idSNT=False,
                 device="cuda",
                 tokenHF = None):
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
        self.FileHandle = None

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
        return ['gros_fichier.bin', 'pointeurs.bin', 'liste_relations.json', 'digests.txt']
    
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
        
        with open(self.processed_paths[2], "w", encoding="UTF-8") as F: #fichier "liste_relations.json"
            json.dump(jason, F)
        with open(self.processed_paths[2], "rb") as F: #fichier "liste_relations.json"
            HH = hashlib.file_digest(F, "md5")
        self.digests["liste_relations"] = HH.hexdigest()

    def compter_graphes(self):
        lbl_modname = "# ::model_name "
        if self.debug_idSNT:
            self.liste_idSNT = []
    
        with open(self.alignment_file, "r", encoding="utf-8") as F:
            total_graphes = 0
            for ligne in F:
                ligne = ligne.strip()
                if ligne.startswith(lbl_modname):
                    model_name = ligne[len(lbl_modname):]
                elif ligne.startswith("{") or ligne.startswith("[{"):
                    total_graphes += 1

        return total_graphes
    
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
        
        attn = TRANSFORMER_ATTENTION(QscalK=self.QscalK, device=self.device)
        
        # Test du format des floats
        T = torch.ones((1,), dtype=torch.bfloat16)*1000/3
        buf = T.to(dtype=torch.float32).numpy().tobytes()[2:4]
        assert buf == bytes([0xa7, 0x43])

        HH = hashlib.new("md5")
        nb_graphes = 0
        iter0 = sourcer_tacred_alig(self.alignment_file, total_graphes)

        with open(self.processed_paths[0], "wb") as FF: # fichier "gros_fichier.bin"
            depart = True
            for jsn in iter0:
                if depart:
                    model_name = jsn["model_name"]
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
                    attn.select_modele(model_name, tokenHF = self.tokenHF)
                    depart = False

                attn.compute_attn_tensor(jsn["tokens"])
                if self.QscalK:
                    data_attn = attn.data_QK.astype(np.float32)
                else:
                    data_attn = attn.data_att.astype(np.float32)
                nbtokens = len(jsn["tokens"])
                Nadj = nbtokens*(nbtokens-1)//2
                grfAdj = faire_graphe_adjoint(
                    len(jsn["tokens"]), jsn["sommets"], jsn["aretes"],
                    data_attn, self.liste_relations
                )
                
                sh1, dimension, deux = grfAdj["G"].x.shape
                assert deux == 2
                #assert dim == dimension
                assert sh1 == Nadj
            
                grfSig = grfAdj["G"].x.to(dtype=torch.bfloat16)
                # Conversion du tableau numpy en un tenseur de brainfloats
                grfSig = grfSig.view(dtype=torch.int16).numpy()
                # On "caste" point à point les brainfloats en entier à 16 bits, pour refaire un tableau numpy

                edge_idx = grfAdj["G"].edge_index.numpy().astype(np.int32)

                if nb_graphes == 0:
                    #Écrire la dimension des embeddings
                    octs = b"%d"%dimension + b"\n"
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
                #FF.write(bufgrfSig)
                    
                #Écrire le signal graphe
                octs = grfSig.reshape(-1).tobytes()
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
        if self.liste_relations is None:
            with open(self.processed_paths[2], "r", encoding="UTF-8") as F: #fichier "liste_roles.json"
                jason = json.load(F)
            self.dico_relations = OrderedDict([tuple(t) for t in jason["dico_relations"]])
            self.liste_relations = [k for k in self.dico_relations]
            
    def ouvrir_gros_fichier(self):
        from numpy import dtype
        if self.FileHandle is None:
            self.FileHandle = open(self.processed_paths[0], "rb") # fichier "gros_fichier.bin"
            # Lire la dimension des embeddings
            ligne = self.FileHandle.readline().decode("ascii").strip()
            self.dimension = int(ligne)
            
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
        
        #Lire le signal graphe
        cnt = Nadj * self.dimension * 2
        grfSig = np.fromfile(self.FileHandle, dtype="int16", count=cnt).reshape((Nadj, self.dimension, 2))

        #Lire la longueur du tableau des masques
        XXX = self.FileHandle.read(self.sizeL)
        (n_msk,) = struct.unpack("l", XXX)

        #print("idx=%d, n_msk=%d"%(idx, n_msk))

        #Lire le tableau des masques
        utiles = np.fromfile(self.FileHandle, dtype=self.dtyp_msk, count=n_msk).reshape(-1)

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
        grfSig = torch.as_tensor(grfSig).view(dtype=torch.bfloat16)
        edge_idx = torch.as_tensor(edge_idx)
        relation = torch.as_tensor(relation)
        direction = torch.as_tensor(direction)
        

        data = geoGraphData(x=grfSig, edge_index=edge_idx,
                    msk=torch.tensor(utiles), y1=relation, y2=direction, )
        return data
    

    def get_strates(self):
        self.ouvrir_digests()
        self.ouvrir_offsets()
        self.lire_liste_roles()
        self.ouvrir_gros_fichier()
        tab = np.zeros((len(self.offsets),), dtype=self.dtyp_relation)
        cumul = np.zeros((1+len(self.liste_relations),), dtype=np.dtype("int32"))
        print("Building stratified table")
        for ii, offset in enumerate(tqdm(self.offsets)):
            self.FileHandle.seek(offset, 0) #From start of file

            # skip graph signal
            XXX = self.FileHandle.read(self.sizeL)
            (nbtokens,) = struct.unpack("l", XXX)
            Nadj = nbtokens * (nbtokens-1) // 2
            cnt = Nadj * self.dimension * 2
            self.FileHandle.seek(cnt * np.dtype("int16").itemsize, 1) # From current position

            # skip msk
            XXX = self.FileHandle.read(self.sizeL)
            (n_msk,) = struct.unpack("l", XXX)
            self.FileHandle.seek(n_msk * self.dtyp_msk.itemsize, 1) # From current position

            # skip edge_index
            EEE = self.FileHandle.read(self.sizeL)
            (E1,) = struct.unpack("l", EEE)
            assert E1 == Nadj * (nbtokens -2)
            self.FileHandle.seek(2* E1 * self.dtyp_edge_idx.itemsize, 1) # From current position

            #Lire la relation
            relation = np.fromfile(self.FileHandle, dtype=self.dtyp_relation, count=1).reshape(-1)
            tab[ii] = relation[0]
            cumul[1+relation[0]] += 1
        Q = 0
        for i, e in enumerate(cumul):
            Q += e
            cumul[i] = Q
        idx = np.argsort(tab)
        print("Stratified table OK.")
        return idx, cumul

    
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

def count_json_stream(file):
    N = 0
    with open(file, "r", encoding="utf-8") as F:
        data = json_stream.load(F)
        try:
            while(True):
                _ = data[N]
                N += 1
        except IndexError:
            pass
    return N

def iter_json(file):
    NN = count_json_stream(file)
    with open(file, "r", encoding="utf-8") as F:
        data = json_stream.load(F)
        try:
            for N in tqdm(range(NN)):
                resu = json_stream.to_standard_types(data[N])
                yield resu
        except IndexError:
            pass


def filtrer_iter(SOURCE):
    #rel_tg-grp est la relation num_token-groupe, rel_mg est la relation num_mot--groupe,
    # et toks_transfo est la liste in extenso des tokens du transformer
    skipped = 0
    for id, docid, rel_tg, rel_mg, toks_transfo, groupe_source, groupe_cible, relation in SOURCE:

        rel_mP = RELATION("mot", "point")

        rel_mP.add(*[(m, "SOURCE") for m in range(groupe_source[0], 1+groupe_source[1])])
        rel_mP.add(*[(m, "CIBLE") for m in range(groupe_cible[0], 1+groupe_cible[1])])

        rel_tP = (rel_tg * rel_mg * rel_mP).p("token", "point")
        source = [t.token for t in rel_tP.select(lambda x: x.point == "SOURCE")]
        source.sort()
        cible = [t.token for t in rel_tP.select(lambda x: x.point == "CIBLE")]
        cible.sort()
        try:
            assert all(x in source for x in range(source[0], 1+source[-1]))
            assert all(x in cible for x in range(cible[0], 1+cible[-1]))
            assert not any(x in cible for x in source)
            assert not any(x in source for x in cible)
            sommets = source + cible
            aretes = [(i, relation, j) for i in range(len(source)) for j in range(len(source), len(sommets))]
            yield {"id": id, "docid": docid, "tokens": toks_transfo, "sommets": sommets, "aretes": aretes}
        except Exception:
            skipped += 1
    print("Total skipped: %d"%skipped)
    
    

#rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
probleme = {"id": "61b3a65fb92325a4f6ff",
"docid": "eng-NG-31-100512-10873664",
"relation": "no_relation",
"token": ["SPARK", "PLUG", "-LRB-", "PKJ16CR-L11", "-RRB-", "-LRB-", "DENSO", "-RRB-", "SPARK", "PLUG", "-LRB-", "PZFR5F-11", "-RRB-", "-LRB-", "NGK", "-RRB-"],
"subj_start": 6,
"subj_end": 6,
"obj_start": 14,
"obj_end": 14,
"subj_type": "ORGANIZATION",
"obj_type": "ORGANIZATION",
"stanford_pos": ["VB", "NN", "-LRB-", "NN", "-RRB-", "-LRB-", "NNP", "-RRB-", "NNP", "NN", "-LRB-", "NN", "-RRB-", "-LRB-", "NNP", "-RRB-"],
"stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
"stanford_head": [0, 10, 4, 2, 4, 7, 2, 7, 10, 1, 12, 10, 12, 15, 10, 15],
"stanford_deprel": ["ROOT", "compound", "punct", "appos", "punct", "punct", "appos", "punct", "compound", "dobj", "punct", "appos", "punct", "punct", "appos", "punct"]
}

def make_aligs(rep, model_name="roberta-base", file_out="a_tej.json"):
    if file_out.endswith(".json"):
        file_out = file_out[:-5]
    filesOUT = {
            "train": file_out + "_train.json",
            "dev":   file_out + "_dev.json",
            "test":  file_out + "_test.json",
    }
    aligneur = ALIGNEUR(model_name)
    for split in ["train", "dev", "test"]:
        fileIN = os.path.join(rep, split + ".json")
        fileOUT = filesOUT[split]

        iter0 = iter_json(fileIN)
        iter1, iter2, iter_S, iter_C, iter_rel, iter_id, iter_docid = itertools.tee(iter0, 7)

        iter_id = (X["id"] for X in iter_id)
        iter_docid = (X["docid"] for X in iter_docid)
        iter_rel = (X["relation"] for X in iter_rel)
        iter_toks = (X["token"] for X in iter1)
        iter_phrases = (rebuild_sentence(X["token"]) for X in iter2)
        iter_source = ((X["subj_start"], X["subj_end"]) for X in iter_S)
        iter_cible = ((X["obj_start"], X["obj_end"]) for X in iter_C)

        iter_aligs = (aligneur.aligner_seq(T, S) for T, S in zip(iter_toks, iter_phrases))

        iter_argus = ((id,) + (docid,) + X + (S,) + (C,) + (R,) for id, docid, X, S, C, R in zip(iter_id, iter_docid, iter_aligs, iter_source, iter_cible, iter_rel))
        # Iterateur qui donne des 6-uplets avec (id, doc_id, rel_tg, rel_mg, toks_transfo, groupe_source, groupe_cible, relation)

        iterF = filtrer_iter(iter_argus)

        with open(fileOUT, "w", encoding="utf-8") as FOUT:
            print("# ::model_name %s"%model_name, file=FOUT)
            start = "["
            for dico in iterF:
                print("%s%s"%(start, json.dumps(dico)), file=FOUT, end="")
                start = ",\n"
            print("\n]", file=FOUT, end="")



def essai(rep, QscalK=True):
    alig_file, rep_ds_grph, id_model = transfo_to_filenames("robertaBase", QscalK)
    print("alig_file :",alig_file)
    print("rep_ds_grph: ", rep_ds_grph)
    print("id_model:", id_model)
    make_aligs(rep, id_model, alig_file)
    print("TERMINÉ.")

    




if __name__ == "__main__":
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    N = count_json_stream(file)
    print("N= %d"%N)