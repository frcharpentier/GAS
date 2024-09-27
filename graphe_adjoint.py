import os
import logging
import random
import numpy as np
import torch
from outils_alignement import ALIGNEUR
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import utils as transfo_utils


from minbert.model import BERT, param_translation
from mingpt.model import GPT

def faire_graphe_adjoint(ntokens, tk_utiles, aretes, descr, liste_roles, bilin=True):
    Nadj = ntokens*(ntokens-1)//2
    # Nombre de sommets du graphe adjoint du graphe complet
    degAdj = 2*ntokens - 4
    # degré de chaque sommet du graphe adjoint du graphe complet

    dim = descr.shape[-1]
    adja = [[False for j in range(i+1,ntokens)] for i in range(ntokens-1)]
    # matrice d’adjacence du graphe normal (pas adjoint)
    if bilin:
        grfSig = np.zeros((Nadj, dim, 2))
        # Signal sur le graphe adjoint
    else:
        grfSig = np.zeros((Nadj, dim))
        # Signal sur le graphe adjoint

    edge_idx = np.zeros((2, Nadj*degAdj), dtype=np.int64)
    # matrice d’adjacence du graphe adjoint au format "edge_index"



    idAdj = []
    roles = np.zeros((Nadj,), dtype=np.int16)
    sens = np.zeros((Nadj,), dtype=np.int8)
    msk_sens = np.zeros((Nadj,),dtype="bool")
    msk_roles = np.zeros((Nadj,),dtype="bool")

    for s,r,c in aretes:
        S, C = tk_utiles[s], tk_utiles[c]
        assert S != C
        I,J = (S,C) if S<C else (C,S)
        J = J-I-1
        adjc = adja[I][J]
        if adjc == False:
            if r.startswith("{") and r.endswith("}"):
                # Relations sans direction comme {groupe}, {idem}, {inter}...
                adja[I][J] = (None, r, None)
            else:
                adja[I][J] = (S,r,C)
        else:
            ss, rr, cc = adjc
            if r.startswith("{") and r.endswith("}"):
                if r == rr:
                    adja[I][J] = (None, r, None) #Relation mais pas de sens
                else:
                    adja[I][J] = (None, None, None) #Ni relation ni sens
            else:
                if ss == S and cc == C:
                    sss, ccc = S, C
                else:
                    sss, ccc = (None, None) #Pas de sens
                if rr == r:
                    rrr = r
                else:
                    rrr = None #Pas de relation
                adja[I][J] = (sss, rrr, ccc)
        

    idxattr = 0
    for s in range(ntokens-1):
        for c in range(s+1, ntokens):
            CC = c-s-1
            lblNd = (s,c)
            idAdj.append(lblNd)
            if bilin:
                grfSig[idxattr, :, 0] = descr[s,c,:]
                grfSig[idxattr, :, 1] = descr[c,s,:]
            else:
                grfSig[idxattr, :] = descr[s,c,:]
            
            arete = adja[s][CC]
            if arete == False:
                roles[idxattr] = 0
                msk_roles[idxattr] = False
                msk_sens[idxattr] = False
                # Ni relation ni sens à prédire pour ce nœud.
            else:
                source, r, cible = arete
                assert source == s or source == c or source == None
                assert cible == c or cible == s or cible == None
                if source == None:
                    # Relations sans direction comme {groupe}, {idem}, {inter}...
                    msk_sens[idxattr] = False
                    # Pas de sens particulier à prédire
                elif source == s:
                    sens[idxattr] = 1.
                    msk_sens[idxattr] = True
                    # Prédire le sens
                else:
                    sens[idxattr] = 0.
                    msk_sens[idxattr] = True
                if r == None or r.startswith("?"):
                    # Si le rôle est un rôle ?ARGn non résolu :
                    roles[idxattr] = 0
                    msk_roles[idxattr] = False
                    # Pas de relation à prédire
                else:
                    roles[idxattr] = liste_roles.index(r)
                    msk_roles[idxattr] = True
                    # Prédire la relation
            idxattr += 1

    # Construction de la matrice d’adjacence au format "edge_index" :
    numerote = lambda s,c : (2*ntokens-1-s)*s//2+c-s-1
    # Fonction pour obtenir le numéro d’un sommet adjoint (s,c) (avec s<c)

    ii = 0
    for s1 in range(ntokens):
        for c1 in range(s1+1, ntokens):
            N1 = numerote(s1, c1)
            for c2 in range(c1+1, ntokens):
                N2 = numerote(s1, c2)
                edge_idx[0, ii] = N1
                edge_idx[1, ii] = N2
                ii += 1
                edge_idx[0, ii] = N2
                edge_idx[1, ii] = N1
                ii += 1
                
            for s2 in range(s1+1, ntokens):
                if s2 != c1:
                    if c1 < s2:
                        N2 = numerote(c1, s2)
                    else:
                        N2 = numerote(s2, c1)
                    edge_idx[0, ii] = N1
                    edge_idx[1, ii] = N2
                    ii += 1
                    edge_idx[0, ii] = N2
                    edge_idx[1, ii] = N1
                    ii += 1


    return idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens
        

    


def faire_graphe_adjoint_0(ntokens, tk_utiles, aretes, descr = None):
    if descr is None: #uniquement pour le débug
        descr = np.random.randn(ntokens, ntokens, 5)
    dim = descr.shape[-1]
    dicNdAdj = {} #Description des sommets du graphe adjoint
    adjacAdj = set() #Matrice d’adjacence du graphe adjoint
    ordres = ["sc", "cs"]

    #description des nœuds adjoints utiles à la phrase
    for s,r,c in aretes:
        s, c = tk_utiles[s], tk_utiles[c]
        descNd = {}
        lblNd = (s,c) if s<c else (c,s)
        dicNdAdj[lblNd] = descNd
        ordre = random.choice(ordres)
        if not r.startswith("?"):
            descNd["role"] = r
        if ordre == "sc":
            descNd["sens"] = 1
            descNd["attr"] = np.concatenate((descr[s,c].reshape(1,dim), descr[c,s].reshape(1,dim)))
        else:
            descNd["sens"] = 0
            descNd["attr"] = np.concatenate((descr[c,s].reshape(1,dim), descr[s,c].reshape(1,dim)))
                    
    #description des autres nœuds adjoints
    for s in range(ntokens-1):
        for c in range(s+1, ntokens):
            lblNd = (s,c)
            if lblNd in dicNdAdj:
                continue
            descNd = {}
            dicNdAdj[lblNd] = descNd
            ordre = random.choice(ordres)
            if ordre == "sc":
                descNd["attr"] = np.concatenate((descr[s,c].reshape(1,dim), descr[c,s].reshape(1,dim)))
            else:
                descNd["attr"] = np.concatenate((descr[c,s].reshape(1,dim), descr[s,c].reshape(1,dim)))

    #Matrice d’adjacence
    for s1 in range(ntokens):
        for c1 in range(s1+1, ntokens):
            #lbl1 = (s1,c1)
            for c2 in range(c1+1, ntokens):
                #lbl2 = (s1, c2)
                adjacAdj.add(((s1,c1),(s1,c2)))
            for s2 in range(s1+1, ntokens):
                if s2 != c1:
                    if c1 < s2:
                        lbl2 = (c1, s2)
                    else:
                        lbl2 = (s2, c1)
                    adjacAdj.add(((s1,c1),lbl2))

    return dicNdAdj, adjacAdj


class TRANSFORMER_ATTENTION:
    def __init__(self, QscalK = False, dtype=np.float32):
        self.dtype = dtype
        self.modele = None
        self.tokenizer = None
        #self.colonnes = ["source_transfo", "source_AMR", "target_transfo", "target_AMR", "relation"]
        self.num_layers = 1
        self.num_heads = 1
        self.data_att = None #[]
        self.QK = QscalK
        #self.suffixes = ["SC", "CS", "Ssep", "Csep"]
        if self.QK:
            self.data_QK = None #[]
        self.type_transformer = "ENC"
        
        
    def select_modele(self, model_name, decoder_mask = True):
        try:
            self.model_type = "hf"
            if model_name.startswith("minBERT://") or model_name.startswith("minbert://"):
                self.model_type = "minBERT"
                model_name = model_name[10:]
            elif model_name.startswith("minGPT://") or model_name.startswith("mingpt://"):
                self.model_type = "minGPT"
                model_name = model_name[9:]
                self.type_transformer = "DEC"
            elif model_name.startswith("hf://") or model_name.startswith("HF://"):
                self.model_type = "hf"
                model_name = model_name[5:]
                if model_name.startswith("gpt"):
                    self.type_transformer = "DEC"
            elif model_name.startswith("huggingface://"):
                self.model_type = "hf"
                model_name = model_name[14:]

            if self.type_transformer == "DEC":
                self.decoder_mask = decoder_mask

            if self.model_type == "minBERT":
                self.modele = BERT.from_huggingface_model_name(model_name)
                self.num_layers = len(self.modele.encoder)
                self.num_heads = self.modele.encoder[0].attn.n_head
            elif self.model_type == "minGPT":
                self.modele = GPT.from_pretrained(model_name)
                self.num_layers = len(self.modele.transformer.h)
                self.num_heads = self.modele.transformer.h[0].attn.n_head
            else:
                self.modele = AutoModel.from_pretrained(model_name, output_attentions=True)
                config = self.modele.config
                self.model_type = config._name_or_path
                self.num_layers = config.num_hidden_layers
                self.num_heads = config.num_attention_heads

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ch_debut, self.ch_suite, self.tok1, self.sep_token = ALIGNEUR.detecter_params(self.tokenizer)
            #self.sep_token = self.tokenizer.sep_token
            #if self.sep_token == None:
            #    pass
            #else:
            #    self.sep_token = self.tokenizer.convert_tokens_to_ids(self.sep_token)

        except Exception as EX:
            logging.debug("Problème : Exception !")
            self.modele = None
            self.tokenizer = None
            raise EX
            return False
        else:
            self.modele.eval()
            return True
        
    def compute_attn_tensor(self, snt):
        assert (self.modele != None and self.tokenizer != None)
        #snt = argjson["snt"]
        #aretes = argjson["aretes"]
        if type(snt) is "str":
            inputs = self.tokenizer(snt, return_tensors='pt')
            input_ids = inputs['input_ids']
            att_mask = inputs["attention_mask"]
        else:
            assert type(snt) == list
            assert len(snt) > 0 and type(snt[0]) in [int, str]
            assert all(type(mot) is type(snt[0]) for mot in snt)
            if type(snt[0]) is str:
                snt = [ T if T in (self.tok1, self.sep_token) else (self.ch_suite + T[1:] if T.startswith("¤") else self.ch_debut + T) for T in snt]
                snt = self.tokenizer.convert_tokens_to_ids(snt)
            input_ids = torch.tensor(snt).reshape(1,-1)
            att_mask = torch.ones(input_ids.shape, dtype=input_ids.dtype)

        ntokens = input_ids.shape[-1]
            
        
        with torch.no_grad():
            #calcul du résultat par le transformer
            if self.model_type == "minBERT":
                if self.QK:
                    _, _, attention, QscalK = self.modele(input_ids, att_mask, output_att = True, output_QK = True)
                    QscalK = [X.detach().numpy() for X in QscalK]
                else:
                    _, _, attention = self.modele(input_ids, att_mask, output_att = True)
            elif self.model_type == "minGPT":
                if self.QK:
                    _, _, attention, QscalK = self.modele(input_ids, output_att = True, output_QK = True)
                    QscalK = [X.detach() for X in QscalK]
                    if not self.decoder_mask:
                        softmaxQK = [F.softmax(X, dim=-1).numpy() for X in QscalK]
                        #Recalculer le softmax à partir des produits scalaires
                        #non masqués
                    QscalK = [X.numpy() for X in QscalK]
                else:
                    _, _, attention = self.modele(input_ids, att_mask, output_att = True)
            else:
                result = self.modele(input_ids)
                attention = result.attentions

        # la variable "attention" est un tuple de tenseurs.
        # Tous sont du même ordre, tous ont les mêmes dimensions.
        # en l’occurrence, (x, h, w, w), où x est le nombre de phrases dans le batch,
        # h est le nombre de têtes, w est le nombre de mots dans la phrase encodée.
        attention = [X.detach().numpy() for X in attention]

        if self.type_transformer == "DEC":
            if self.decoder_mask:
                triangles = [np.tril(att) for att in attention]
                trig1 = [np.tril(att, k=-1) for att in attention]
                trig1t = [np.transpose(M, axes=(0,1,3,2)) for M in trig1]
                attmodif = [A+B for A,B in zip(triangles, trig1t)]
                dim = self.num_heads*self.num_layers
                self.data_att = np.concatenate(tuple(att for att in attmodif), axis=3)
                self.data_att = self.data_att.reshape(ntokens, ntokens, dim)
                if self.QK:
                    self.data_QK = np.concatenate(tuple(att for att in QscalK), axis=3)
                    self.data_QK = self.data_QK.reshape(ntokens, ntokens, dim)
            else:
                dim = self.num_heads*self.num_layers
                self.data_att = np.concatenate(tuple(att for att in softmaxQK), axis=3)
                self.data_att = self.data_att.reshape(ntokens, ntokens, dim)
                if self.QK:
                    self.data_QK = np.concatenate(tuple(att for att in QscalK), axis=3)
                    self.data_QK = self.data_QK.reshape(ntokens, ntokens, dim)
        else: #if self.type_transfo == "ENC"
            dim = self.num_heads*self.num_layers
            self.data_att = np.concatenate(tuple(att for att in attention), axis=3)
            self.data_att = self.data_att.reshape(ntokens, ntokens, dim)
            if self.QK:
                self.data_QK = np.concatenate(tuple(att for att in QscalK), axis=3)
                self.data_QK = self.data_QK.reshape(ntokens, ntokens, dim)

    def compute_line_graph(self):
        pass
                



def test():
    tokens = ["\u00a4<s>", "\u00a4Est", "\u00a4ablish", "\u00a4ing", "Models",
              "in", "Industrial", "Innovation", "\u00a4</s>"]
    sommets = [1, 2, 3, 4, 6, 7]
    aretes = [[0, ":>THEME", 3], [0, "{groupe}", 1], [0, "{groupe}", 2],
              [1, ":>THEME", 3], [1, "{groupe}", 0], [1, "{groupe}", 2],
              [2, ":>THEME", 3], [2, "{groupe}", 0], [2, "{groupe}", 1],
              [3, ":mod", 5], [5, ":>THEME", 4]]
    
    dicNdAdj, adjacAdj = faire_graphe_adjoint_0(len(tokens), sommets, aretes)
    print("Tokens:")
    print(tokens)
    print("--------------")
    print("Descripteurs de sommets du graphe adjoint :")
    for (s,c), v in dicNdAdj.items():
        k = "%s ~~~ %s"%(tokens[s], tokens[c])
        D = dict()
        if "role" in v:
            D["role"] = v["role"]
        if "sens" in v:
            D["sens"] = v["sens"]
        print("%s : %s"%(k, str(D)))
    print("--------------")
    print("Adjacence du graphe adjoint :")
    for ((s1,c1),(s2,c2)) in adjacAdj :
        k1 = "%s ~~~ %s"%(tokens[s1], tokens[c1])
        k2 = "%s ~~~ %s"%(tokens[s2], tokens[c2])
        print("[%s] ### [%s]"%(k1,k2))

def test2():
    snt = "Establishing Models in Industrial Innovation"
    trafo = TRANSFORMER_ATTENTION()
    #trafo.select_modele("minBERT://roberta-base")
    trafo.select_modele("minGPT://gpt2")
    trafo.compute_attn_tensor(snt)

def test3():
    tokens = ["\u00a4<s>", "\u00a4Est", "\u00a4ablish", "\u00a4ing", "Models",
              "in", "Industrial", "Innovation", "\u00a4</s>"]
    sommets = [1, 2, 3, 4, 6, 7]
    aretes = [[0, ":>THEME", 3], [0, "{groupe}", 1], [0, "{groupe}", 2],
              [1, ":>THEME", 3], [1, "{groupe}", 0], [1, "{groupe}", 2],
              [2, ":>THEME", 3], [2, "{groupe}", 0], [2, "{groupe}", 1],
              [3, ":mod", 5], [5, ":>THEME", 4]]
    
    jsn = {"tokens": ["\u00a4<s>", "The", "lack", "or", "serious", "shortage", "of", "intermediate", "layers", "of", "Party", "organizations", "and", "units", "between", "the", "two", "has", "resulted", "in", "its", "inability", "to", "consider", "major", "issues", "with", "endless", "minor", "issues", "on", "hand", "\u00a4,", "such", "that", "even", "if", "it", "is", "highly", "capable", "\u00a4,", "it", "won", "\u00a4't", "last", "long", "\u00a4,", "as", "it", "will", "be", "dragged", "down", "by", "numerous", "petty", "things", "\u00a4.", "\u00a4</s>"], "sommets": [2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 39, 40, 44, 45, 46, 48, 52, 53, 55, 56, 57], "dicTokens": [["1.1.1"], ["1.1.2.3"], ["1.1.2"], ["1.1.1.2.1"], ["1.1.1.2"], ["1.1.1.2.2.3"], ["1.1.1.2.2.1"], ["1.1.1.2.2.2"], ["1.1.1.1"], ["1.1.1.1.1", "1.1.1.1.1.1"], ["1.2.3"], ["1"], ["1.1.1.2.2.3"], ["1.2.3.2"], ["1.2.2"], ["1.2.2.2.1"], ["1.2.2.2"], ["1.2.3.1.2"], ["1.2.3.1.1"], ["1.2.3.1"], ["1.2"], ["1.2"], ["1.3"], ["1.3"], ["1.3.1.4"], ["1.3.1.4"], ["1.2.1"], ["1.3.1.2.2"], ["1.3.1.2"], ["1.3.1.1"], ["1.3.1"], ["1.3.1.3"], ["1.3.1.5"], ["1.3.1.5.1"], ["1.3.1.5.1.3"], ["1.3.1.5.1.1.1"], ["1.3.1.5.1.1.2"], ["1.3.1.5.1.1"]], "aretes": [[0, ":>AGENT", 9], [0, ":>THEME", 4], [0, "{or}", 2], [2, ":mod", 1], [2, "?ARG2", 4], [2, "{or}", 0], [4, ":mod", 3], [5, ":mod", 28], [5, ":part", 6], [5, ":part", 7], [5, "{idem}", 12], [6, ":part", 4], [6, "{and}", 7], [7, ":part", 4], [7, "{and}", 6], [8, "{syntax}", 9], [9, ":mod", 2], [10, ":>ATTRIBUTE", 19], [10, ":location", 13], [11, ":>GOAL", 20], [11, ":>GOAL", 21], [11, ":>THEME", 0], [11, ":>THEME", 2], [12, ":mod", 28], [12, ":part", 6], [12, ":part", 7], [12, "{idem}", 5], [14, ":>AGENT", 5], [14, ":>AGENT", 12], [14, ":>THEME", 16], [14, ":mod", 20], [14, ":mod", 21], [16, ":mod", 15], [19, ":mod", 18], [19, ":quant", 17], [20, ":condition", 10], [20, ":polarity", 26], [20, "{groupe}", 21], [21, ":condition", 10], [21, ":polarity", 26], [21, "{groupe}", 20], [22, ":>AGENT", 11], [22, ":>RESULT", 30], [22, "{groupe}", 23], [23, ":>AGENT", 11], [23, ":>RESULT", 30], [23, "{groupe}", 22], [24, "{groupe}", 25], [24, "{syntax}", 28], [25, "{groupe}", 24], [25, "{syntax}", 28], [28, ":degree", 27], [30, ":>DESTINATION", 31], [30, ":>THEME", 28], [30, ":concession", 28], [30, ":polarity", 29], [31, "?ARG1", 28], [32, ":>AGENT", 33], [32, ":>RESULT", 30], [33, ":>AGENT", 37], [33, ":>DESTINATION", 34], [33, ":>THEME", 28], [37, ":mod", 36], [37, ":quant", 35]]}

    tokens = jsn["tokens"]
    sommets = jsn["sommets"]
    aretes = jsn["aretes"]
    
    attn = TRANSFORMER_ATTENTION()
    attn.select_modele("minbert://roberta-base")
    attn.compute_attn_tensor(tokens)
    idAdj, grfSig, edge_idx, roles, sens, msk_roles, msk_sens = faire_graphe_adjoint(len(tokens), sommets, aretes, attn.data_att)
    pass

if __name__ == "__main__":
    test3()

        