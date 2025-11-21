from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import networkx as nx
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data


import numpy as np
import re






class GraphData:
    def __init__(self, x, edge_index):
        self.x = torch.as_tensor(x)
        self.edge_index = torch.as_tensor(edge_index)



def faire_graphe_adjoint(ntokens, tk_utiles, aretes, descr, liste_roles, bilin=True): #, outputARGn=False):
    Nadj = ntokens*(ntokens-1)//2
    # Nombre de sommets du graphe adjoint du graphe complet
    degAdj = 2*ntokens - 4
    # degré de chaque sommet du graphe adjoint du graphe complet

    s, relation, c = aretes[0]
    S, C = tk_utiles[s], tk_utiles[c]
    direction = 1 if S < C else 0
    assert all(r == relation for s,r,c in aretes)
    relation = np.array([liste_roles.index(relation)], dtype=np.int32)
    aretes_utiles = [(tk_utiles[s], tk_utiles[c]) for s,r,c in aretes]
    if direction == 1:
        assert all(S < C for S, C in aretes_utiles)
    else:
        assert all(S>C for S, C in aretes_utiles)
        aretes_utiles = [(c,s) for s, c in aretes_utiles]
    direction = np.array([direction], dtype=np.int32)

    useful_nodes = []

    dim = descr.shape[-1]
    adja = [[False for j in range(i+1,ntokens)] for i in range(ntokens-1)]
    # matrice d’adjacence du graphe normal (pas adjoint)
    if bilin:
        grfSig = np.zeros((Nadj, dim, 2), dtype=descr.dtype)
        # Signal sur le graphe adjoint
    else:
        grfSig = np.zeros((Nadj, dim), dtype=descr.dtype)
        # Signal sur le graphe adjoint

    #edge_idx = np.zeros((2, Nadj*degAdj), dtype=np.int32)
    edge_idx = np.zeros((2, Nadj*degAdj//2), dtype=np.int32)
    # matrice d’adjacence du graphe adjoint au format "edge_index"



    idAdj = []
    
    idxattr = 0
    for s in range(ntokens-1):
        for c in range(s+1, ntokens):
            CC = c-s-1
            lblNd = (s,c)
            if lblNd in aretes_utiles:
                useful_nodes.append(idxattr)
            idAdj.append(lblNd)
            if bilin:
                grfSig[idxattr, :, 0] = descr[s,c,:]
                grfSig[idxattr, :, 1] = descr[c,s,:]
            else:
                grfSig[idxattr, :] = descr[s,c,:]
            
            idxattr += 1

    # Note: Pour obtenir le tenseur grfSig avec torch, faire les opérations suivantes :
    # idx = torch.triu_indices(ntokens, ntokens, offset=1) # indices de la partie triangulaire supérieure d’une matrice (ntokens, ntokens)
    # aa = descr[idx[0], idx[1], :]
    # bb = (descr.transpose(0,1))[idx[0], idx[1], :]
    # grfSig = torch.hstack((aa, bb))
    #
    # Ca, c’est pour une seule phrase. Pour plusieurs, faire :
    #
    # On part d’une liste qui contient les ntoks pour chaque phrase du batch ntoks = [n1, n2, n3]
    # calculer idx =  [torch.vstack((torch.full((n*(n-1)//2,), i), torch.triu_indices(n,n,1))) for i, n in enumerate(ntoks)]
    # calculer idx = torch.hstack(idx)
    # aa = descr[idx[0], idx[1], idx[2], :]
    # bb = (descr.transpose(1,2))[idx[0], idx[1], idx[2], :]
    # grfSig = torch.hstack((aa, bb))

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
                
            for s2 in range(s1+1, ntokens):
                if s2 != c1:
                    if c1 < s2:
                        N2 = numerote(c1, s2)
                    else:
                        N2 = numerote(s2, c1)
                    edge_idx[0, ii] = N1
                    edge_idx[1, ii] = N2
                    ii += 1

    edge_idx = np.concatenate((edge_idx, edge_idx[(1,0),:]), axis=1)
    
    G = GraphData(grfSig, edge_idx)
    useful_nodes = np.array(useful_nodes)

    resu = {
        "idAdj"        : idAdj,
        "G"            : G,
        "utiles"       : useful_nodes,
        "relation"     : relation,
        "direction"    : direction
    }
    
    return resu