import os
import random
import numpy as np

def faire_graphe_adjoint(ntokens, tk_utiles, aretes, descr = None):
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


def test():
    tokens = ["\u00a4<s>", "\u00a4Est", "\u00a4ablish", "\u00a4ing", "Models",
              "in", "Industrial", "Innovation", "\u00a4</s>"]
    sommets = [1, 2, 3, 4, 6, 7]
    aretes = [[0, ":>THEME", 3], [0, "{groupe}", 1], [0, "{groupe}", 2],
              [1, ":>THEME", 3], [1, "{groupe}", 0], [1, "{groupe}", 2],
              [2, ":>THEME", 3], [2, "{groupe}", 0], [2, "{groupe}", 1],
              [3, ":mod", 5], [5, ":>THEME", 4]]
    
    dicNdAdj, adjacAdj = faire_graphe_adjoint(len(tokens), sommets, aretes)
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


if __name__ == "__main__":
    test()

        