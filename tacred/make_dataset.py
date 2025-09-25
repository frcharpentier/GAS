import os
from alig.algebre_relationnelle import RELATION
import json_stream
import itertools
from tacred.fichier_zero import rebuild_sentence
from alig.outils_alignement import ALIGNEUR

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
    N = 0
    with open(file, "r", encoding="utf-8") as F:
        data = json_stream.load(F)
        try:
            while(True):
                resu = json_stream.to_standard_types(data[N])
                N += 1
                yield resu
        except IndexError:
            pass




def machin():
    nom_modele = "roberta-base"
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    
    aligneur = ALIGNEUR(nom_modele)
    
    iter0 = iter_json(file)
    iter1, iter2 = itertools.tee(iter0, 2)


    iter_toks = (X["token"] for X in iter1)
    iter_phrases = (rebuild_sentence(X["token"]) for X in iter2)

    iter_aligs = (aligneur.aligner_seq(T, S) for T, S in zip(iter_toks, iter_phrases))
    return iter_aligs




if __name__ == "__main__":
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    N = count_json_stream(file)
    print("N= %d"%N)