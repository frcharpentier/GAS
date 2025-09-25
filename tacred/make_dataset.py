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


def filtrer_iter(rel_tg, rel_mg, groupe_source, groupe_cible, relation, toks_transfo):
    #rel_tg-grp est la relation num_token-groupe, rel_mg est la relation num_mot--groupe,
    # et toks_transfo est la liste in extenso des tokens du transformer
    
    rel_mP = RELATION("mot", "point")

    for m in range(groupe_source[0], 1+groupe_source[1]):
        rel_mP.add((m, "SOURCE"))
    for m in range(groupe_cible[0], 1+groupe_cible[1]):
        rel_mP.add((m, "CIBLE"))

    rel_tP = (rel_tg * rel_mg * rel_mP).p("token", "point")
    source = [t.token for t in rel_tP.select(lambda x: x.point == "SOURCE")]
    source.sort()
    cible = [t.token for t in rel_tP.select(lambda x: x.point == "CIBLE")]
    cible.sort()
    assert all(x in source for x in range(source[0], 1+source[-1]))
    assert all(x in cible for x in range(cible[0], 1+cible[-1]))
    assert not any(x in cible for x in source)
    assert not any(x in source for x in cible)
    sommets = source + cible
    aretes = [(i, relation, j) for i in range(len(source)) for j in range(len(source), len(sommets))]
    return {"tokens": toks_transfo, "sommets": sommets, "aretes": aretes}
    
    



def machin(rep):
    nom_modele = "roberta-base"
    #rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    
    aligneur = ALIGNEUR(nom_modele)
    
    iter0 = iter_json(file)
    iter1, iter2, iter_S, iter_C, iter_rel = itertools.tee(iter0, 5)

    iter_toks = (X["token"] for X in iter1)
    iter_phrases = (rebuild_sentence(X["token"]) for X in iter2)
    iter_rel = (X["relation"] for X in iter_rel)
    iter_source = ((X["subj_start"], X["subj_end"]) for X in iter_S)
    iter_cible = ((X["obj_start"], X["obj_end"]) for X in iter_C)

    iter_aligs = (aligneur.aligner_seq(T, S) for T, S in zip(iter_toks, iter_phrases))

    iter_argus = (X + (S,) + (C,) + (R,) for X, S, C, R in zip(iter_aligs, iter_source, iter_cible, iter_rel))

    resu = (
        filtrer_iter(
            rel_tg=X[0],
            rel_mg=X[1],
            groupe_source=X[3],
            groupe_cible=X[4],
            relation=X[5],
            toks_transfo=X[2]
        ) for X in iter_argus
    )

    return resu




if __name__ == "__main__":
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    N = count_json_stream(file)
    print("N= %d"%N)