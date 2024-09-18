import os
import sys
import json
from tqdm import tqdm
import re

from transformers import AutoTokenizer
from amr_utils.amr_readers import AMR_Reader
from amr_utils.amr_readers import Matedata_Parser as Metadata_Parser
from amr_utils.alignments import AMR_Alignment
from examiner_framefiles import EXPLICITATION_AMR #expliciter_AMR
from algebre_relationnelle import RELATION
from enchainables import MAILLON
from outils_alignement import (DICO_ENUM,
    GRAPHE_PHRASE, ALIGNEUR, preparer_alignements,
    compter_compos_connexes)


@MAILLON
def iterer_alignements(SOURCE, alignements):
    print(" *** DÉBUT ***")

    for idSNT, listAlig in tqdm(alignements.items()):
        yield idSNT, listAlig

@MAILLON
def filtrer_sous_graphes(SOURCE):
    for idSNT, listAlig in SOURCE:
        eliminer = []
        for i, a in enumerate(listAlig):
            if a.type in ("subgraph", "dupl-subgraph"):
                N = compter_compos_connexes(a.edges)
                if N > 1:
                    #print("AMR %s, tokens %s, %d composantes"%(idSNT, str([a.amr.words[tt] for tt in a.word_ids]), N))
                    eliminer.append(i)
        if len(eliminer) > 0:
            listAlig = [ a for i, a in enumerate(listAlig) if not i in eliminer]
        yield idSNT, listAlig



@MAILLON
def filtrer_ss_grf_2(SOURCE):
    for idSNT, listAlig in SOURCE:
        amr = listAlig[0].amr
        eliminer = []
        for i, a in enumerate(listAlig):
            if (a.type in ("subgraph", "dupl-subgraph")) and (len(a.nodes) > 1):
                sommets = [amr.isi_node_mapping[som] for som in a.nodes]
                aretes = [(amr.isi_node_mapping[s], amr.isi_node_mapping[c]) for s, _, c in amr.edges]
                aretes = [(s,c) for s,c in aretes if (s in sommets) != (c in sommets)] # l’opérateur != fait exactement office de ou exclusif.
                aretes = set((s if s in sommets else c) for s, c in aretes)
                if len(aretes) > 1:
                    eliminer.append(i)
        if len(eliminer) > 0:
            listAlig = [ a for i, a in enumerate(listAlig) if not i in eliminer]
        yield idSNT, listAlig


    
        

@MAILLON
def iterer_graphe(SOURCE):
    aligneur = ALIGNEUR("roberta-base", ch_debut="Ġ", ch_suite="")
    for idSNT, listAlig in SOURCE:
        amr = listAlig[0].amr
        mots_amr = amr.words
        snt = amr.metadata["snt"]

        graphe = GRAPHE_PHRASE(amr)
        #if amr.id == "bolt12_10474_1831.9":
        #    pass

        trf_grp, amr_grp, toks_transfo = aligneur.aligner_seq(mots_amr, snt)
        #trf-grp est la relation num_token-groupe, amr_grp est la relation num_mot--groupe,
        # et toks_transfo est la liste in extenso des tokens du transformer
        graphe.ajouter_aligs(listAlig, toks_transfo, trf_grp, amr_grp)

        yield idSNT, amr, graphe









@MAILLON
def filtrer_reentrance(SOURCE):
    for idSNT, amr, graphe in SOURCE:
        if not graphe.verifier_reentrance():
            #Éliminons ces AMR problématiques
            #continue
            yield idSNT, amr, graphe

        #yield idSNT, amr, graphe

@MAILLON
def filtrer_anaphore_0(SOURCE):
    autorises = ["reentrancy:repetition", "reentrancy:coref"] #"reentrancy:primary"]
    for idSNT, amr, graphe in SOURCE:
        try:
            graphe.traiter_reentrances()
        except AssertionError:
            continue

        yield idSNT, amr, graphe


@MAILLON
def filtrer_anaphore(SOURCE):
    autorises = ["reentrancy:repetition", "reentrancy:coref"] #, "reentrancy:primary"]
    for idSNT, amr, graphe in SOURCE:
        graphe.anaphores_mg = RELATION(*(graphe.REN_mg.sort))
        graphe.anaphores_ag = RELATION(*(graphe.REN_ag.sort))
        if len(graphe.REN_ag) == 0:
            yield idSNT, amr, graphe
            continue
        toks = graphe.tokens
        settoks = set(m for (m, _) in graphe.SG_mg)
        toks_libres = [not t in settoks for t in range(graphe.N)]
        #liste_toks_libres = [toks[i] if v else "" for i, v in enumerate(toks_libres)]
        
        #libres2 = [True] * graphe.N

        #primaires = REL_ren.select(lambda x: x.type == "reentrancy:primary")
        graphe.anaphores_mg = graphe.REN_mg.select(lambda x: (x.type in autorises))
        graphe.anaphores_ag = graphe.REN_ag.select(lambda x: (x.type in autorises))#.proj("cible", "type", "groupe").rmdup()
        #mots = graphe.anaphores_mg.proj("mot").rmdup()
        #if len(mots) == 0:
        #    yield idSNT, amr, graphe
        #    continue

        #for (t,) in mots:
        #    if not toks_libres[t]:
        #        print("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
        #        break


        anaph = (graphe.anaphores_mg.p("mot", "groupe") * graphe.anaphores_ag.p("cible", "groupe")).p("mot", "cible", "groupe")

        r1 = (anaph.ren("tok", "c1", "g1") * anaph.ren("tok", "c2", "g2")).s(lambda x: x.c1 < x.c2)
        r2 = r1.p("g1").ren("grp") + r1.p("g2").ren("grp")
        r3 = anaph.s(lambda x: not toks_libres[x.mot]).p("groupe").ren("grp")
        if len(r3) > 0:
            print("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
            r2 = r2 + r3
        if len(r2) > 0:
            interdits = set(g for (g,) in r2)
            graphe.anaphores_mg = graphe.anaphores_mg.s(lambda x: not(x.groupe in interdits))
            graphe.anaphores_ag = graphe.anaphores_ag.s(lambda x: not(x.groupe in interdits))
            #print("%s : Anaphores corrigées."%idSNT)
    

        yield idSNT, amr, graphe
        

@MAILLON
def calculer_graphe_toks(SOURCE):
    for idSNT, amr, graphe in SOURCE:
        ok = graphe.calculer_graphe_toks()
        if not ok:
            continue
        yield idSNT, amr, graphe

@MAILLON
def traiter_graphe_toks(SOURCE):
    conjonctions = {"and": "{and}", "or":"{or}", "slash":"{or}"}
    for idSNT, amr, graphe in SOURCE:
        gr = graphe.graphe_toks
        aelim = gr.s(lambda x: re.match(":op\d+", x.rel))
        if len(aelim) > 0:
            sommets_a_elim = [ sss for (sss,) in aelim.p("mot_s")]
            parents = gr.s(lambda x: x.mot_c in sommets_a_elim).ren("mot_s", "pivot", "rel")
            desc = gr.s(lambda x: x.mot_s in sommets_a_elim).ren("pivot", "mot_c", "rel2")
            desc2 = desc.s(lambda x: re.match(":op\d+", x.rel2)).p("pivot", "mot_c")
            R1 = graphe.SG_mg.p("mot", "groupe").s(lambda x: x.mot in sommets_a_elim)
            R2 = graphe.SG_sg.p("sommet", "groupe")
            R3 = (R1*R2).p("mot", "sommet").ren("pivot", "sommet")
            pivots = RELATION("pivot", "conj")
            for T in R3:
                if T.sommet in amr.nodes:
                    amrn = amr.nodes[T.sommet]
                    if amrn in conjonctions:
                        pivots.add((T.pivot, conjonctions[amrn]))
            
            cjx_trp = (desc2.ren("pivot", "m1")) * pivots * (desc2.ren("pivot", "m2"))
            cjx_trp = cjx_trp.p("m1", "m2", "conj").s(lambda x: x.m1 != x.m2)


            nvx_trp = (parents*desc2).p("mot_s", "mot_c", "rel")
            gr = gr.s(lambda x: not(x in parents.table or x in desc.table))
            gr = gr + nvx_trp
            gr2 = gr.p("mot_s", "mot_c")
            cjx_trp = cjx_trp.s(lambda x: not (x.m1, x.m2) in gr2.table)
            gr = gr + (cjx_trp.ren("mot_s", "mot_c", "rel"))
            graphe.graphe_toks = gr

        yield idSNT, amr, graphe

@MAILLON
def ecrire_liste(SOURCE, fichier_out = "./AMR_et_graphes_phrases_2.txt"):
    NgraphesEcrits = 0
    limNgraphe = -1
    with open(fichier_out, "w", encoding="UTF-8") as FF:
        for idSNT, amr, graphe in SOURCE:
            print(amr.amr_to_string(), file=FF)
            jsn = graphe.jsonifier()
            print(jsn, file=FF)
            print(file=FF)
            NgraphesEcrits += 1
            if limNgraphe > 0 and NgraphesEcrits > limNgraphe:
                break


def faire_liste_1(fichier_out="a_tej.txt"):

    kwargs = dict()
    prefixe_alignements = "../alignement_AMR/leamr/data-release/alignments/ldc+little_prince."
    kwargs["fichier_sous_graphes"] = prefixe_alignements + "subgraph_alignments.json"
    kwargs["fichier_reentrances"] = prefixe_alignements + "reentrancy_alignments.json"
    kwargs["fichier_relations"] = prefixe_alignements + "relation_alignments.json"
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    snt_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/unsplit"
    kwargs["doublons"] = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']
    # Cette liste est une liste d’identifiants AMR en double dans le répertoire amr_rep
    # Il n’y en a que quatre. On les éliminera, c’est plus simple, ça ne représente que huit AMR.
    # Cette liste a été établie en exécutant la fonction "dresser_liste_doublons" ci-dessus.
    kwargs["fichiers_amr"] = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    kwargs["fichiers_snt"] = [os.path.abspath(os.path.join(snt_rep, f)) for f in os.listdir(snt_rep)]
    

    alignements = preparer_alignements(explicit_arg=True, **kwargs)
    chaine = iterer_alignements(alignements)
    chaine = chaine >> filtrer_sous_graphes() >> filtrer_ss_grf_2()
    chaine = chaine >> iterer_graphe() >> filtrer_anaphore()
    chaine = chaine >> calculer_graphe_toks()
    chaine = chaine >> ecrire_liste(fichier_out = fichier_out)

    chaine.enchainer()



def faire_dico_aligs():
    prefixe_alignements = "../alignement_AMR/leamr/data-release/alignments/ldc+little_prince."
    json_files = [None, None, None]
    json_files[0] = prefixe_alignements + "subgraph_alignments.json"
    json_files[1] = prefixe_alignements + "reentrancy_alignments.json"
    json_files[2] = prefixe_alignements + "relation_alignments.json"

    alignments = dict()
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf8') as f:
            aligs = json.load(f)
        for k, v in aligs.items():
            assert type(v) is list
            if not k in alignments:
                alignments[k] = v
            else:
                alignments[k].extend(v)

    return alignments


def transfo_aligs(amr, jason):
    egal = lambda x,y: (amr.isi_node_mapping[x] == amr.isi_node_mapping[y])
    aligs = []
    elimine = False
    for a in jason:
        if 'nodes' not in a:
            a['nodes'] = []
        if 'edges' not in a:
            a['edges'] = []
        for i,e in enumerate(a['edges']):
            s,r,t = e
            if r is None:
                try:
                    new_e = [e_2 for e_2 in amr.edges if egal(e_2[0],s) and egal(e_2[2],t)]
                except KeyError:
                    print('Failed to un-anonymize:', amr.id, e)
                    elimine = True
                    break #sortir de for i,e
                else:
                    new_e = new_e[0]
                    a['edges'][i] = [s, new_e[1], t]
        if elimine:
            break #sortir de for a
        alig = AMR_Alignment(a['type'], a['tokens'], a['nodes'], [tuple(e) for e in a['edges']] if "edges" in a else None)
        alig.word_ids = alig.tokens
        del alig.tokens
        alig.amr = amr
        aligs.append(alig)
    if elimine:
        return []
    else:
        return aligs


def refaire_probleme():
    AMR = """# ::id bolt-eng-DF-200-192451-5797408_0021.7 ::amr-annotator UCO-AMR-05 ::preferred 
# ::alignments 0-1.1 1-1 4-1.2.2 5-1.2 6-1.2.1.r 8-1.2.1.2 10-1.2.1.1 10-1.2.1.1.r 11-1.2.1 12-1.2.1 15-1 16-1.2.3.1 16-1.2.3.1.r 17-1.2.3 17-1.2.3.2.1 18-1.2.3.2 19-1.2.3.2.1.1 19-1.2.3.2.1.1.r 20-1.2.3.2.1.2
# ::snt You thought it was perfectly okay for the world to not take action, that's your view but not mine.
# ::tok You thought it was perfectly okay for the world to not take action , that 's your view but not mine .
(t / think-01~e.1,15 
      :ARG0 (y / you~e.0) 
      :ARG1 (o / okay-04~e.5 
            :ARG1~e.6 (a / act-02~e.11,12 :polarity~e.10 -~e.10 
                  :ARG0 (w / world~e.8)) 
            :ARG1-of (p / perfect-02~e.4) 
            :ARG2-of (v / view-02~e.17 
                  :ARG0~e.16 y~e.16 
                  :ARG1-of (c / contrast-01~e.18 
                        :ARG2 (v2 / view-02~e.17 :polarity~e.19 -~e.19 
                              :ARG0 (i / i~e.20) 
                              :ARG2 o)))))"""
    
    aligs = [{'type': 'subgraph', 'tokens': [0], 'nodes': ['1.1']},
{'type': 'subgraph', 'tokens': [1], 'nodes': ['1']},
{'type': 'subgraph', 'tokens': [4], 'nodes': ['1.2.2']},
{'type': 'subgraph', 'tokens': [5], 'nodes': ['1.2']},
{'type': 'subgraph', 'tokens': [8], 'nodes': ['1.2.1.2']},
{'type': 'subgraph', 'tokens': [10], 'nodes': ['1.2.1.1']},
{'type': 'subgraph', 'tokens': [12], 'nodes': ['1.2.1']},
{'type': 'subgraph', 'tokens': [17], 'nodes': ['1.2.3.2.1']},
{'type': 'subgraph', 'tokens': [18], 'nodes': ['1.2.3.2']},
{'type': 'subgraph', 'tokens': [19], 'nodes': ['1.2.3.2.1.1']},
{'type': 'subgraph', 'tokens': [20], 'nodes': ['1.2.3.2.1.2']},
{'type': 'dupl-subgraph', 'tokens': [17], 'nodes': ['1.2.3']},
{'type': 'reentrancy:primary', 'tokens': [1], 'edges': [['1', None, '1.2']]},
{'type': 'reentrancy:primary', 'tokens': [1], 'edges': [['1', None, '1.1']]},
{'type': 'reentrancy:coref', 'tokens': [16], 'edges': [['1.2.3.2.1', None, '1.2']]},
{'type': 'reentrancy:coref', 'tokens': [16], 'edges': [['1.2.3', None, '1.1']]},
{'type': 'relation', 'tokens': [1], 'edges': [['1', None, '1.1'], ['1', None, '1.2']]},
{'type': 'relation', 'tokens': [4], 'edges': [['1.2', None, '1.2.2']]},
{'type': 'relation', 'tokens': [5], 'edges': [['1.2', None, '1.2.1']]},
{'type': 'relation', 'tokens': [10], 'edges': [['1.2.1', None, '1.2.1.1']]},
{'type': 'relation', 'tokens': [12], 'edges': [['1.2.1', None, '1.2.1.2']]},
{'type': 'relation', 'tokens': [17], 'edges': [['1.2', None, '1.2.3'], ['1.2.3', None, '1.1'], ['1.2.3.2.1', None, '1.2.3.2.1.2'], ['1.2.3.2.1', None, '1.2']]},
{'type': 'relation', 'tokens': [18], 'edges': [['1.2.3', None, '1.2.3.2'], ['1.2.3.2', None, '1.2.3.2.1']]},
{'type': 'relation', 'tokens': [19], 'edges': [['1.2.3.2.1', None, '1.2.3.2.1.1']]}]


    AMR = """# ::id bolt12_4474_0751.12 ::amr-annotator SDL-AMR-09 ::preferred 
# ::alignments 0-1.3 3-1.1.1.1 4-1.1.1 5-1.1 6-1.1.2.2 7-1.1.2.1 8-1.1.2 9-1 10-1.2.1 12-1.2.2.1 13-1.2.2.1 13-1.2.2.1.1 13-1.2.2.1.1.r 13-1.2.2.1.2 13-1.2.2.1.2.r 14-1.2 15-1.2.3.r 17-1.2.3 18-1.2.3.1.r 20-1.2.3.1
# ::snt Meanwhile, the crowded highways and complicated road traffic cause people to have greater expectations in the development of the subway.
# ::tok Meanwhile , the crowded highways and complicated road traffic cause people to have greater expectations in the development of the subway .
(c / cause-01~e.9 
      :ARG0 (a / and~e.5 
            :op1 (h / highway~e.4 
                  :ARG1-of (c2 / crowd-01~e.3)) 
            :op2 (t / traffic~e.8 
                  :mod (r / road~e.7) 
                  :ARG1-of (c3 / complicate-01~e.6))) 
      :ARG1 (e / expect-01~e.14 
            :ARG0 (p / person~e.10) 
            :ARG1 (t2 / thing 
                  :ARG1-of (h2 / have-degree-91~e.12,13 
                        :ARG2~e.13 (g / great~e.13) 
                        :ARG3~e.13 (m / more~e.13))) 
            :topic~e.15 (d / develop-02~e.17 
                  :ARG1~e.18 (s / subway~e.20))) 
      :time (m2 / meanwhile~e.0))"""

    aligs = [{'type': 'subgraph', 'tokens': [0], 'nodes': ['1.3']},
             {'type': 'subgraph', 'tokens': [3], 'nodes': ['1.1.1.1']},
             {'type': 'subgraph', 'tokens': [4], 'nodes': ['1.1.1']},
             {'type': 'subgraph', 'tokens': [5], 'nodes': ['1.1']},
             {'type': 'subgraph', 'tokens': [6], 'nodes': ['1.1.2.2']},
             {'type': 'subgraph', 'tokens': [7], 'nodes': ['1.1.2.1']},
             {'type': 'subgraph', 'tokens': [8], 'nodes': ['1.1.2']},
             {'type': 'subgraph', 'tokens': [9], 'nodes': ['1']},
             {'type': 'subgraph', 'tokens': [10], 'nodes': ['1.2.1']},
             {'type': 'subgraph', 'tokens': [13], 'nodes': ['1.2.2.1.1', '1.2.2.1', '1.2.2.1.2', '1.2.2'], 'edges': [['1.2.2', None, '1.2.2.1'], ['1.2.2.1', None, '1.2.2.1.1'], ['1.2.2.1', None, '1.2.2.1.2']]},
             {'type': 'subgraph', 'tokens': [14], 'nodes': ['1.2']},
             {'type': 'subgraph', 'tokens': [17], 'nodes': ['1.2.3']},
             {'type': 'subgraph', 'tokens': [20], 'nodes': ['1.2.3.1']},
             {'type': 'relation', 'tokens': [0], 'edges': [['1', None, '1.3']]},
             {'type': 'relation', 'tokens': [3], 'edges': [['1.1.1', None, '1.1.1.1']]},
             {'type': 'relation', 'tokens': [5], 'edges': [['1.1', None, '1.1.1'], ['1.1', None, '1.1.2']]},
             {'type': 'relation', 'tokens': [6], 'edges': [['1.1.2', None, '1.1.2.2']]},
             {'type': 'relation', 'tokens': [7], 'edges': [['1.1.2', None, '1.1.2.1']]},
             {'type': 'relation', 'tokens': [9], 'edges': [['1', None, '1.1'], ['1', None, '1.2']]},
             {'type': 'relation', 'tokens': [14], 'edges': [['1.2', None, '1.2.1'], ['1.2', None, '1.2.2']]},
             {'type': 'relation', 'tokens': [17], 'edges': [['1.2.3', None, '1.2.3.1'], ['1.2', None, '1.2.3']]}]


    amr_reader = AMR_Reader()
    amr = amr_reader.loads(AMR, link_string=True)
    amr.words = amr.tokens
    idSNT = amr.id
    listAligs = transfo_aligs(amr, aligs)

    @MAILLON
    def F1(S):
        yield idSNT, listAligs

    chaine = F1()

    chaine = chaine >> filtrer_sous_graphes() >> filtrer_ss_grf_2()
    chaine = chaine >> iterer_graphe() >> filtrer_anaphore()
    chaine = chaine >> calculer_graphe_toks() >> traiter_graphe_toks()
    chaine = chaine >> ecrire_liste(fichier_out = "./pipo.txt")

    chaine.enchainer()

    




if __name__ == "__main__":
    #refaire_probleme()
    faire_liste_1(fichier_out="a_tej_2.txt")
