import os
import sys
import json
from tqdm import tqdm
import re

from amr_utils.amr_readers import AMR_Reader
from examiner_framefiles import EXPLICITATION_AMR #expliciter_AMR
from algebre_relationnelle import RELATION
from enchainables import MAILLON
from outils_alignement import (AMR_modif,
    GRAPHE_PHRASE, ALIGNEUR, preparer_alignements,
    compter_compos_connexes, transfo_aligs)

from strategies_syntaxe import definir_strategie


@MAILLON
def iterer_alignements(SOURCE, alignements):
    """Génération de listes idSNT, listeAlignements. L’AMR est membre de chacun des éléments de la liste d’alignements.
    """
    #print(" *** DÉBUT ***")
    for idSNT, listAlig in tqdm(alignements.items()):
        yield idSNT, listAlig

@MAILLON
def filtrer_vides(SOURCE):
    """Éliminer les amrs réduits à un sommet, sans arête.
    """
    for idSNT, listAlig in SOURCE:
        amr = listAlig[0].amr
        if len(amr.edges) == 0:
            continue
        yield idSNT, listAlig

@MAILLON
def filtrer_sous_graphes(SOURCE):
    """Éliminer les alignements à des sous-graphes non connexes.
    """
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
    """Éliminer les alignements vers les sous-graphes AMR où des mots différents sont alignés à des sommets différents du sous-graphe
    """
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
def iterer_graphe(SOURCE, nom_modele):
    """Générer des tuples idSNT, amr, G, où G est un objet de la classe GRAPHE, c’est-à-dire un objet qui intègre toutes les arêtes et les alignements au format relationnel
    """
    aligneur = ALIGNEUR(nom_modele)
    for idSNT, listAlig in SOURCE:
        amr = listAlig[0].amr
        mots_amr = amr.words
        snt = amr.metadata["snt"]

        graphe = GRAPHE_PHRASE(amr)
        
        if not snt.startswith(" "):
            snt = " " + snt
        # On ajoute un espace au début pour améliorer la tokenisation

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
    """Élimination des anaphores problématiques
    """
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
        
        graphe.anaphores_mg = graphe.REN_mg.select(lambda x: (x.type in autorises))
        graphe.anaphores_ag = graphe.REN_ag.select(lambda x: (x.type in autorises))#.proj("cible", "type", "groupe").rmdup()
        


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
    """Calcul du graphe entre les tokens du transformer
    """
    for idSNT, amr, graphe in SOURCE:
        ok = graphe.calculer_graphe_toks()
        if not ok:
            continue
        yield idSNT, amr, graphe


@MAILLON
def traiter_opN(SOURCE):
    """Transformation du graphe des AMR afin d’éliminer les relations syntaxiques du type :op_N
    """
    def decompose(rel):
        resu = re.search("^:(\D+)(\d+)$", rel)
        if resu is None:
            return False
        else:
            return resu[1]
    for idSNT, listAlig in SOURCE:
        amr = listAlig[0].amr
        gr = amr.rel_arcs
        aelim = RELATION("source", "label_s", "cible", "rel_N", "rel")
        for s,t,r in gr:
            rel = decompose(r)
            if rel and not rel == "ARG":
                #lbl = amr.nodes[amr.isi_node_mapping[s]]
                lbl = amr.nodes[s]
                if lbl == "and":
                    pass
                aelim.add((s, lbl, t, r, rel))
        
        
        if len(aelim) > 0:
            sommets_pivot = aelim.compter("source", "label_s", "rel")
            for src, lbl, rel, cpt in sommets_pivot:
                clef = "%s_%s_%s"%(lbl, rel, ("N" if (cpt > 1) else "1"))
                (elim_ascen, elim_syntax, elim_descen,
                 distr_parents, distr_enfants,
                 modif_syntax, conj) = definir_strategie(clef)
                if (all(b == False for b in (elim_ascen, elim_syntax, elim_descen,
                            distr_parents, distr_enfants,
                            modif_syntax, conj))):
                    continue
                R = gr.s(lambda x: x.source == src)
                syntax = RELATION("pivot", "cible", "rel_N", "rel")
                descendants = RELATION("pivot", "cible", "rel")
                for s,t,r in R:
                    rel = decompose(r)
                    if rel:
                        syntax.add((s,t,r,rel))
                    else:
                        descendants.add((s,t,r))
                parents = gr.s(lambda x: x.cible == src).ren("source", "pivot", "rel")
                trp_a_elim = RELATION("source", "cible", "relation")
                trp_a_ajou = RELATION("source", "cible", "relation")
                if elim_ascen:
                    trp_a_elim.add(*[(s,t,r) for s,t,r in parents])
                if elim_syntax:
                    trp_a_elim.add(*[(s,t,r) for s,t,r,_ in syntax])
                if elim_descen:
                    trp_a_elim.add(*[(s,t,r) for s,t,r in descendants])
                if distr_parents:
                    R = (parents * (syntax.p("pivot", "cible"))).p("source", "cible", "rel")
                    trp_a_ajou.add(*[(s,t,r) for s,t,r in R])
                if distr_enfants:
                    R = syntax.p("pivot", "cible").ren("pivot", "source")
                    R = (R * descendants).p("source", "cible", "rel")
                    trp_a_ajou.add(*[(s,t,r) for s,t,r in R])
                if modif_syntax:
                    if type(modif_syntax) is dict:
                        syn = modif_syntax["syn"]
                        reverse = modif_syntax["reverse"]
                    else:
                        syn = modif_syntax
                        reverse = False
                    if reverse:
                        trp_a_ajou.add(*[(t,s, syn) for s,t,_,_ in syntax])
                    else:
                        trp_a_ajou.add(*[(s,t, syn) for s,t,_,_ in syntax])
                if conj:
                    R = syntax.p("pivot", "cible")
                    R = (R.ren("pivot", "m1")) * (R.ren("pivot", "m2"))
                    R = (R.p("m1","m2")).s(lambda x: x.m1 != x.m2)
                    trp_a_ajou.add(*[(m1,m2,conj) for m1,m2 in R])

                gr = gr.s(lambda x: not x in trp_a_elim)
                gr = gr.add(*trp_a_ajou)
                amr.rel_arcs = gr

        yield idSNT, listAlig



@MAILLON
def ecrire_liste(SOURCE, fichier_out = "./AMR_et_graphes_phrases_2.txt", model_name=None): #, explicit_arg=False):
    """Création du fichier final
    """
    NgraphesEcrits = 0
    limNgraphe = -1
    with open(fichier_out, "w", encoding="UTF-8") as FF:
        if model_name is not None:
            print("# ::model_name %s"%model_name, file=FF)
        for idSNT, amr, graphe in SOURCE:
            #print(idSNT)
            print(amr.amr_to_string(), file=FF)
            jsn = graphe.jsonifier() #explicit_arg)
            print(jsn, file=FF)
            print(file=FF)
            NgraphesEcrits += 1
            if limNgraphe > 0 and NgraphesEcrits > limNgraphe:
                break


def construire_graphes(fichier_out="a_tej.txt"):

    explicit_arg = True
    kwargs = dict()
    nom_modele = "roberta-base"
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
    kwargs["fichiers_amr"] = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)] #[:1]
    kwargs["fichiers_snt"] = [os.path.abspath(os.path.join(snt_rep, f)) for f in os.listdir(snt_rep)] #[:1]
    

    alignements = preparer_alignements(explicit_arg=explicit_arg, **kwargs)
    chaine = iterer_alignements(alignements) >> filtrer_vides()
    chaine = chaine >> filtrer_sous_graphes() >> filtrer_ss_grf_2()
    chaine = chaine >> traiter_opN()
    chaine = chaine >> iterer_graphe(nom_modele) >> filtrer_anaphore()
    chaine = chaine >> calculer_graphe_toks()
    chaine = chaine >> ecrire_liste(fichier_out = fichier_out, model_name=nom_modele) #, explicit_arg=explicit_arg)
    print("\n%s\n"%chaine.docu)
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


    AMR = """# ::id bolt12_632_6428.3 ::amr-annotator SDL-AMR-09 ::preferred 
# ::alignments 0-1 1-1.1.1.3 4-1.1.1.4 5-1.1.1 6-1.1.1.3 10-1.1.1.2.1 11-1.1.1.2.1.1 12-1.1.1.2.1.1.1.r 13-1.1.1.2.1.1.1.1 14-1.1.1.2.1.1.1 15-1.1.1.2.2 16-1.1.1.2 17-1.1.1.2.3 17-1.1.1.2.3.r
# ::snt So we have to frequently ask ourselves: is the law enacted by this country really useful?
# ::tok So we have to frequently ask ourselves : is the law enacted by this country really useful ?
(c2 / cause-01~e.0 
      :ARG1 (o / obligate-01 
            :ARG2 (a / ask-01~e.5 
                  :ARG0 w 
                  :ARG1 (u / useful-05~e.16 
                        :ARG1 (l / law~e.10 
                              :ARG1-of (e / enact-01~e.11 
                                    :ARG0~e.12 (c / country~e.14 
                                          :mod (t / this~e.13)))) 
                        :ARG1-of (r / real-04~e.15) 
                        :polarity~e.17 (a2 / amr-unknown~e.17)) 
                  :ARG2 (w / we~e.1,6) 
                  :ARG1-of (f / frequent-02~e.4))))"""
    
    aligs = [{'type': 'subgraph', 'tokens': [0], 'nodes': ['1']},
{'type': 'subgraph', 'tokens': [1], 'nodes': ['1.1.1.3']},
{'type': 'subgraph', 'tokens': [2, 3], 'nodes': ['1.1']},
{'type': 'subgraph', 'tokens': [4], 'nodes': ['1.1.1.4']},
{'type': 'subgraph', 'tokens': [5], 'nodes': ['1.1.1']},
{'type': 'subgraph', 'tokens': [10], 'nodes': ['1.1.1.2.1']},
{'type': 'subgraph', 'tokens': [11], 'nodes': ['1.1.1.2.1.1']},
{'type': 'subgraph', 'tokens': [13], 'nodes': ['1.1.1.2.1.1.1.1']},
{'type': 'subgraph', 'tokens': [14], 'nodes': ['1.1.1.2.1.1.1']},
{'type': 'subgraph', 'tokens': [15], 'nodes': ['1.1.1.2.2']},
{'type': 'subgraph', 'tokens': [16], 'nodes': ['1.1.1.2']},
{'type': 'subgraph', 'tokens': [17], 'nodes': ['1.1.1.2.3']},
{'type': 'reentrancy:primary', 'tokens': [5], 'edges': [['1.1.1', ':ARG0', '1.1.1.3']]},
{'type': 'reentrancy:coref', 'tokens': [6], 'edges': [['1.1.1', ':ARG2', '1.1.1.3']]},
{'type': 'relation', 'tokens': [0], 'edges': [['1', None, '1.1']]},
{'type': 'relation', 'tokens': [2, 3], 'edges': [['1.1', None, '1.1.1']]},
{'type': 'relation', 'tokens': [4], 'edges': [['1.1.1', None, '1.1.1.4']]},
{'type': 'relation', 'tokens': [5], 'edges': [['1.1.1', ':ARG0', '1.1.1.3'], ['1.1.1', None, '1.1.1.2'], ['1.1.1', ':ARG2', '1.1.1.3']]},
{'type': 'relation', 'tokens': [11], 'edges': [['1.1.1.2.1', None, '1.1.1.2.1.1'], ['1.1.1.2.1.1', None, '1.1.1.2.1.1.1']]},
{'type': 'relation', 'tokens': [13], 'edges': [['1.1.1.2.1.1.1', None, '1.1.1.2.1.1.1.1']]},
{'type': 'relation', 'tokens': [15], 'edges': [['1.1.1.2', None, '1.1.1.2.2']]},
{'type': 'relation', 'tokens': [16], 'edges': [['1.1.1.2', None, '1.1.1.2.1']]},
{'type': 'relation', 'tokens': [17], 'edges': [['1.1.1.2', None, '1.1.1.2.3']]}]

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


    AMR = """# ::id nw.chtb_0325.10 ::date 2012-12-29T22:53:20 ::annotator SDL-AMR-09 ::preferred
# ::snt ( End )
# ::save-date Sat Dec 29, 2012 ::file nw_chtb_0325_10.txt
(e / end-01)"""

    aligs = [{'type': 'subgraph', 'tokens': [1], 'nodes': ['1']}]


    amr_reader = AMR_Reader()
    Explicit = EXPLICITATION_AMR()
    Explicit.dicFrames, Explicit.dicAMRadj = EXPLICITATION_AMR.transfo_pb2va_tsv()
    amr = amr_reader.loads(AMR, link_string=True)
    amr = AMR_modif(amr)
    amr.words = amr.tokens
    amr = Explicit.expliciter_AMR(amr)
    
    idSNT = amr.id
    listAligs = transfo_aligs(amr, aligs, Explicit)

    @MAILLON
    def F1(S):
        yield idSNT, listAligs

    chaine = F1()

    chaine = chaine >> filtrer_sous_graphes() >> filtrer_ss_grf_2()
    chaine = chaine >> traiter_opN()
    chaine = chaine >> iterer_graphe("roberta-base") >> filtrer_anaphore()
    chaine = chaine >> calculer_graphe_toks()
    chaine = chaine >> ecrire_liste(fichier_out = "./pipo.txt")

    chaine.enchainer()

    




if __name__ == "__main__":
    #refaire_probleme()
    construire_graphes(fichier_out="./AMR_et_graphes_phrases_explct.txt")
