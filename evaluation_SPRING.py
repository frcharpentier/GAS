import json
from alig.examiner_framefiles import EXPLICITATION_AMR
from amr_utils.amr_readers import AMR_Reader, AMR
from alig.outils_alignement import quick_read_amr_file, load_aligs_from_json, AMR_modif
from alig.fabrication_listes import (
    iterer_alignements,
    filtrer_vides,
    filtrer_sous_graphes,
    filtrer_ss_grf_2,
    traiter_opN,
    iterer_graphe,
    filtrer_anaphore,
    calculer_graphe_toks,
    ecrire_liste
)






def enum_AMRS(fichier_amr, fichier_tokenisation):
    with open(fichier_tokenisation, "r", encoding="utf-8") as F:
        snt_dict = json.load(F)

    cumul = ""
    etat = 0
    with open(fichier_amr, "r", encoding="utf-8") as F:
        for ligne in F:
            ligne = ligne.rstrip()
            if etat == 0:
                if ligne.startswith("# ::id "):
                    etat = 1
                    ids = ligne[7:]
                    dico = snt_dict[ids]
                    if "tok" in dico and (dico["tok"] is not None):
                        ligne += "\n"
                        ligne += "# ::tok %s"%(dico["tok"])
                    else:
                        etat = 0
                    cumul = ligne + "\n"
            elif etat == 1:
                cumul += ligne + "\n"
                if len(ligne.strip()) == 0:
                    while cumul.endswith("\n"):
                        cumul = cumul[:-1]
                    if len(cumul) > 0:
                        yield cumul
                        cumul = ""
                    etat = 0
    while cumul.endswith("\n"):
        cumul = cumul[:-1]
    if len(cumul) > 0:
        yield cumul
            
                
    


def load_AMRs(amr_reader, enumerateur, remove_wiki=False, output_alignments=False, link_string=False):
        amrs = []
        alignments = {}
        amr_idx = 0

        for sent in enumerateur:

            no_tokens = False
            if sent.strip().startswith('('):
                no_tokens = True

            resu = amr_reader.loads(sent, remove_wiki, output_alignments, no_tokens, link_string)
            if not resu:
                continue
            if output_alignments:
                amr, aligns = resu[0], resu[1]
            else:
                amr = resu
            if amr.id == "":
                amr.id = str(amr_idx)

            amrs.append(amr)
            if output_alignments:
                alignments[amr.id] = aligns
            amr_idx += 1

        if output_alignments:
            return amrs, alignments
        return amrs


def preparer_alignements_SPRING(fichier_amr, fichier_tokenisation, leamr_subg, leamr_rel, leamr_reent, explicit_arg=True, doublons=[]):
    #explicit_arg, si VRAI, transformera tous les rôles ARGn en une description sémantique
    #plus explicite. (Sans doute plus facile à classer, également.)

    #fichiers_snt = [fichier_amr]
    #fichiers_amr = [fichier_amr]
    
    #fichier_sous_graphes = leamr_subg
    #fichier_reentrances = leamr_reent
    #fichier_relations = leamr_rel



    amr_reader = AMR_Reader()
    enumerateur = enum_AMRS(fichier_amr, fichier_tokenisation)
    
    if explicit_arg:
        Explicit = EXPLICITATION_AMR()
        Explicit.dicFrames, Explicit.dicAMRadj = EXPLICITATION_AMR.transfo_pb2va_tsv()

    amr_liste = []
    # amr_liste sera une liste remplie d’objets AMR
    amr_dict = dict()
    # amr_dict sera un dico dont les clés seront les identifiants de phrases,
    # et dont les valeurs seront les objets AMR correspondants
    
    
    #print(amrfile)
    listeG = [AMR_modif(G) for G in load_AMRs(amr_reader, enumerateur, remove_wiki=True, link_string=True) if not G.id in doublons] #Élimination des doublons
    # listeG est une liste remplie d’objets AMR_modif (classe dérivée de la classe AMR).
    if explicit_arg:
        listeG = [Explicit.expliciter_AMR(G) for G in listeG]
    amr_liste.extend(listeG)
    for amr in listeG:
        amrid = amr.id

        amr_dict[amrid] = amr
        
        #if not "snt" in amr.metadata:
        #    if amrid in snt_dict:
        #        amr.metadata["snt"] = snt_dict[amrid]
        #    else:
        #        #toks = graphe.tokens
        #        toks = amr.words
        #        amr.metadata["snt"] = " ".join(toks)
        assert "snt" in amr.metadata

    


    #monamr = amr_dict["DF-199-192821-670_2956.4"]

    print("%d graphes AMR au total."%len(amr_liste))
    if explicit_arg:
        alignements = load_aligs_from_json([
            leamr_subg,
            leamr_rel,
            leamr_reent],
            amr_liste,
            Explicit)
    else:
        alignements = load_aligs_from_json([
            leamr_subg,
            leamr_rel,
            leamr_reent],
            amr_liste)
    # alignements est un dico dont les clés sont les identifiants de phrase
    # et dont les valeurs sont des listes de dico d’alignements
    # un dico d’alignements est un dico avec les clés type,
    # tokens(càd mots), nodes, et edges

    return alignements



def essai():
    fichier_amr = "/home/frederic/projets/AMR_Martinez/sortie/amrs_SPRING_test.txt"
    fichier_tokenisation = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.tokenisation_alignments.json"

    for i, chaine in enumerate(enum_AMRS(fichier_amr, fichier_tokenisation)):
        print(chaine)
        print("###################")
        print()
        if i > 5:
            break

def essai2():
    fichier_amr = "/home/frederic/projets/AMR_Martinez/sortie/amrs_SPRING_test.txt"
    fichier_tokenisation = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.tokenisation_alignments.json"
    leamr_subg = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.subgraph_alignments.json"
    leamr_rel = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.relation_alignments.json"
    leamr_reent = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.reentrancy_alignments.json"

    alignements = preparer_alignements_SPRING(
        fichier_amr,
        fichier_tokenisation,
        leamr_subg,
        leamr_rel,
        leamr_reent,
    )


def make_SPRING_alig_file(
            nom_modele = "facebook/bart-large",
            fichier_out = "/home/frederic/projets/detection_aretes/alig_AMR_spring_pred_test.txt",
            fichier_amr = "/home/frederic/projets/AMR_Martinez/sortie/amrs_SPRING_test.txt",
            fichier_tokenisation = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.tokenisation_alignments.json",
            leamr_subg = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.subgraph_alignments.json",
            leamr_rel = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.relation_alignments.json",
            leamr_reent = "/home/frederic/projets/AMR_Martinez/sortie/spring_test.reentrancy_alignments.json"):

    alignements = preparer_alignements_SPRING(
        fichier_amr,
        fichier_tokenisation,
        leamr_subg,
        leamr_rel,
        leamr_reent,
    )
    chaine = iterer_alignements(alignements) >> filtrer_vides()
    chaine = chaine >> filtrer_sous_graphes() >> filtrer_ss_grf_2()
    chaine = chaine >> traiter_opN()
    chaine = chaine >> iterer_graphe(nom_modele) >> filtrer_anaphore()
    chaine = chaine >> calculer_graphe_toks()
    chaine = chaine >> ecrire_liste(fichier_out = fichier_out, model_name=nom_modele) #, explicit_arg=explicit_arg)
    print("\n%s\n"%chaine.docu)
    chaine.enchainer()



if __name__ == "__main__":
    #essai()
    #essai2()
    make_SPRING_alig_file()