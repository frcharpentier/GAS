import json
import numpy as np
from alig.examiner_framefiles import EXPLICITATION_AMR
import matplotlib.pyplot as plt
from outils.report_generator import HTML_REPORT, HTML_IMAGE
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
from calcul_scores import calcul_scores
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
import fire






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
    
    
    listeG = [AMR_modif(G) for G in load_AMRs(amr_reader, enumerateur, remove_wiki=True, link_string=True) if not G.id in doublons] #Élimination des doublons
    # listeG est une liste remplie d’objets AMR_modif (classe dérivée de la classe AMR).
    if explicit_arg:
        listeG = [Explicit.expliciter_AMR(G) for G in listeG]
    amr_liste.extend(listeG)
    for amr in listeG:
        amrid = amr.id

        amr_dict[amrid] = amr
        
        assert "snt" in amr.metadata

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

class graphe_de_tokens:
    AUTRE = "{OTHER}"
    def __init__(self):
        self.dico = dict()

    def __setitem__(self, key, value):
        assert type(key) is tuple and len(key) == 2
        i, j = key
        assert type(i) is int
        assert type(j) is int
        if i < j:
            i,j = j,i
            direction = -1
        else:
            direction = 1
        if value.startswith("{") and value.endswith("}"):
            direction = 0
        if not (i,j) in self.dico:
            self.dico[(i,j)] = (value, direction)
        else:
            val, dir = self.dico[(i,j)]
            if not (val == value and dir == direction):
                self.dico[(i,j)] = (graphe_de_tokens.AUTRE, 0)

    def __getitem__(self, key):
        if key in self.dico:
            return self.dico[key]
        else:
            return (graphe_de_tokens.AUTRE, 0)
        
def json_to_token_graph(dicjsn,liste_rel, alias):
    assert type(dicjsn) is dict
    sommets = dicjsn["sommets"]
    assert type(sommets) == list and all((type(s) == int) for s in sommets)
    aretes = dicjsn["aretes"]
    assert type(aretes) == list
    assert all((type(a) is list) for a in aretes)
    assert all((len(a) == 3) for a in aretes)
    graphe = graphe_de_tokens()
    dico_filtre = {k : a for (l,a) in zip(liste_rel, alias) for k in l}
    for a in aretes:
        i, j, rel = a[0], a[2], a[1]
        assert type(i) == type(j) == int
        assert type(rel) == str
        i, j = sommets[i], sommets[j]
        if rel.startswith(":") and rel.endswith(")") and len(rel)>=4 and rel[-3] == "(" and rel[-2] in "0123456789":
            rel = rel[:-3]
        if rel in dico_filtre:
            graphe[i,j] = dico_filtre[rel]
    return graphe



def comparer_deux_graphes(ground, pred, confusion=None):
    if confusion is None:
        confusion = dict()
    else:
        assert type(confusion) == dict
    aretes = set( k for k in ground.dico.keys())
    aretes = aretes.union(set( k for k in pred.dico.keys()))
    for a in aretes:
        clef = "G= %s ; P= %s"%(ground[a][0], pred[a][0])
        confusion[clef] = 1+ confusion.get(clef, 0)
    return confusion


def enum_paires_graphes(fichier_ground, fichier_pred):
    def lecture_un_fichier(fich):
        with open(fich, "r", encoding="utf-8") as F:
            etat = 0
            derniere_ligne = ""
            for ligne in F:
                ligne = ligne.rstrip()
                if etat == 0:
                    if ligne.startswith("# ::id "):
                        ids = ligne.split(maxsplit=3)[2]
                        etat = 1
                elif etat == 1:
                    if len(ligne.strip()) == 0:
                        etat = 0
                        if derniere_ligne.startswith('{"tokens":') and derniere_ligne.endswith("}"):
                            yield ids, derniere_ligne
                            ids = None
                derniere_ligne = ligne
        if ids and derniere_ligne.startswith('{"tokens":') and derniere_ligne.endswith("}"):
            yield ids, derniere_ligne

    idsGROUND = [ids for ids, lig in lecture_un_fichier(fichier_ground)]
    idsINTER = [ids for ids,lig in lecture_un_fichier(fichier_pred) if ids in idsGROUND]

    genGROUND = lecture_un_fichier(fichier_ground)
    genPRED = lecture_un_fichier(fichier_pred)

    for ids in tqdm(idsINTER):
        while True:
            idsG, jsnG = next(genGROUND)
            if idsG == ids:
                break
        while True:
            idsP, jsnP = next(genPRED)
            if idsP == ids:
                break
        yield ids, jsnG, jsnP

def evaluer_SPRING(fichierGROUND, fichierPRED, nom_rapport, nb_classes = 15, equilibrage = True):
    confusion = dict()

    assert nb_classes in [15,21]

    liste_rel = [[':>AGENT'], [':beneficiary', ':>BENEFICIARY'],
                   [':>CAUSE'], [':condition', ':>CONDITION'], [':degree'],
                   [':>EXPERIENCER'], [':location', ':>LOCATION'],
                   [':manner', ':>MANNER'], [':mod', ':>MOD'],
                   [':>PATIENT'], [':poss'], [':purpose', ':>PURPOSE'],
                   [':>THEME'], [':time', ':>TIME'], [':topic', ':>TOPIC'],
                   ["{and_or}"],["{and}"],["{groupe}"],["{idem}"],["{inter}"],["{or}"]
                ]
    
    alias = [':>AGENT', ':>BENEFICIARY', ':>CAUSE', ':>CONDITION',
                 ':degree', ':>EXPERIENCER', ':>LOCATION', ':>MANNER',
                 ':>MOD', ':>PATIENT', ':poss', ':>PURPOSE',
                 ':>THEME', ':>TIME', ':>TOPIC',
                 "{and_or}", "{and}", "{groupe}", "{idem}", "{inter}", "{or}"
                ]
    
    if nb_classes == 15:
        liste_rel = liste_rel[:15]
        alias = alias[:15]

    for ids, jsnG, jsnP in enum_paires_graphes(fichierGROUND, fichierPRED):
        jsnG = json.loads(jsnG)
        jsnP = json.loads(jsnP)
        ground = json_to_token_graph(jsnG, liste_rel, alias)
        pred   = json_to_token_graph(jsnP, liste_rel, alias)
        confusion = comparer_deux_graphes(ground, pred, confusion)

    alias.append(graphe_de_tokens.AUTRE)
    conf = []
    for G in alias:
        ligneConf = []
        conf.append(ligneConf)
        for P in alias:
            clef = clef = "G= %s ; P= %s"%(G, P)
            ligneConf.append(confusion.get(clef, 0))


    # ÉQUILIBRAGE
    if equilibrage:
        conf = [lig[:-1] for lig in conf[:-1]]
        alias = alias[:-1]

    scores = calcul_scores(conf, nb_classes)
    acc, bal_acc = scores[nb_classes]
    
    #print("ACC : %f, BAL_ACC : %f"%(acc, bal_acc))
    #with open(nom_rapport, "w", encoding="utf-8") as F:
    #    print("ACC : %f, BAL_ACC : %f"%(acc, bal_acc), file=F)
    #    print("confusion = [", file = F)
    #    for ligConf in conf:
    #        print("%s,"%repr(ligConf), file=F)
    #    print("]", file=F)

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Performances modèle Spring", 1)
        R.titre(("NB classes : %d" % nb_classes), 2)
        R.titre(("Exactitude : %f, exactitude équilibrée : %f"%(acc, bal_acc)), 2)

        with R.new_img_with_format("svg") as IMG:
            confnp = np.array(conf)
            sum = confnp.sum(axis=1)
            sum = sum.reshape((confnp.shape[0], 1))
            confnp = confnp / sum
            confDisp = ConfusionMatrixDisplay(confnp, display_labels = alias)
            NN, _ = confnp.shape
            if NN < 7:
                NN = 7
            fig, ax = plt.subplots(figsize=(NN,NN))
            confDisp.plot(xticks_rotation="vertical", ax=ax)
            #pour calculer disp.figure_, qui est une figure matplotlib
            fig = confDisp.figure_
            fig.savefig(IMG.fullname)
        matrix = repr(conf)
        R.texte_copiable(matrix, hidden=True, buttonText="Copier la matrice de confusion")
        R.ligne()



    
    
        



        

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
    #make_SPRING_alig_file()
    #----------------------
    #ground = "./alig_AMR_spring_test.txt"
    #pred = " "
    #resu = "./Experiment_Results/resu_SPRING.txt"
    #evaluer_SPRING(ground, pred, resu)
    #----------------------
    fire.Fire()