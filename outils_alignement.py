from transformers import AutoTokenizer
from amr_utils.amr_readers import AMR_Reader
from amr_utils.alignments import AMR_Alignment
from algebre_relationnelle import RELATION
import os
import sys
import json
import numpy as np
import tqdm


def alig_to_string(self):
    resu = dict()
    resu["type"] = self.type
    resu["tokens"] = self.tokens
    resu["nodes"] = self.nodes
    resu["edges"] = self.edges
    return repr(resu)

AMR_Alignment.__str__ = alig_to_string
AMR_Alignment.__repr__ = alig_to_string

class ALIGNEUR:
    def __init__(self, nom_modele, ch_debut = "", ch_suite = "##", tok1 = None, tokn = None):
        # ch_debut est la chaine qui préfixe les tokens qui constituent le début d’un mot.
        # généralement, il s’agit d’une chaine vide
        # ch_suite est la chaine qui préfixe les tokens qui consituent la suite d’un mot
        # déjà commencé. Pour BERT, il s’agit de "##". Pour RoBERTa, il s’agit de "Ġ" (bizarre)
        self.nom_modele = nom_modele
        self.ch_debut = ch_debut
        self.ch_suite = ch_suite
        self.tokenizer = AutoTokenizer.from_pretrained(nom_modele)
        if tok1 == None:
            self.tok1 = self.tokenizer.cls_token
        else:
            self.tok1 = tok1
        if tokn == None:
            self.tokn = self.tokenizer.sep_token
        else:
            self.tokn = tokn
   
        
    def aligner_seq(self, toksH, phrase):
        # Chercher le parcours de coût minimal
        # pour traverser une grille du coin supérieur gauche
        # au coin inférieur droit, d’intersection en intersection.
        # les intervalles entre deux intersections sont indexés
        # verticalement et horizontalement
        # par les tokens toksV et toksH.
        # À chaque intersection, on a le droit de se déplacer
        # vers la droite ou vers le bas, pour un coût qui vaut 1.
        # Si les tokens vertical et horizontal de l’intervalle
        # à traverser sont identiques, on peut se déplacer en
        # diagonale (vers le bas et la droite) pour un coût nul.
        # Il s’agit de l’algo de Needleman-Wunsch, que je considère
        # comme un cas particulier de l’algo A*.
        #
        # toksH contient les tokens de l’AMR
        # phrase contient la phrase à faire tokeniser

        toksV = toks_transformer = [self.tokenizer.decode(x).strip().lower() for x in self.tokenizer(phrase).input_ids]
        assert toksV[0] == self.tok1
        assert toksV[-1] == self.tokn
        toksV = toksV[1:-1] #Éliminons les tokens CLS et SEP. (On les remettra par la suite.)
        
        visites = dict()
        front = {(0,0): (0, (0,0))}
        nV = len(toksV)
        nH = len(toksH)
        
        estim = lambda x: abs(nV-x[0] - nH + x[1])
        clef = lambda x : x[1][0] + estim(x[0])

        while True:
            choix = min(front.items(), key=clef)
            
            (posV, posH), (cout, mvt) = choix
            if (posV, posH) == (nV, nH):
                break
            # On va faire évoluer ce cheminement, et
            # considérer tous les cheminements possibles en 
            #ajoutant à chaque fois un déplacement élémentaire.
            del front[(posV, posH)]
            visites[(posV, posH)] = mvt
            if posV < nV and posH < nH and len(toksV[posV]) <= len(toksH[posH]):
                H = toksH[posH].lower()
                mvt0 = 0
                V = toksV[posV + mvt0].lower()
                vf = True
                while H.startswith(V) and posV+mvt0+1 < nV:
                    mvt0 += 1
                    H = H[len(V):]
                    V = toksV[posV + mvt0].lower()
                if len(H) == 0:
                    #possibilité de déplacement en diagonale
                    posV2, posH2 = posV+mvt0, posH+1
                    if not (posV2, posH2) in visites:
                        if (posV2, posH2) in front:
                            cout0, mvt0 = front[(posV2, posH2)]
                            if cout0 > cout+0:
                                front[(posV2, posH2)] = (cout, (1,mvt0))
                        else:
                            front[(posV2, posH2)] = (cout, (1,mvt0))
            if posV < nV:
                #possibilité de déplacement vertical
                posV2, posH2 = posV+1, posH
                if not (posV2, posH2) in visites:
                    if (posV2, posH2) in front:
                        cout0, mvt0 = front[(posV2, posH2)]
                        if cout0 > cout+1:
                            front[(posV2, posH2)] = (cout+1, (0,1))
                    else:
                        front[(posV2, posH2)] = (cout+1, (0,1))
            if posH < nH:
                #possibilité de déplacement horizontal
                posV2, posH2 = posV, posH+1
                if not (posV2, posH2) in visites:
                    if (posV2, posH2) in front:
                        cout0, mvt0 = front[(posV2, posH2)]
                        if cout0 > cout+0:
                            front[(posV2, posH2)] = (cout+1, (1,0))
                    else:
                        front[(posV2, posH2)] = (cout+1, (1,0))
        
        chem = []
        while (posV, posH) != (0,0):
            chem.append(mvt)
            posH, posV = posH - mvt[0], posV-mvt[1]
            mvt = visites[(posV, posH)]
        chem = chem[::-1]

        # On connaît le cheminement optimal dans la grille.
        # déduisons-en un alignement de la chaine toksH vers la chaine toksV
        # On représentera cet alignement sous forme d’une relation de sorte("token", "mot")
        # Les tokens et les mots seront représentés par leur numéro d’ordre.
        resu = RELATION(("token", "mot"))
        
        #resu.add_1((0, -1)) #Le token [CLS] ne correspond à aucun mot

        i = 1 #Numéro de token
        j = 0 #Numéro de mot
        cumulH = 0
        cumulV = 0
        
        for H, V in chem:
            if H==0 or V==0:
                #Accumulation
                cumulH += H
                cumulV += V
            else:
                if cumulH > 0 or cumulV > 0:
                    if cumulH == cumulV:
                        #distribuons un token pour un mot
                        resu.add_n([(i+k, j+k) for k in range(cumulV)])
                    elif cumulH > 0:
                        #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
                        for l in range(cumulH):
                            resu.add_n([(i+k, j+l) for k in range(cumulV)])
                    else:
                        #alignons l’ensemble des tokens sautés sur le vide.
                        resu.add_n([(i+k, -1) for k in range(cumulV)])
                    i += cumulV
                    j += cumulH
                    cumulH = 0
                    cumulV = 0
                assert H == 1
                resu.add_n([(i+k, j) for k in range(V)])
                i += V
                j += 1
        if cumulH > 0 or cumulV > 0:
            if cumulH == cumulV:
                #distribuons un token pour un mot
                resu.add_n([(i+k, j+k) for k in range(cumulV)])
            elif cumulH > 0:
                #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
                for l in range(cumulH):
                    resu.add_n([(i+k, j+l) for k in range(cumulV)])
            else:
                #alignons l’ensemble des tokens sautés sur le vide.
                resu.add_n([(i+k, -1) for k in range(cumulV)])
            i += cumulV
            j += cumulH
            cumulH = 0
            cumulV = 0

        #resu.add_1((i, -1)) #Le dernier token [SEP] ne correspond à aucun mot.

        toksV = (self.tok1,) + tuple(toksV) + (self.tokn,)
        return resu, list(toksV)

        

    

def test_fichier_reentrance(fichier):
    with open(fichier, "r", encoding="UTF-8") as F:
        jsn = json.load(F)
    for phid, aligs in jsn.items():
        assert type(aligs) is list
        dicaligs = dict()
        for alig in aligs:
            aretes = alig["edges"]
            assert len(aretes) == 1
            arete = aretes[0]
            assert len(arete) == 3
            s, ed, c = arete
            if not c in dicaligs:
                dicaligs[c] = [alig["type"]]
            else:
                dicaligs[c].append(alig["type"])
        for c, lste in dicaligs.items():
            nprim = 0
            ntot = 0
            for t in lste:
                if t == "reentrancy:primary":
                    nprim += 1
                ntot += 1
            if ntot < 1:
                print("ntot < 1")
                print(json.dumps(aligs))
            assert nprim == 1


def dresser_liste_doublons(amr_rep):
    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]

    reader = AMR_Reader()
    amr_liste = []
    amr_dict = dict()
    doublons = dict()

    for amrfile in fichiers_amr:
        #print(amrfile)
        listeG = reader.load(amrfile)
        amr_liste.extend([G.id for G in listeG])
        for graphe in listeG:
            gid = graphe.id
            if gid in amr_dict:
                if gid in doublons:
                    doublons[gid] = doublons[gid]+1
                else:
                    doublons[gid] = 2
            else:
                amr_dict[gid] = graphe

    return doublons


def load_aligs_from_json(json_files, amrs=None):
    if amrs:
        amrs = {amr.id:amr for amr in amrs}
    else:
        raise Exception('To un-anonymize alignments, the parameter "amrs" is required.')
    alignments = dict()
    type_aligs = set()
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf8') as f:
            aligs = json.load(f)
        for k, v in aligs.items():
            assert type(v) is list
            if not k in alignments:
                alignments[k] = v
            else:
                alignments[k].extend(v)

    alignments = {k:v for k,v in alignments.items() if not k.startswith("lpp_1943.")}
    # On élimine les phrases qui viennent du Petit Prince.
    ids = [k for k in alignments]
    for k in ids:
        elimine = False
        if not k in amrs:
            print('Failed to un-anonymize: no matching AMR:', k)
            del alignments[k]
            continue
        amr = amrs[k]
        egal = lambda x,y: (amr.isi_node_mapping[x] == amr.isi_node_mapping[y])
        aligs = []
        for a in alignments[k]:
            type_aligs.add(a["type"])
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
                        print('Failed to un-anonymize:', amr.id, e, file=sys.stderr)
                        elimine = True
                        break #sortir de for i,e
                    else:
                        new_e = new_e[0]
                        a['edges'][i] = [s, new_e[1], t]
            if elimine:
                break #sortir de for a
            alig = AMR_Alignment(a['type'], a['tokens'], a['nodes'], [tuple(e) for e in a['edges']] if "edges" in a else None)
            alig.amr = amr
            aligs.append(alig)
        if elimine:
            del alignments[k]
        else:
            alignments[k] = aligs
    
    print("Il y a des alignements de %d types différents"%len(type_aligs))
    print("Voici la liste :")
    print(type_aligs)
    print("-----")
    return alignments

class AMRSuivant(Exception):
    pass

class DICO_ENUM:
    def __init__(self, init=[]):
        self.liste = init
        self.dico = {k : i for i, k in enumerate(self.liste)}
        self.idx = len(self.liste)

    def __getitem__(self, idx):
        if type(idx) is int:
            if idx < self.idx:
                return self.liste[idx]
            else:
                return None
        else:
            assert type(idx) is str
            if idx in self.dico:
                return self.dico[idx]
            else:
                self.liste.append(idx)
                self.dico[idx] = self.idx
                self.idx += 1
                return (self.idx -1)
            
#DICO_rel = DICO_ENUM(["{nihil}", "{solidus}", "{quaedam}", "{idem}"])

class GRAPHE_PHRASE:
    def __init__(self, amr):
        self.amr = amr
        self.id = amr.id
        self.mots = amr.tokens #À changer ultérieurement
        self.tokens = amr.tokens #À changer ultérieurement
        #self.setSommets = set()
        #self.setTokens = set()

        self.SG_mg = None
        self.SG_sg = None
        self.DSG_mg = None
        self.DSG_sg = None
        self.REL_mg = None
        self.REL_ag = None
        self.REN_mg = None
        self.REN_ag = None
        
        self.N = len(self.tokens)
        self.transfo_AMR()

    def transfo_AMR(self):
        sommets = RELATION("sommet", "variable", "constante")
        arcs = RELATION("source", "cible", "relation")
        arcs_redir = RELATION("source", "cible", "rel_redir")

        for k, v in self.amr.nodes.items():
            if k in self.amr.variables:
                sommets.add((self.amr.isi_node_mapping[k], v, None))
            else:
                sommets.add((self.amr.isi_node_mapping[k], None, v))

        for s, r, t in self.amr.edges:
            s = self.amr.isi_node_mapping[s]
            t = self.amr.isi_node_mapping[t]
            arcs.add((s, t, r))

        for s, r, t in self.amr.edges_redir():
            s = self.amr.isi_node_mapping[s]
            t = self.amr.isi_node_mapping[t]
            arcs_redir.add((s, t, r))

        self.amr_sommets = sommets
        self.amr_arcs = arcs
        self.amr_arcs_redir = arcs_redir

    def ajouter_aligs(self, listAlig):
        self.SG_mg = RELATION("mot", "groupe")
        self.SG_sg = RELATION("sommet", "groupe")
        self.DSG_mg = RELATION("mot", "groupe")
        self.DSG_sg = RELATION("sommet", "groupe")
        self.REL_mg = RELATION("mot", "groupe")
        self.REL_ag = RELATION("source", "cible", "relation", "groupe")
        self.REN_mg = RELATION("mot", "type", "groupe")
        self.REN_ag = RELATION("source", "cible", "relation", "type", "groupe")

        id_groupe = 0
        
        for a in listAlig:
            if a.type == "subgraph":
                self.SG_mg.add(*[(mot, "G%d"%id_groupe) for mot in a.tokens])
                self.SG_sg.add(*[(sommet, "G%d"%id_groupe) for sommet in a.nodes])
                id_groupe += 1
            elif a.type == "dupl-subgraph":
                self.DSG_mg.add(*[(mot, "G%d"%id_groupe) for mot in a.tokens])
                self.DSG_sg.add(*[(sommet, "G%d"%id_groupe) for sommet in a.nodes])
                id_groupe += 1
            elif a.type == "relation":
                self.REL_mg.add(*[(mot, "G%d"%id_groupe) for mot in a.tokens])
                self.REL_ag.add(*[(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r, "G%d"%id_groupe) for s,r,c in a.edges])
                id_groupe += 1
            elif a.type.startswith("reentrancy"):
                self.REN_mg.add(*[(mot, a.type, "G%d"%id_groupe) for mot in a.tokens])
                self.REN_ag.add(*[(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r, a.type, "G%d"%id_groupe) for s,r,c in a.edges])
                id_groupe += 1

    def verifier_reentrance(self):
        # le schéma de self.amr_arcs est ("source", "cible", "relation")
        # le schéma de self.REN_ag est ("source", "cible", "relation", "type", "groupe")
        reen = self.REN_ag.proj("source", "cible", "relation")
        liste_ = [nd for (nd,) in reen.proj("cible").enum()]
        
        arcs_nd = self.amr_arcs.select(lambda x: x.cible in liste_)
        
        if not len(reen) == len(arcs_nd):
            return False
        if any(not e in arcs_nd.table for e in reen.table):
            return False
        if any(not e in reen.table for e in arcs_nd.table):
            return False
        
        return True


    def traiter_reentrances(self):
        self.anaphores_mg = RELATION(*(self.REN_mg.sort))
        self.anaphores_ag = RELATION(*(self.REN_ag.sort))
        if len(self.REN_ag) == 0:
            return
        toks = self.tokens
        settoks = set(m for (m, _) in self.SG_mg.enum())
        toks_libres = [not t in settoks for t in range(self.N)]
        liste_toks_libres = [toks[i] if v else "" for i, v in enumerate(toks_libres)]
        autorises = ["reentrancy:repetition", "reentrancy:coref"] #, "reentrancy:primary"]
        libres2 = [True] * self.N

        #primaires = REL_ren.select(lambda x: x.type == "reentrancy:primary")
        self.anaphores_mg = self.REN_mg.select(lambda x: (x.type in autorises))
        self.anaphores_ag = self.REN_ag.select(lambda x: (x.type in autorises))#.proj("cible", "type", "groupe").rmdup()
        mots = self.anaphores_mg.proj("mot").rmdup()
        if len(mots) == 0:
            return

        for (t,) in mots.enum():
            if not toks_libres[t]:
                raise AssertionError("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
            libres2[t] = False

        for i, vf in enumerate(libres2):
            if not vf:
                toks_libres[i] = False
                liste_toks_libres[i] = ""

        groupes = self.anaphores_mg.proj("groupe").rmdup()
        ajout_mg = RELATION(*(self.REN_mg.sort))
        ajout_ag = RELATION(*(self.REN_ag.sort))

        idG = 0
        for (g,) in groupes.enum():
            sommets = self.anaphores_ag.select(lambda x: x.groupe == g).proj("cible")
            assert len(sommets) == 1
            som = list(s for (s,) in sommets.enum())[0]
            t_t = [t for (t,_,_) in self.anaphores_mg.select(lambda x: (x.groupe == g)).enum()]
            t_t.sort()
            assert all(s == 1+t_t[i] for i, s in enumerate(t_t[1:]))

            tt = [toks[t] for t in t_t]
            ph = " ".join(tt)
            if ph in [",", "person"]:
                continue
            ntt = len(tt)
            mtch = find_sub_list(liste_toks_libres, tt)
            if len(mtch) > 0:
                print("snt %s\nRépétition ou anaphore supplémentaire trouvée ! (%s)\n"%(self.id,ph))
                for t  in mtch:
                    ajout_mg.add(*[(t+k, "reentrancy:ajout", "Gaj%d"%idG) for k in range(ntt)])
                    ajout_ag.add(("_", "%s"%som, "_", "reentrancy:ajout", "Gaj%d"%idG))

                    idG += 1

        if len(ajout_ag) > 0: 
            self.anaphores_mg = self.anaphores_mg + ajout_mg
            self.anaphores_ag = self.anaphores_ag + ajout_ag
        

    def calculer_graphe_toks(self):
        #calcul du graphe de sous-graphes
        rels = self.amr_arcs_redir.ren("source", "cible", "rel")
        sgg = self.SG_sg.ren("source", "gr_s") * rels * self.SG_sg.ren("cible", "gr_c")
        sgg = sgg.p("gr_s", "gr_c", "rel").s(lambda x: x.gr_s != x.gr_c)
        N = len(sgg)
        if N > len(sgg.p("gr_s", "gr_c")):
            # Si le cardinal diminue quand on enlève le type de
            # relation sémantique, c’est qu’il y a plusieurs arêtes
            # différentes entre deux sous-graphes.
            # C’est un cas d’erreur.
            return False
        
        #calcul des relations mot-mot (pour les mots alignés au même sous-graphe)
        rel1 = self.SG_mg + self.anaphores_mg.p("mot", "groupe")
        rel2 = (rel1.ren("mot_s", "groupe") * (rel1.ren("mot_c", "groupe"))).p("mot_s", "mot_c")
        rel2 = rel2.s(lambda x: x.mot_s != x.mot_c)
        groupes = rel2 * (RELATION("rel").add(("{groupe}",)))
        
        #calcul de la relation mot - groupe (princeps)
        rel3 = self.SG_sg + (self.anaphores_ag.p("cible", "groupe").ren("sommet", "groupe"))
        rel3 = (rel1 * rel3).ren("mot", "g1", "sommet")
        rel3 = (rel3 * self.SG_sg).p("mot", "groupe")

        #calcul du graphe amr de mots :
        gr_mots = rel3.ren("mot_s", "gr_s") * sgg * rel3.ren("mot_c", "gr_c")
        gr_mots = gr_mots.p("mot_s", "mot_c", "rel")

        #calcul de la relation mot-mot pour les anaphores.
        rel4 = (rel3.ren("mot_s", "groupe") * rel3.ren("mot_c", "groupe")).p("mot_s", "mot_c")
        rel4 = rel4.s(lambda x: x.mot_s != x.mot_c)
        idem = (rel4-rel2)*(RELATION("rel").add(("{idem}", )))

        #relation finale:
        self.graphe_toks = groupes + idem + gr_mots
        return True
    
    def jsonifier(self):
        jsn = dict()
        #jsn["mots"] = self.mots
        jsn["tokens"] = self.tokens
        N = len(self.tokens)
        settoks = set(m for (m, _, _) in self.graphe_toks.enum())
        settoks = settoks.union(set(m for (_, m, _) in self.graphe_toks.enum()))
        sommets = [i for i in range(N) if i in settoks]
        NS = len(sommets)
        corresp = [None]*N
        for i, s in enumerate(sommets):
            corresp[s] = i
        
        rel1 = self.SG_mg + self.anaphores_mg.p("mot", "groupe")
        rel2 = self.SG_sg + (self.anaphores_ag.p("cible", "groupe").ren("sommet", "groupe"))
        rel = (rel1 * rel2).p("mot", "sommet")
        dico = dict()
        for (m,s) in rel.enum():
            if m in dico:
                dico[m].append(s)
            else:
                dico[m] = [s]
        for m in dico:
            dico[m].sort()

        dictok = [dico[sommets[i]] for i in range(NS)]
        aretes = []
        for (s,c,r) in self.graphe_toks.enum():
            s,c = corresp[s], corresp[c] 
            aretes.append((s,r,c))
        aretes.sort()
        jsn["sommets"] = sommets
        jsn["dicTokens"] = dictok
        jsn["aretes"] = aretes
        jsn = json.dumps(jsn)
        return jsn

def compter_compos_connexes(aretes_):
    if len(aretes_) == 0:
        return 0
    dico = dict()
    N = 0
    aretes = []
    for s, r, c in aretes_:
        if not s in dico:
            dico[s] = N
            s1 = N
            N += 1
        else:
            s1 = dico[s]
        if not c in dico:
            dico[c] = N
            c1 = N
            N += 1
        else:
            c1 = dico[c]
        aretes.append((s1, c1))

    #chaque sommet reçoit une couleur différente au départ
    couleurs = list(range(N))

    #réduction des couleurs:
    for s,c in aretes:
        if couleurs[s] != couleurs[c]:
            cs = couleurs[s]
            cc = couleurs[c]
            for i, k in enumerate(couleurs):
                if k == cc:
                    couleurs[i] = cs

    #comptage des couleurs après réduction
    N = len(set(couleurs))
    return N
        
def comparer_reentrances(amr, nd, edges):
    edges = [(amr.isi_node_mapping[s], r, amr.isi_node_mapping[t]) for s,r,t in edges]
    nd = amr.isi_node_mapping[nd]
    edgesR = [(amr.isi_node_mapping[s], r, nd) for s, r, t in amr.edges if amr.isi_node_mapping[t]==nd]
    # edgesR contient tous les arcs de l’AMR dont la cible est nd.
    # La fonction vérifie que cet ensemble d’arcs est bien celui passé en paramète.
    # (Qui est l’ensemble des arcs de réentrance.)
    if not len(edgesR) == len(edges):
        return False
    if any(not e in edges for e in edgesR):
        return False
    if any(not e in edgesR for e in edges):
        return False
    return True



def find_sub_list(liste, morceau, indice=0):
    matches = []
    n = len(morceau)
    for i in range(indice, len(liste)):
        if liste[i] == morceau[0] and liste[i:i+n] == morceau:
            matches.append(i)
    return matches

               
        

def ecrire_structure_dans_fichier(graphes, fichier):
    with open(fichier, "w", encoding="UTF-8") as F:
        for graphe in graphes:
            amr = graphe.amr
            print(amr.amr_string, file=F)
            jsn = graphe.jsonifier()
            print(jsn, file=F)
            print(file=F)




def construire_graphes():
    prefixe_alignements = "../alignement_AMR/leamr/data-release/alignments/ldc+little_prince."
    fichier_sous_graphes = prefixe_alignements + "subgraph_alignments.json"
    fichier_reentrances = prefixe_alignements + "reentrancy_alignments.json"
    fichier_relations = prefixe_alignements + "relation_alignments.json"
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    doublons = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']

    # Cette liste est une liste d’identifiants AMR en double dans le répertoire amr_rep
    # Il n’y en a que quatre. On les éliminera, c’est plus simple, ça ne représente que huit AMR.
    # Cette liste a été établie en exécutant la fonction "dresser_liste_doublons" ci-dessus.

    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    #fichiers_amr = fichiers_amr[0:1]

    reader = AMR_Reader()
    amr_liste = []
    amr_dict = dict()

    for amrfile in fichiers_amr:
        #print(amrfile)
        listeG = [G for G in reader.load(amrfile, remove_wiki=True, link_string=True) if not G.id in doublons] #Élimination des doublons
        amr_liste.extend(listeG)
        for graphe in listeG:
            amrid = graphe.id
            if graphe.id == "PROXY_APW_ENG_20080515_0931.24":
                pass
            amr_dict[graphe.id] = graphe
            assert hasattr(graphe, "tokens")
            if not "snt" in graphe.metadata:
                toks = graphe.tokens
                graphe.metadata["snt"] = " ".join(toks)
            


    #monamr = amr_dict["DF-199-192821-670_2956.4"]

    print("%d graphes AMR au total."%len(amr_liste))
    alignements = load_aligs_from_json([
        fichier_sous_graphes,
        fichier_relations,
        fichier_reentrances],
        amr_liste)
    
    # Un examen des alignements a montré que quelques alignements se font vers une portion de l’AMR non-connexe.
    # On va simplement éliminer les AMR concernés
    eliminations = []
    for idSNT, listAlig in alignements.items():
        for a in listAlig:
            if a.type in ("subgraph", "dupl-subgraph"):
                N = compter_compos_connexes(a.edges)
                if N > 1:
                    #print("AMR %s, tokens %s, %d composantes"%(idSNT, str([a.amr.tokens[tt] for tt in a.tokens]), N))
                    eliminations.append(idSNT)
                    break #sortir de la boucle for a
    eliminations = set(eliminations)
    print("Élimination de %d AMR problématiques"%len(eliminations))
    for k in eliminations:
        del alignements[k]
    print("Voilà.")
    #set_Realigs = set()
    nb_amr_ssgrphes_dedoubles = 0
    #nb_coref_probleme = 0
    nb_amr_erreurs_reentrances = 0
    nb_amr_reentrance = 0
    nb_amr_anaphore = 0
    nb_pbs_relations = 0

    #with open("liste_problemes.txt", "w", encoding="UTF-8") as FFF:
    if True:
        FFF = sys.stdout
        graphes_phrases = []
        for idSNT, listAlig in alignements.items():
            try:
                amr = listAlig[0].amr  #amr_dict[idSNT]
                #amr_dic_rel = transfo_AMR(amr)
                toks = amr.tokens
                ntoks = len(toks)
                snt = amr.metadata["snt"]

                graphe = GRAPHE_PHRASE(amr)

                graphe.ajouter_aligs(listAlig)

                if len(graphe.DSG_sg) > 0:
                    nb_amr_ssgrphes_dedoubles += 1
                    continue #On saute les AMRs qui possèdent des sous-graphes dédoublés.
   
                if len(graphe.REN_ag) > 0:
                    nb_amr_reentrance += 1

                if not graphe.verifier_reentrance():
                    nb_amr_erreurs_reentrances += 1
                    #Éliminons ces AMR problématiques
                    continue


                autorises = ["reentrancy:repetition", "reentrancy:coref"] #"reentrancy:primary"]
                mentions = graphe.REN_ag.select(lambda x: (x.type in autorises))
                if len(mentions) > 0:
                    nb_amr_anaphore += 1
                
                graphe.traiter_reentrances()
                
                ok = graphe.calculer_graphe_toks()
                if not ok:
                    nb_pbs_relations += 1
                    continue
                    
                #Enregistrement du graphe dans le dico final
                #graphes_phrases.append(graphe)
                graphes_phrases.append(graphe)
            except Exception as e:
                print("Exception !")
                print(e)
                raise
    print("Nombres d’AMR à sous-graphes dédoublés : %d"%nb_amr_ssgrphes_dedoubles)
    #print("Nombres prbs dans les coréférences : %d"%nb_coref_probleme)
    print("Nombres d’AMR à réentrance : %d"%nb_amr_reentrance)
    print("Nombres d’AMR à anaphore : %d"%nb_amr_anaphore)
    print("Nombres d’AMR à erreurs dans les réentrances : %d"%nb_amr_erreurs_reentrances)
    print("Nombre de problèmes dans les relations : %d"%nb_pbs_relations)
    #print(str(set_Realigs))
    print()
    print("Nb d’AMR restants : %d"%len(graphes_phrases))
    print("Écriture dans le fichier...")
    ecrire_structure_dans_fichier(graphes_phrases[:500], "./AMR_et_graphes_phrases_2.txt")
    print("TERMINÉ.")
            
AMR_problematique = """
# ::id bolt12_6453_3271.7 ::date 2012-12-19T12:03:10 ::annotator SDL-AMR-09 ::preferred
# ::snt First, what is the biggest puzzle between China and the US?
# ::save-date Sun Oct 22, 2017 ::file bolt12_6453_3271_7.txt
(p / puzzle-01 :li 1
      :ARG1 (a2 / and
            :op1 (c / country :wiki "China" :name (n2 / name :op1 "China"))
            :op2 (c2 / country :wiki "United_States" :name (n / name :op1 "US")))
      :ARG2 (a / amr-unknown)
      :ARG1-of (h / have-degree-91
            :ARG2 (b / big)
            :ARG3 (m / most)))
"""

def test_aligneur():
    aligneur = ALIGNEUR("roberta-base")
    #phrase = "The lack or serious shortage of intermediate layers of Party organizations and units between the two has resulted in its inability to consider major issues with endless minor issues on hand, such that even if it is highly capable, it won't last long, as it will be dragged down by numerous petty things."
    #toksH = ["The", "lack", "or", "serious", "shortage", "of", "intermediate",
    #         "layers", "of", "Party", "organizations", "and", "units", "between",
    #         "the", "two", "has", "resulted", "in", "its", "inability", "to",
    #         "consider", "major", "issues", "with", "endless", "minor", "issues",
    #         "on", "hand", ",", "such", "that", "even", "if", "it", "is", "highly",
    #         "capable", ",", "it", "will", "n't", "last", "long", ",", "as", "it", 
    #         "will", "be", "dragged", "down", "by", "numerous", "petty", "things", "."]
    phrase = "if Establishing it won't last long."
    toksH = ["if", "Estab", "lishing", "it", "will", "n't", "last", "long"]
    rel, toksV = aligneur.aligner_seq(toksH, phrase)
    for t in rel.table:
        tV, tH = t
        if tV >= 0 and tV < len(toksV):
            tV = toksV[tV]
        else:
            tV = "[]"
        if tH >= 0 and tH < len(toksH):
            tH = toksH[tH]
        else:
            tH = "[]"
        print("%s --> %s"%(tV, tH))

if __name__ == "__main__":
    #test_aligneur()
    construire_graphes()
