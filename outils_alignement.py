from transformers import AutoTokenizer
from amr_utils.amr_readers import AMR_Reader
from amr_utils.alignments import AMR_Alignment
from algebre_relationnelle import RELATION, transfo_AMR
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

class GRAPHE_PHRASE_2:
    def __init__(self, amr):
        self.amr = amr
        self.id = amr.id
        self.mots = amr.tokens #À changer ultérieurement
        self.tokens = amr.tokens #À changer ultérieurement
        self.setSommets = set()
        self.setTokens = set()

        self.SG_mg = None
        self.SG_sg = None
        self.DSG_mg = None
        self.DSG_sg = None
        self.REN_mg = None
        self.REN_ag = None
        
        self.N = len(self.tokens)
        #self.aretes = dict()

    def ajouter_alig_sous_graphe(self, rel_tok_grp, rel_sg_grp):
        assert rel_tok_grp.sort == ("mot", "groupe")
        assert rel_sg_grp.sort == ("sommet", "groupe")

        for (m, _) in rel_tok_grp.enum():
            self.setTokens.add(m)

        for (s, _) in rel_sg_grp.enum():
            self.setSommets.add(s)

        if self.SG_mg:
            self.SG_mg = self.SG_mg + rel_tok_grp
        else:
            self.SG_mg = rel_tok_grp

        if self.SG_sg:
            self.SG_sg = self.SG_sg + rel_sg_grp
        else:
            self.SG_sg = rel_sg_grp

    def ajouter_alig_reent(self, rel_tok_grp, rel_arc_grp):
        if self.REN_mg:
            self.REN_mg = self.REN_mg + rel_tok_grp
        else:
            self.REN_mg = rel_tok_grp
        if self.REN_ag:
            self.REN_ag = self.REN_ag + rel_arc_grp
        else:
            self.REN_ag = rel_arc_grp

    def calculer_graphe_toks(self):
        #calcul des relations {groupe}
        rel_tok_grp = self.SG_mg + self.REN_mg.p("mot", "groupe")
        groupes = (rel_tok_grp.ren("mot_s", "groupe") * (rel_tok_grp.ren("mot_c", "groupe"))).p("mot_s", "mot_c")
        groupes = groupes.s(lambda x: x.mot_s != x.mot_c)
        groupes = groupes * (RELATION("rel_redir").add("{groupe}"))
        
        #calcul des relations {idem}
        rel

    def hasToken(self, tok):
        return tok in self.setTokens
    
    def hasAnyToken(self, toks):
        if type(toks) is int:
            return self.hasToken(toks)
        return any(t in self.setTokens for t in toks)
    
    def hasAllTokens(self, toks):
        if type(toks) is int:
            return self.hasToken(toks)
        return all(t in self.setTokens for t in toks)
    
    def hasNode(self, nd):
        nd = self.amr.isi_node_mapping[nd]
        return nd in self.setSommets
    
    def hasAnyNode(self, nds):
        if type(nds) is str:
            return self.hasNode(nds)
        return any(self.amr.isi_node_mapping[nd] in self.setSommets for nd in nds)
    
    def hasAllNodes(self, nds):
        if type(nds) is str:
            return self.hasNode(nds)
        return all(self.amr.isi_node_mapping[nd] in self.setSommets for nd in nds)


class GRAPHE_PHRASE:
    def __init__(self, amr):
        self.amr = amr
        self.id = amr.id
        self.mots = amr.tokens #À changer ultérieurement
        self.tokens = amr.tokens #À changer ultérieurement
        self.dicNoeuds = dict()
        self.dicTokens = dict()
        
        # dicNoeuds est un dico dont les clés sont des sommets de l’AMR et les valeurs
        # sont des (numéro de mot, mot in extenso) dans la phrase.
        # dicTokens est un dico dont les clés sont des numéros de tokens dans la phrase
        # et les valeurs sont des numéros de sommets dans l’AMR.
        self.N = len(self.tokens)
        self.aretes = dict()

    def grouper(self, tokens, ndsAMR):
        # tokens est un entier ou une liste d’entier,
        # ndsAMR est une chaine ou une liste de chaines
        if type(tokens) is int:
            tokens = [tokens]
        if type(ndsAMR) is str:
            ndsAMR = [ndsAMR]
        ndsAMR = [self.amr.isi_node_mapping[n] for n in ndsAMR]
        for i in tokens:
            for j in tokens:
                if j != i:
                    if not (i,j) in self.aretes:
                        self.aretes[(i,j)] = "{groupe}"
            if not i in self.dicTokens:
                self.dicTokens[i] = [n for n in ndsAMR]
            else:
                self.dicTokens[i].extend(ndsAMR)
        for nd in ndsAMR:
            if not nd in self.dicNoeuds:
                self.dicNoeuds[nd] = [i for i in tokens]
            else:
                self.dicNoeuds[nd].extend(tokens)

    def sous_graphe_aligne(self, N):
        N = self.amr.isi_node_mapping[N]
        assert N in self.dicNoeuds, "Noeud non aligné encore !"
        tok = self.dicNoeuds[N][0]
        return self.dicTokens[tok]


    def idem(self, tokens, N2):
        # tokens est un entier ou une liste d’entier,
        # N2 est  un noeud AMR
        if type(tokens) is int:
            tokens = [tokens]
        else:
            assert type(tokens) in [list, tuple]
        sgN2 = self.sous_graphe_aligne(N2)
        self.grouper(tokens, sgN2)
        for i in tokens:
            for j in self.dicNoeuds[N2]:
                if i != j:
                    if not (i,j) in self.aretes:
                        self.aretes[(i,j)] = "{idem}"
                    if not (j,i) in self.aretes:
                        self.aretes[(j,i)] = "{idem}"

    def relation(self, S, rel, C):
        fautes = 0
        S = self.amr.isi_node_mapping[S]
        C = self.amr.isi_node_mapping[C]
        sgS = self.sous_graphe_aligne(S)
        sgC = self.sous_graphe_aligne(C)
        if C in sgS or S in sgC:
            # Arête interne à un sous-graphe aligné
            # Ne pas poursuivre
            return 0
        for i in self.dicNoeuds[S]:
            for j in self.dicNoeuds[C]:
                if i != j:
                    #assert not (i,j) in self.aretes
                    if (i,j) in self.aretes:
                        print("Il existe déjà une arête ! (%d, %d) %s"%(i,j,self.amr.id))
                        fautes += 1
                    self.aretes[(i,j)] = rel
        return fautes
            
    def hasToken(self, tok):
        return tok in self.dicTokens
    
    def hasAnyToken(self, toks):
        if type(toks) is int:
            return self.hasToken(toks)
        return any(t in self.dicTokens for t in toks)
    
    def hasAllTokens(self, toks):
        if type(toks) is int:
            return self.hasToken(toks)
        return all(t in self.dicTokens for t in toks)

    def hasNode(self, nd):
        nd = self.amr.isi_node_mapping[nd]
        return nd in self.dicNoeuds
    
    def hasAnyNode(self, nds):
        if type(nds) is str:
            return self.hasNode(nds)
        return any(self.amr.isi_node_mapping[nd] in self.dicNoeuds for nd in nds)
    
    def hasAllNodes(self, nds):
        if type(nds) is str:
            return self.hasNode(nds)
        return all(self.amr.isi_node_mapping[nd] in self.dicNoeuds for nd in nds)
    
    def calculer_groupes(self, simple=True):
        aretes = [k for k, v in self.aretes.items() if v == "{groupe}"]
        for i,j in aretes:
            assert (j,i) in aretes
        aretes = [(i,j) for (i,j) in aretes if i<j]
        #chaque sommet reçoit une couleur différente au départ
        
        N = len(self.tokens)
        if simple:
            #Le groupe zéro contient tous les tokens alignés à rien.
            couleurs = [0]*N
            g = 1
            for i in range(len(self.tokens)):
                if i in self.dicTokens:
                    couleurs[i] = g
                    g +=1
        else:
            #sinon, les tokens non alignés ont chacun leur groupe
            #individuel
            couleurs = [g for g in range(N)]

        #réduction des couleurs:
        for s,c in aretes:
            if couleurs[s] != couleurs[c]:
                cs = couleurs[s]
                cc = couleurs[c]
                if cc < cs:
                    cs, cc = cc, cs
                for i, k in enumerate(couleurs):
                    if k == cc:
                        couleurs[i] = cs
        groupes = dict()
        for i,g in enumerate(couleurs):
            if not g in groupes:
                groupes[g] = (i,)
            else:
                groupes[g] += (i,)
        if simple:
            groupes = [tuple(sorted(v)) for g, v in groupes.items() if g>0]
            #Éliminer le groupe des tokens alignés à rien
        else:
            groupes = [tuple(sorted(v)) for g, v in groupes.items()]
        grpTok = [[self.tokens[i] for i in g] for g in groupes]

        return groupes, grpTok, couleurs
    
    def serialiser(self):
        jsn = dict()
        #jsn["mots"] = self.mots
        jsn["tokens"] = self.tokens
        N = len(self.tokens)
        sommets = [i for i in range(N) if self.hasToken(i)]
        NS = len(sommets)
        corresp = [None]*N
        for i, s in enumerate(sommets):
            corresp[s] = i
        dictok = [self.dicTokens[sommets[i]] for i in range(NS)]
        aretes = []
        for (i,j), v in self.aretes.items():
            i,j = corresp[i], corresp[j] 
            aretes.append((i,v,j))
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

def comparer_reentrances_REL(arcs_amr, nd, reentrances):
    # le schéma de arcs_amr est ("source", "cible", "relation")
    # le schéma de reentrances est ("source", "cible", "relation", "type", "groupe")
    arcs_nd = arcs_amr.select(lambda x: x.cible == nd).rmdup()
    reen = reentrances.proj("source", "cible", "relation").rmdup()
    if not len(reen.table) == len(arcs_nd.table):
        return False
    if any(not e in arcs_nd.table for e in reen.table):
        return False
    if any(not e in reen.table for e in arcs_nd.table):
        return False

def find_sub_list(liste, morceau, indice=0):
    matches = []
    n = len(morceau)
    for i in range(indice, len(liste)):
        if liste[i] == morceau[0] and liste[i:i+n] == morceau:
            matches.append(i)
    return matches

def traiter_reentrances_REL(graphe, REN_mg, REN_ag):
    if len(REN_ag) == 0:
        return REN_mg, REN_ag
    toks = graphe.tokens
    toks_libres = [not graphe.hasToken(t) for t in range(graphe.N)]
    liste_toks_libres = [toks[i] if v else "" for i, v in enumerate(toks_libres)]
    autorises = ["reentrancy:repetition", "reentrancy:coref"] #, "reentrancy:primary"]
    libres2 = [True] * graphe.N

    #primaires = REL_ren.select(lambda x: x.type == "reentrancy:primary")
    anaphores_mg = REN_mg.select(lambda x: (x.type in autorises)).rmdup()
    anaphores_ag = REN_ag.select(lambda x: (x.type in autorises)).proj("cible", "type", "groupe").rmdup()
    mots = anaphores_mg.proj("mot").rmdup()
    if len(mots) == 0:
        return REN_mg, REN_ag

    for (t,) in mots.enum():
        if not toks_libres[t]:
            raise AssertionError("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
        libres2[t] = False

    for i, vf in enumerate(libres2):
        if not vf:
            toks_libres[i] = False
            liste_toks_libres[i] = ""

    groupes = anaphores_mg.proj("groupe").rmdup()
    ajout_mg = RELATION(*(REN_mg.sort))
    ajout_ag = RELATION(*(REN_ag.sort))

    idG = 0
    for (g,) in groupes.enum():
        sommets = anaphores_ag.select(lambda x: x.groupe == g).proj("cible")
        assert len(sommets) == 1
        som = list(s for (s,) in sommets.enum())[0]
        t_t = [t for (t,_,_) in anaphores_mg.select(lambda x: (x.groupe == g)).enum()]
        t_t.sort()
        assert all(s == 1+t_t[i] for i, s in enumerate(t_t[1:]))

        tt = [toks[t] for t in t_t]
        ph = " ".join(tt)
        if ph in [",", "person"]:
            continue
        ntt = len(tt)
        mtch = find_sub_list(liste_toks_libres, tt)
        if len(mtch) > 0:
            print("snt %s\nRépétition ou anaphore supplémentaire trouvée ! (%s)\n"%(graphe.id,ph))
            for t  in mtch:
                ajout_mg.add(*[(t+k, "reentrancy:ajout", "Gaj%d"%idG) for k in range(ntt)])
                ajout_ag.add(("_", "%s"%som, "_", "reentrancy:ajout", "Gaj%d"%idG))

                idG += 1

    if len(ajout_ag) > 0: 
        REN_mg = REN_mg + ajout_mg
        REN_ag = REN_ag + ajout_ag
    return REN_mg, REN_ag

def traiter_reentrances(graphe, amr, aligs_ren_dict):
    toks = amr.tokens
    toks_libres = [not graphe.hasToken(t) for t in range(graphe.N)]
    liste_toks_libres = [toks[i] if v else "" for i, v in enumerate(toks_libres)]
    autorises = ["reentrancy:repetition", "reentrancy:coref"] #, "reentrancy:primary"]
    libres2 = [True] * graphe.N
    anadico = {nd : [] for nd in aligs_ren_dict}
    for nd, l_alg in aligs_ren_dict.items():
        assert graphe.hasNode(nd) #Toujours le cas,
        # sauf pour les AMR à sous-graphes dédoublés,
        # mais on les a éliminés.
        anaphores = [a for a in l_alg if a.type in autorises]
        primary = [a for a in l_alg if a.type == "reentrancy:primary"]
        assert len(primary) == 1, "Plus d’une réentrance primaire"
        primary = primary[0]
        toks_anaphores = set(tuple(a.tokens) for a in anaphores)
        #if tuple(primary.tokens) in toks_anaphores:
        #    raise AssertionError("les tokens de la réentrance primaire sont aussi des anaphores.")
        for tt in toks_anaphores:
            for t in tt:
                if not toks_libres[t]:
                    raise AssertionError("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
                libres2[t] = False
            anadico[nd].append(tt) #anaphore à marquer plus tard
    for i, vf in enumerate(libres2):
        if not vf:
            toks_libres[i] = False
            liste_toks_libres[i] = ""


    for nd, l_alg in aligs_ren_dict.items():
        anaphores = [a for a in l_alg if a.type in autorises]
        toks_anaphores = set(tuple(a.tokens) for a in anaphores)
        for tt in toks_anaphores:
            ttt = [toks[t] for t in tt]
            ph = " ".join(ttt)
            if ph in [",", "person"]:
                continue
            nttt = len(ttt)
            mtch = find_sub_list(liste_toks_libres, ttt)
            if len(mtch) > 0:
                print("snt %s\nRépétition ou anaphore supplémentaire trouvée ! (%s)\n"%(amr.id,ph))
                for t  in mtch:
                    anadico[nd].append(list(range(t, t+nttt)))
    # Marquons maintenant les anaphores :
    for nd, ttt in anadico.items():
        if len(ttt) > 0:
            for tt in ttt:
                graphe.idem(tt, nd)
    return len(anadico)

                
        

def ecrire_structure_dans_fichier(graphes, fichier):
    with open(fichier, "w", encoding="UTF-8") as F:
        for graphe in graphes:
            amr = graphe.amr
            print(amr.amr_string, file=F)
            jsn = graphe.serialiser()
            print(jsn, file=F)
            print(file=F)


def transfo_aligs(listAlig, amr):
    SG_mg = RELATION("mot", "groupe")
    SG_sg = RELATION("sommet", "groupe")
    DSG_mg = RELATION("mot", "groupe")
    DSG_sg = RELATION("sommet", "groupe")
    REL_mg = RELATION("mot", "groupe")
    REL_ag = RELATION("source", "cible", "relation", "groupe")
    REN_mg = RELATION("mot", "type", "groupe")
    REN_ag = RELATION("source", "cible", "relation", "type", "groupe")

    idSG = 0
    idDSG = 0
    idREL = 0
    idREN = 0
    for a in listAlig:
        if a.type == "subgraph":
            SG_mg.add(*[(mot, "G%d"%idSG) for mot in a.tokens])
            SG_sg.add(*[(sommet, "G%d"%idSG) for sommet in a.nodes])
            idSG += 1
        elif a.type == "dupl-subgraph":
            DSG_mg.add(*[(mot, "G%d"%idDSG) for mot in a.tokens])
            DSG_sg.add(*[(sommet, "G%d"%idDSG) for sommet in a.nodes])
            idDSG += 1
        elif a.type == "relation":
            REL_mg.add(*[(mot, "G%d"%idREL) for mot in a.tokens])
            REL_ag.add(*[(amr.isi_node_mapping[s], amr.isi_node_mapping[c], r, "G%d"%idREL) for s,r,c in a.edges])
            idREL += 1
        elif a.type.startswith("reentrancy"):
            REN_mg.add(*[(mot, a.type, "G%d"%idREN) for mot in a.tokens])
            REN_ag.add(*[(amr.isi_node_mapping[s], amr.isi_node_mapping[c], r, a.type, "G%d"%idREN) for s,r,c in a.edges])
            idREN += 1

    return SG_mg, SG_sg, DSG_mg, DSG_sg, REL_mg, REL_ag, REN_mg, REN_ag



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
    fichiers_amr = fichiers_amr[0:1]

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
                amr_dic_rel = transfo_AMR(amr)
                toks = amr.tokens
                ntoks = len(toks)
                snt = amr.metadata["snt"]

                #graphe = GRAPHE_PHRASE(amr)
                graphe = GRAPHE_PHRASE_2(amr)
                #noeuds_dup = set()

                SG_mg, SG_sg, DSG_mg, DSG_sg, REL_mg, REL_ag, REN_mg, REN_ag = transfo_aligs(listAlig, amr)

                if len(DSG_sg) > 0:
                    nb_amr_ssgrphes_dedoubles += 1
                    continue #On saute les AMRs qui possèdent des sous-graphes dédoublés.

                      
                #for a in aligs_sg:
                #    graphe.grouper(a.tokens, a.nodes)
                graphe.ajouter_alig_sous_graphe(SG_mg, SG_sg)
                
                if len(DSG_mg) > 0:
                    nb_amr_reentrance += 1

                if any(comparer_reentrances_REL(amr_dic_rel["arcs"], nd, REN_ag) == False for (nd,) in REN_ag.proj("cible").rmdup().enum()):
                    nb_amr_erreurs_reentrances += 1
                    #Éliminons ces AMR problématiques
                    continue


                autorises = ["reentrancy:repetition", "reentrancy:coref"] #"reentrancy:primary"]
                
                
                mentions = REN_ag.select(lambda x: (x.type in autorises))
                if len(mentions) > 0:
                    nb_amr_anaphore += 1
                
                REN_mg, REN_ag = traiter_reentrances_REL(graphe, REN_mg, REN_ag)

                graphe.ajouter_alig_reent(REN_mg, REN_ag)
                
                gr_toks = graphe.calculer_graphe_toks()
                if not gr_toks:
                    nb_pbs_relations += 1
                    continue
                    
                #Enregistrement du graphe dans le dico final
                #graphes_phrases.append(graphe)
                graphes_phrases.append(gr_toks)
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
    #ecrire_structure_dans_fichier(graphes_phrases[:500], "./AMR_et_graphes_phrases_2.txt")
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
