from transformers import AutoTokenizer
from amr_utils.amr_readers import AMR_Reader, AMR
from amr_utils.amr_readers import Matedata_Parser as Metadata_Parser
from amr_utils.alignments import AMR_Alignment
from examiner_framefiles import EXPLICITATION_AMR #expliciter_AMR
from aligneur_seq import aligneur_seq
from algebre_relationnelle import RELATION
from enchainables import MAILLON
import re
import os
import sys
import random
import json
import numpy as np
import tqdm



def alig_to_string(self):
    resu = dict()
    resu["type"] = self.type
    #resu["tokens"] = self.tokens
    resu["tokens"] = self.word_ids
    resu["nodes"] = self.nodes
    resu["edges"] = self.edges
    return repr(resu)

AMR_Alignment.__str__ = alig_to_string
AMR_Alignment.__repr__ = alig_to_string

class AMR_modif(AMR):
    def __init__(self, amr):
        self.id = amr.id
        self.tokens = amr.tokens
        self.words = amr.tokens #nouvel attribut
        self.root = amr.root
        self.nodes = amr.nodes
        self.edges = amr.edges
        self.metadata = amr.metadata
        self.variables = amr.variables
        if hasattr(amr, "reconstruction"):
            self.reconstruction = amr.reconstruction
        if hasattr(amr, "prefix"):
            self.prefix = amr.prefix
        if hasattr(amr, "amr_chaine_brute"):
            self.amr_chaine_brute = amr.amr_chaine_brute
        if hasattr(amr, "isi_node_mapping"):
            self.isi_node_mapping = amr.isi_node_mapping
        if hasattr(amr, "jamr_node_mapping"):
            self.isi_node_mapping = amr.jamr_node_mapping


    def amr_to_string(self):
        liste = ["id", "amr-annotator", "preferred"]
        resu1 = [("id", self.id)]
        resu2 = []
        for k, v in self.metadata.items():
            if k == "id":
                pass
            if k in ("tok", "snt", "alignments"):
                resu2.append((k,v))
            else:
                resu1.append((k,v))
        if not "tok" in self.metadata and self.words:
            resu2.append(("tok", " ".join(self.words)))
        resu1.sort(key=lambda x: liste.index(x[0]) if x[0] in liste else len(liste)   )
        resu1 = " ".join(["::%s %s"%(k,v) for k,v in resu1])
        resu2 = "\n".join(["# ::%s %s"%(k,v) for k,v in resu2])
        resu = "# " + resu1 + "\n" + resu2 + "\n" + self.amr_chaine_brute
        return resu
    
    def isi_edges(self):
        for s,r,t in self.edges:
            yield (self.isi_node_mapping[s], r, self.isi_node_mapping[t])

    def isi_edges_redir(self):
        for s,r,t in self.edges_redir():
            yield (self.isi_node_mapping[s], r, self.isi_node_mapping[t])

    #def fabriquer_relations(self):
    #    Rsommets = RELATION("sommet", "variable", "constante")
    #    Rarcs = RELATION("source", "cible", "relation")
    #
    #
    #    for k, v in self.nodes.items():
    #        if k in self.variables:
    #            Rsommets.add((self.isi_node_mapping[k], v, None))
    #        else:
    #            Rsommets.add((self.isi_node_mapping[k], None, v))
    #
    #    for s, r, t in self.isi_edges():
    #        Rarcs.add((s, t, r))
    #
    #
    #    self.Rsommets = Rsommets
    #    self.Rarcs = Rarcs

    def __getattr__(self,item):
        if item == "rel_sommets":
            if not hasattr(self, "Rsommets"):
                #self.fabriquer_relations()
                Rsommets = RELATION("sommet", "variable", "constante")
                dico = set()
                for (s,t,r) in self.rel_arcs:
                    dico.add(s)
                    dico.add(t)
                for nd in dico:
                    if nd in self.variables:
                        Rsommets.add((nd, self.nodes[nd], None))
                    else:
                        Rsommets.add((nd, None, self.nodes[nd]))
                self.Rsommets = Rsommets
            return self.Rsommets
        elif item == "rel_arcs":
            if not hasattr(self, "Rarcs"):
                Rarcs = RELATION("source", "cible", "relation")
                for s, r, t in self.isi_edges():
                    Rarcs.add((s, t, r))
                self.Rarcs = Rarcs
            return self.Rarcs
        elif item == "rel_arcs_redir":
            if not hasattr(self, "Rarcs_redir"):
                Rarcs_redir = RELATION("source", "cible", "rel_redir")
                for s, t, r in self.rel_arcs:
                    if r.endswith('-of') and r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
                        s, t, r = t, s, r[:-len("-of")]
                    Rarcs_redir.add((s, t, r))
                self.Rarcs_redir = Rarcs_redir
            return self.Rarcs_redir
        else:
            return super().__getitem__(item)


class ALIGNEUR:
    def __init__(self, nom_modele, ch_debut = "", ch_suite = "##", tok1 = None, tokn = None):
        # ch_debut est la chaine qui préfixe les tokens qui constituent le début d’un mot.
        # généralement, il s’agit d’une chaine vide. Pour RoBERTa, il s’agit de "Ġ" (bizarre)
        # ch_suite est la chaine qui préfixe les tokens qui consituent la suite d’un mot
        # déjà commencé. Pour BERT, il s’agit de "##"

        assert len(ch_debut) == 0 or len(ch_suite)==0

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

    def aligner_seq(self, motsH, phrase):
        toks_nums = [x for x in self.tokenizer(phrase).input_ids]
        toks_transformer = self.tokenizer.convert_ids_to_tokens(toks_nums)
        if len(self.ch_debut) == 0 and len(self.ch_suite) > 0:
            lpf = len(self.ch_suite)
            toks_transformer = ["¤"+x[lpf:] if x.startswith(self.ch_suite) else x for x in toks_transformer]
        elif len(self.ch_debut) > 0 and len(self.ch_suite) == 0:
            lpf = len(self.ch_debut)
            toks_transformer = [x[lpf:] if x.startswith(self.ch_debut) else "¤"+x for x in toks_transformer]

        toksV = [self.tokenizer.decode(x).strip().lower() for x in toks_nums]
        assert toksV[0] == self.tok1
        assert toksV[-1] == self.tokn
        toksV = toksV[1:-1] #Éliminons les tokens CLS et SEP.
        rel_tg, rel_mg = aligneur_seq(motsH, toksV, zeroV = 1)
        return rel_tg, rel_mg, toks_transformer
    

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


def transfo_aligs(amr, jason, explicit=None):
    egal = lambda x,y: (amr.isi_node_mapping[x] == amr.isi_node_mapping[y])
    aligs = []
    elimine = False
    for a in jason:
        #type_aligs.add(a["type"])
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
            elif (not explicit is None) and r.startswith(":ARG"):
                SS = amr.nodes[amr.isi_node_mapping[s]]
                TT = amr.nodes[amr.isi_node_mapping[t]]
                RR = explicit.expliciter(SS,r,TT)
                if RR != r:
                    a['edges'][i] = [s,RR,t]

                
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


def load_aligs_from_json(json_files, amrs=None, explicit=None):
    if amrs:
        amrs = {amr.id:amr for amr in amrs}
    else:
        raise Exception('To un-anonymize alignments, the parameter "amrs" is required.')
    alignments = dict()
    #type_aligs = set()
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
        if not k in amrs:
            print('Failed to un-anonymize: no matching AMR:', k)
            del alignments[k]
            continue
        aligs = transfo_aligs(amrs[k], alignments[k], explicit)
        if len(aligs) == 0:
            del alignments[k]
        else:
            alignments[k] = aligs
    
    #print("Il y a des alignements de %d types différents"%len(type_aligs))
    #print("Voici la liste :")
    #print(type_aligs)
    #print("-----")
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
        #self.mots = amr.tokens #À changer ultérieurement
        #self.tokens = amr.tokens #À changer ultérieurement
        self.mots = amr.words

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
        
        #self.N = len(self.tokens)
        self.N = len(self.mots)
        self.amr_sommets = self.amr.rel_sommets
        self.amr_arcs = self.amr.rel_arcs
        self.amr_arcs_redir = self.amr.rel_arcs_redir

    def ajouter_aligs(self, listAlig, toks_transfo=None, rel_tg=None, rel_mg=None):
        #toks_transfo est la liste in extenso des tokens du transformer
        #rel_tg est la relation num_token -- groupe
        #rel_mg est la relation num_mot -- groupe
        
        SG_mG = RELATION("mot", "G")        #subgraph mot -- groupe
        SG_sG = RELATION("sommet", "G")     #sous-graphe  sommet - groupe
        DSG_mG = RELATION("mot", "G")       #sous-graphe dupliqué mot -- groupe
        DSG_sG = RELATION("sommet", "G")    #sous-graphe dupliqué sommet -- groupe
        REL_mG = RELATION("mot", "G")       # relation mot -- groupe
        REL_aG = RELATION("source", "cible", "relation", "G") # relation arete (source, cible, relation) -- groupe
        REN_mG = RELATION("mot", "type", "G") # réentrance mot -- groupe
        REN_aG = RELATION("source", "cible", "relation", "type", "G") # réentrance arête -- groupe

        id_groupe = 0

        amr_nodes = self.amr_sommets.p("sommet")
        
        for a in listAlig: ## Construction des tables relationnelles
            if a.type == "subgraph":
                #SG_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                #SG_sG.add(*[(self.amr.isi_node_mapping[sommet], "G%d"%id_groupe) for sommet in a.nodes])
                a_nodes = [self.amr.isi_node_mapping[sommet] for sommet in a.nodes]
                a_nodes = [s for s in a_nodes if (s,) in amr_nodes.table]
                if len(a_nodes) > 0:
                    SG_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                    SG_sG.add(*[(sommet, "G%d"%id_groupe) for sommet in a_nodes])
                id_groupe += 1
            elif a.type == "dupl-subgraph":
                #DSG_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                #DSG_sG.add(*[(self.amr.isi_node_mapping[sommet], "G%d"%id_groupe) for sommet in a.nodes])
                a_nodes = [self.amr.isi_node_mapping[sommet] for sommet in a.nodes]
                a_nodes = [s for s in a_nodes if (s,) in amr_nodes.table]
                if len(a_nodes) > 0:
                    DSG_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                    DSG_sG.add(*[(sommet, "G%d"%id_groupe) for sommet in a_nodes])
                id_groupe += 1
            elif a.type == "relation":
                #REL_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                #REL_aG.add(*[(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r, "G%d"%id_groupe) for s,r,c in a.edges])
                #id_groupe += 1
                a_edges = [(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r) for s,r,c in a.edges]
                a_edges = [(s,c,r) for s,c,r in a_edges if (s,c,r) in self.amr_arcs.table]
                if len(a_edges) > 0:
                    REL_mG.add(*[(mot, "G%d"%id_groupe) for mot in a.word_ids])
                    REL_aG.add(*[(s,c,r, "G%d"%id_groupe) for s,c,r in a_edges])
                    id_groupe += 1
            elif a.type.startswith("reentrancy"):
                #REN_mG.add(*[(mot, a.type, "G%d"%id_groupe) for mot in a.word_ids])
                #REN_aG.add(*[(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r, a.type, "G%d"%id_groupe) for s,r,c in a.edges])
                #id_groupe += 1
                a_edges = [(self.amr.isi_node_mapping[s], self.amr.isi_node_mapping[c], r) for s,r,c in a.edges]
                a_edges = [(s,c,r) for s,c,r in a_edges if (s,c,r) in self.amr_arcs.table]
                if len(a_edges) > 0:
                    REN_mG.add(*[(mot, a.type, "G%d"%id_groupe) for mot in a.word_ids])
                    REN_aG.add(*[(s,c,r, a.type, "G%d"%id_groupe) for s,c,r in a_edges])
                    id_groupe += 1

        if toks_transfo is None:
            self.SG_mg = SG_mG.ren("mot", "groupe")
            self.SG_sg = SG_sG.ren("sommet", "groupe")
            self.DSG_mg = DSG_mG.ren("mot", "groupe")
            self.DSG_sg = DSG_sG.ren("sommet", "groupe")
            self.REL_mg = REL_mG.ren("mot", "groupe")
            self.REL_ag = REL_aG.ren("source", "cible", "relation", "groupe")
            self.REN_mg = REN_mG.ren("mot", "type", "groupe")
            self.REN_ag = REN_aG.ren("source", "cible", "relation", "type", "groupe")
        else:
            #self.mots = self.tokens
            self.tokens = toks_transfo
            self.N = len(self.tokens)
            rel_SG_gG = (rel_mg.ren("mot", "g") * SG_mG.ren("mot", "G")).p("g", "G")
            rel_SG_gG = rel_SG_gG * RELATION("typ").add(("SG",))
            rel_DSG_gG = (rel_mg.ren("mot", "g") * DSG_mG.ren("mot", "G")).p("g", "G")
            rel_DSG_gG = rel_DSG_gG * RELATION("typ").add(("DSG",))
            rel_REL_gG = (rel_mg.ren("mot", "g") * REL_mG.ren("mot", "G")).p("g", "G")
            rel_REL_gG = rel_REL_gG * RELATION("typ").add(("REL",))
            rel_REN_gG = (rel_mg.ren("mot", "g") * REN_mG.ren("mot", "type", "G")).p("g", "G")
            rel_REN_gG = rel_REN_gG * RELATION("typ").add(("REN",))

            rel_gG1 = rel_SG_gG + rel_DSG_gG
            rel_gG = rel_gG1 + rel_REL_gG + rel_REN_gG
            rel_GG = (rel_gG1.ren("g","G1","typ") * rel_gG1.ren("g","G2","typ")).p("G1","G2").s(lambda x: x.G1 <= x.G2)
            rel_GG_sup = RELATION("G1", "G2")
            rel_GG_sup.add(*[(G, G) for (G,) in rel_REL_gG.p("G")])
            rel_GG_sup.add(*[(G, G) for (G,) in rel_REN_gG.p("G")])
            rel_GG = rel_GG + rel_GG_sup
            #rel_GG est une relation de type groupe de tokens -- groupe de tokens, tels qu’il 
            #existe un arc correspondant dans l’AMR.

            grps = [G for (G,) in rel_gG.p("G")]
            couleurs = {g : i for i, g in enumerate(grps)}
            for G1, G2 in rel_GG:
                c1 = couleurs[G1]
                c2 = couleurs[G2]
                if c1 > c2:
                    c1, c2 = c2, c1
                if(c1 !=  c2):
                    for G, c in couleurs.items():
                        if c == c2:
                            couleurs[G] = c1
            rel_GH = RELATION("G", "H")
            rel_GH.add(*[(G,"H%d"%H) for G,H in couleurs.items()])

            
            SG_tH = (rel_tg * rel_mg * SG_mG * rel_GH).p("token", "H")
            DSG_tH = (rel_tg * rel_mg * DSG_mG * rel_GH).p("token", "H")
            REL_tH = (rel_tg * rel_mg * REL_mG * rel_GH).p("token", "H")
            REN_tH = (rel_tg * rel_mg * REN_mG * rel_GH).p("token", "type", "H")

            SG_sH = (SG_sG * rel_GH).p("sommet", "H")
            DSG_sH = (DSG_sG * rel_GH).p("sommet", "H")
            REL_aH = (REL_aG * rel_GH).p("source", "cible", "relation", "H")
            REN_aH = (REN_aG * rel_GH).p("source", "cible", "relation", "type", "H")

            self.SG_mg = SG_tH.ren("mot", "groupe")
            self.SG_sg = SG_sH.ren("sommet", "groupe")
            self.DSG_mg = DSG_tH.ren("mot", "groupe")
            self.DSG_sg = DSG_sH.ren("sommet", "groupe")
            self.REL_mg = REL_tH.ren("mot", "groupe")
            self.REL_ag = REL_aH.ren("source", "cible", "relation", "groupe")
            self.REN_mg = REN_tH.ren("mot", "type", "groupe")
            self.REN_ag = REN_aH.ren("source", "cible", "relation", "type", "groupe")
            

    def verifier_reentrance(self):
        # le schéma de self.amr_arcs est ("source", "cible", "relation")
        # le schéma de self.REN_ag est ("source", "cible", "relation", "type", "groupe")
        reen = self.REN_ag.proj("source", "cible", "relation")
        liste_ = set(nd for (nd,) in reen.proj("cible"))
        #liste_ contient la liste des cibles d’arcs indiqués comme arcs de réentrance
        
        arcs_nd = self.amr_arcs.select(lambda x: x.cible in liste_)
        #arcs_nd contient la liste des cibles d’arcs dans l’AMR contenus dans liste_
        
        #if not len(reen) == len(arcs_nd):
        #    return False
        if any(not e in arcs_nd.table for e in reen.table):
            return False
        if any(not e in reen.table for e in arcs_nd.table):
            return False
        
        #les deux listes doivent être identiques
        
        return True


    def traiter_reentrances(self):
        self.anaphores_mg = RELATION(*(self.REN_mg.sort))
        self.anaphores_ag = RELATION(*(self.REN_ag.sort))
        if len(self.REN_ag) == 0:
            return
        toks = self.tokens
        settoks = set(m for (m, _) in self.SG_mg)
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

        for (t,) in mots:
            if not toks_libres[t]:
                print("Certaines anaphores utilisent des mots déjà reliés à des sommets de l’AMR")
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
        for (g,) in groupes:
            sommets = self.anaphores_ag.select(lambda x: x.groupe == g).proj("cible")
            if not len(sommets) == 1:
                pass
            assert len(sommets) == 1
            som = list(s for (s,) in sommets)[0]
            t_t = [t for (t,_,_) in self.anaphores_mg.s(lambda x: (x.groupe == g))]
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
        if False:
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
        gr_mots = gr_mots.s(lambda x: x.mot_s != x.mot_c)

        #calcul de la relation mot-mot pour les anaphores.
        rel4 = (rel3.ren("mot_s", "groupe") * rel3.ren("mot_c", "groupe")).p("mot_s", "mot_c")
        rel4 = rel4.s(lambda x: x.mot_s != x.mot_c)
        idem = (rel4-rel2)*(RELATION("rel").add(("{idem}", )))

        #relation finale:
        graphe_toks = groupes + idem + gr_mots
        #filtrage
        dic_filtrage = dict()
        sans_classement = set()
        interdits = set()
        for s,c,r in (groupes+idem):
            dic_filtrage[(s,c)] = r
        for s,c,r in gr_mots:
            if ((s,c) in dic_filtrage) or ((c,s) in dic_filtrage):
                interdits.add((s,c,r))
                if (s,c) in dic_filtrage:
                    interdits.add((s,c,dic_filtrage[(s,c)]))
                if (c,s) in dic_filtrage:
                    interdits.add((c,s,dic_filtrage[(c,s)]))
                sans_classement.add((s,c,"{ne_pas_classer}"))
                sans_classement.add((c,s,"{ne_pas_classer}"))
        if len(interdits) > 0:
            graphe_toks = graphe_toks.s(lambda x: x not in interdits)
            #graphe_toks.add(*list(sans_classement))
            #print("### %s "%self.amr.id)


        if False:
            N = len(graphe_toks)
            if N > len(graphe_toks.p("mot_s", "mot_c")):
                # Si le cardinal diminue quand on enlève le type de
                # relation sémantique, c’est qu’il y a plusieurs arêtes
                # différentes entre deux sous-graphes.
                # C’est un cas d’erreur.
                return False
        self.graphe_toks = graphe_toks
        return True
    
    def jsonifier(self):
        jsn = dict()
        #jsn["mots"] = self.mots
        jsn["tokens"] = self.tokens
        N = len(self.tokens)
        settoks = set(m for (m, _, _) in self.graphe_toks)
        settoks = settoks.union(set(m for (_, m, _) in self.graphe_toks))
        sommets = [i for i in range(N) if i in settoks]
        NS = len(sommets)
        corresp = [None]*N
        for i, s in enumerate(sommets):
            corresp[s] = i
        
        rel1 = self.SG_mg + self.anaphores_mg.p("mot", "groupe")
        rel2 = self.SG_sg + (self.anaphores_ag.p("cible", "groupe").ren("sommet", "groupe"))
        rel = (rel1 * rel2).p("mot", "sommet")
        dico = dict()
        for (m,s) in rel:
            if m in dico:
                dico[m].append(s)
            else:
                dico[m] = [s]
        for m in dico:
            dico[m].sort()

        dictok = [dico[sommets[i]] for i in range(NS)]
        aretes = []
        for (s,c,r) in self.graphe_toks:
            s,c = corresp[s], corresp[c] 
            aretes.append((s,r,c))
        aretes.sort()
        jsn["sommets"] = sommets
        jsn["dicTokens"] = dictok
        jsn["aretes"] = aretes
        jsn = json.dumps(jsn)
        return jsn

def compter_compos_connexes(aretes_):
    # aretes_ est un ensemble de triplets (s, r, c) alignés avec un ou plusieurs mots.
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
        #À ce stade, on a transformé la source et la cible en des numéros uniques
        aretes.append((s1, c1))

    #chaque sommet reçoit une couleur différente au départ. N est le nombre de sommets
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
        



def find_sub_list(liste, morceau, indice=0):
    matches = []
    n = len(morceau)
    for i in range(indice, len(liste)):
        if liste[i] == morceau[0] and liste[i:i+n] == morceau:
            matches.append(i)
    return matches


def yield_prefix(nf):
    etat = 0
    lignes = []
    with open(nf, "r", encoding="utf-8") as FICHIER:
        for ligne in FICHIER:
            ligne = ligne.strip()
            if ligne.startswith("#"):
                lignes.append(ligne)
                etat = 1
            elif etat == 1:
                yield "\n".join(lignes)
                lignes = []
                etat = 2
        if etat == 1:
            yield "\n".join(lignes)
            lignes = []

def quick_read_amr_file(nom_fichier, dico_snt):
    # Renvoie un dico python dont les clés sont les ids d’AMR et les valeurs
    # sont les phrases
    for pfx in yield_prefix(nom_fichier):
        metadata, _ = Metadata_Parser.readlines(pfx)
        if "id" in metadata and "snt" in metadata:
            id = metadata["id"]
            snt = metadata["snt"]
            dico_snt[id] = snt
    return dico_snt


    
        

def compter_reifications():
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    doublons = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']

    amr_reader = AMR_Reader()
    dicomptage = dict()
    for amrfile in tqdm.tqdm(fichiers_amr):
        #print(amrfile)
        listeG = [G for G in amr_reader.load(amrfile, remove_wiki=True, link_string=False) if not G.id in doublons] #Élimination des doublons
        for amr in listeG:
            for idV, V in amr.nodes.items():
                if V[-3:-1] == "-9":
                    if not V in dicomptage:
                        dicomptage[V] = 1
                    else:
                        dicomptage[V] += 1
    for k,v in dicomptage.items():
        print(k,v)

             
def recenser_relation_numerotee():
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    doublons = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']

    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    amr_reader = AMR_Reader()
    dico_opn = dict()
    for amrfile in fichiers_amr:
        listeG = [G for G in amr_reader.load(amrfile, remove_wiki=True, link_string=True) if not G.id in doublons]
        for AMR in listeG:
            idSNT = AMR.id
            dictmp = dict()
            for source,r,t in AMR.edges_redir():
                source = AMR.isi_node_mapping[source]
                resu = re.search("^:(\D+)(\d+)$",r)
                if resu:
                    rel = resu[1]
                    if rel == "ARG":
                        continue
                    chiffre = int(resu[2])
                    if not rel in dictmp:
                        dictmp[rel] = dict()
                    if source in dictmp[rel]:
                        #dictmp[rel][source] = max(dictmp[rel][source], chiffre)
                        dictmp[rel][source] += 1
                    else:
                        #dictmp[rel][source] = chiffre
                        dictmp[rel][source] = 1
            for rel, dico in dictmp.items():
                #if not rel in dico_opn:
                #    dico_opn[rel] = dict()
                rel1 = rel + "_1"
                reln = rel + "_N"
                for s, nb in dico.items():
                        assert nb >= 1
                        S = AMR.nodes[s]
                        if nb == 1:
                            if not rel1 in dico_opn:
                                dico_opn[rel1] = dict()
                            if not S in dico_opn[rel1]:
                                dico_opn[rel1][S] = [idSNT]
                            else:
                                dico_opn[rel1][S].append(idSNT)
                        else:
                            #S = S + "+"
                            if not reln in dico_opn:
                                dico_opn[reln] = dict()
                            if not S in dico_opn[reln]:
                                dico_opn[reln][S] = (nb, [idSNT])
                            else:
                                nnb, lsm = dico_opn[reln][S]
                                lsm.append(idSNT)
                                dico_opn[reln][S] = (max(nnb, nb), lsm)
    resu0 = dict()
    resu1 = dict()
    for rel, dico in dico_opn.items():
        list_opn = []
        if rel.endswith("_1"):
            for k,v in dico.items():
                list_opn.append((k,len(v)))
                resu1[rel + "_" + k] = v
        else:
            for k,v in dico.items():
                list_opn.append(("%s"%k, v[0], len(v[1])))
                resu1[rel + "_" + k] = v[1]
        
        list_opn.sort(key = lambda x : x[-1])
        resu0[rel] = list_opn
    return resu0, resu1

def construire_html_pour_opN(
        resu0, resu1,
        patron = "./observation_op_N_patron.html",
        fout = "./observation_op_N.html"):
    prefixe = []
    suffixe = []
    with open(patron, "r", encoding="utf-8") as F:
        etat = 0
        aremplir = prefixe
        for ligne in F:
            ligne = ligne.strip()
            if etat == 0 and ligne == "<!-- INSÉRER ICI -->":
                    etat = 1
                    aremplir = suffixe
            else:
                aremplir.append(ligne)
    prefixe = "\n".join(prefixe)
    suffixe = "\n".join(suffixe)
    with open(fout, "w", encoding="utf-8") as F:
        print(prefixe, file=F)
        print('\n<div class="gauche">\n', file=F)
        cpt = 0
        for K, liste in resu0.items():
            print("<div>\n<h2>relation %s</h2>\n"%K, file=F)
            if K.endswith("_1"):
                for rel, n in liste:
                    print("<button onclick='FFF_%d.F(event)'>%s: %d</button>  "%(cpt, rel, n), file=F)
                    cpt += 1
            else:
                for rel, n, N in liste:
                    print("<button onclick='FFF_%d.F(event)'>%s (%d): %d</button> "%(cpt, rel, n, N), file=F)
                    cpt += 1
            print("\n</div>\n", file=F)
        print("\n</div>\n", file=F)
        print("<script>\n", file=F)
        cpt = 0
        for K, liste in resu0.items():
            for rels in liste:
                rel = rels[0]
                nom = "Relation %s, Nœud %s"%(K, rel)
                idsSNT = resu1[K + "_" + rel]
                if len(idsSNT) > 50:
                    idsSNT = random.sample(idsSNT, 50)
                ligne = "let FFF_%d = new exemples(\"%s\",%s);\n"%(cpt, nom, json.dumps(idsSNT))
                print(ligne, file=F)
                cpt += 1
        print("</script>\n", file=F)
        print(suffixe, file=F)


def yield_AMR_1_par_1(fichier):
    lignes = []
    with open(fichier, "r", encoding="utf-8") as F:
        for ligne in F:
            ligne = ligne.strip()
            if len(ligne) == 0:
                yield "\n".join(lignes)
                lignes = []
            else:
                lignes.append(ligne)

def parse_one_AMR(parser, AMRtxt, remove_wiki=False, output_alignments=False, link_string=False):
    no_tokens = False
    resu = parser.loads(AMRtxt, remove_wiki, output_alignments, no_tokens, link_string)
    if not resu:
        return False
    if output_alignments:
        amr, aligns = resu[0], resu[1]
        return amr, aligns
    else:
        amr = resu
        return amr


def essai_AMR_string():
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    doublons = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']
    amr_reader = AMR_Reader()
    for amrfile in fichiers_amr:
        #print(amrfile)
        for AMRtxt in yield_AMR_1_par_1(amrfile):
            AMR1 = amr_reader.loads(AMRtxt, remove_wiki=True, output_alignments=False, no_tokens=False, link_string=True)
            if not AMR1:
                continue
            print(AMR1.id)
            string = AMR1.amr_string()
            AMR2 = amr_reader.loads(string, remove_wiki=True, output_alignments=False, no_tokens=False, link_string=True)
            #print(string)
            iden = AMR1.id
            assert all(k in AMR2.nodes and AMR2.nodes[k] == v for k,v in AMR1.nodes.items()), ("Nodes %s %s"%(amrfile, iden))
            assert all(k in AMR1.nodes and AMR1.nodes[k] == v for k,v in AMR2.nodes.items()), ("Nodes %s %s"%(amrfile, iden))
            assert all(e in AMR2.edges for e in AMR1.edges), ("Edges %s %s"%(amrfile, iden))
            assert all(e in AMR1.edges for e in AMR2.edges), ("Edges %s %s"%(amrfile, iden))
            assert all(e in AMR2.variables for e in AMR1.variables), ("Variables %s %s"%(amrfile, iden))
            assert all(e in AMR1.variables for e in AMR2.variables), ("Variables %s %s"%(amrfile, iden))
            assert all(k in AMR2.isi_node_mapping and AMR2.isi_node_mapping[k] == v for k,v in AMR1.isi_node_mapping.items()), ("ISI Node Mapping %s %s"%(amrfile, iden))
            assert all(k in AMR1.isi_node_mapping and AMR1.isi_node_mapping[k] == v for k,v in AMR2.isi_node_mapping.items()), ("ISI Node Mapping %s %s"%(amrfile, iden))



#essai_AMR_string()




def preparer_alignements(explicit_arg=False, **kwargs):
    #explicit_arg, si VRAI, transformera tous les rôles ARGn en une description sémantique
    #plus explicite. (Sans doute plus facile à classer, également.)

    fichiers_amr = kwargs["fichiers_amr"]
    fichiers_snt = kwargs["fichiers_snt"]
    if "doublons" in kwargs:
        doublons = kwargs["doublons"]
    else:
        doublons = []
    fichier_sous_graphes = kwargs["fichier_sous_graphes"]
    fichier_reentrances = kwargs["fichier_reentrances"]
    fichier_relations = kwargs["fichier_relations"]



    amr_reader = AMR_Reader()
    
    if explicit_arg:
        Explicit = EXPLICITATION_AMR()
        Explicit.dicFrames = EXPLICITATION_AMR.transfo_pb2va_tsv()

    amr_liste = []
    # amr_liste sera une liste remplie d’objets AMR
    amr_dict = dict()
    # amr_dict sera un dico dont les clés seront les identifiants de phrases,
    # et dont les valeurs seront les objets AMR correspondants
    snt_dict = dict()
    # snt_dict sera un dico dont les clés sont des identifiants de phrases
    # et dont les valeurs sont les phrases modèle.

    for sntfile in fichiers_snt:
        snt_dict = quick_read_amr_file(sntfile, snt_dict)
        # snt_dict est un dico dont les clés sont des identifiants de phrases
        # et dont les valeurs sont les phrases modèle.

    for amrfile in fichiers_amr:
        #print(amrfile)
        listeG = [AMR_modif(G) for G in amr_reader.load(amrfile, remove_wiki=True, link_string=True) if not G.id in doublons] #Élimination des doublons
        # listeG est une liste remplie d’objets AMR_modif (classe dérivée de la classe AMR).
        if explicit_arg:
            listeG = [Explicit.expliciter_AMR(G) for G in listeG]
        amr_liste.extend(listeG)
        for amr in listeG:
            amrid = amr.id
            amr_dict[amr.id] = amr
            
            if not "snt" in amr.metadata:
                if amrid in snt_dict:
                    amr.metadata["snt"] = snt_dict[amrid]
                else:
                    #toks = graphe.tokens
                    toks = amr.words
                    amr.metadata["snt"] = " ".join(toks)

    


    #monamr = amr_dict["DF-199-192821-670_2956.4"]

    print("%d graphes AMR au total."%len(amr_liste))
    if explicit_arg:
        alignements = load_aligs_from_json([
            fichier_sous_graphes,
            fichier_relations,
            fichier_reentrances],
            amr_liste,
            Explicit)
    else:
        alignements = load_aligs_from_json([
            fichier_sous_graphes,
            fichier_relations,
            fichier_reentrances],
            amr_liste)
    # alignements est un dico dont les clés sont les identifiants de phrase
    # et dont les valeurs sont des listes de dico d’alignements
    # un dico d’alignements est un dico avec les clés type,
    # tokens(càd mots), nodes, et edges

    return alignements
    


    
        
    
            
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
    #motsH = ["The", "lack", "or", "serious", "shortage", "of", "intermediate",
    #         "layers", "of", "Party", "organizations", "and", "units", "between",
    #         "the", "two", "has", "resulted", "in", "its", "inability", "to",
    #         "consider", "major", "issues", "with", "endless", "minor", "issues",
    #         "on", "hand", ",", "such", "that", "even", "if", "it", "is", "highly",
    #         "capable", ",", "it", "will", "n't", "last", "long", ",", "as", "it", 
    #         "will", "be", "dragged", "down", "by", "numerous", "petty", "things", "."]
    #phrase = "if Establishing it won't last long."
    phrase = "I am sure many of us are well aware of Samuel Huntington's \"Clash of Civilizations\" theory, that future conflict would be along cultural lines between \"West\", \"East\" and \"Confucian\" blocks, whatever they are."
    #motsH = ["if", "Estab", "lishing", "it", "will", "n't", "last", "long"]
    motsH = "I am sure many of us are well aware of Samuel Huntington 's \" Clash of Civilizations \" theory , that future conflict would be along cultural lines between \" West \" , \" East \" and \" Confucian \" blocks , whatever they are ."
    motsH = motsH.split()
    rel_tg, rel_mg, toksV = aligneur.aligner_seq(motsH, phrase)
    aligs = [(v,h) for v,h in (rel_tg * rel_mg).p("token", "mot")]
    aligs.sort()
    for tV, tH in aligs:
        if tV >= 0 and tV < len(toksV):
            tV = toksV[tV]
        else:
            tV = "[]"
        if tH >= 0 and tH < len(motsH):
            tH = motsH[tH]
        else:
            tH = "[]"
        print("%s --> %s"%(tV, tH))


if __name__ == "__main__":
    #test_aligneur()
    #compter_reifications()
    
    #essai_AMR_string()
    #construire_graphes(fichier_out = "./AMR_et_graphes_phrases_explct.txt", explicit_arg = True)
    #construire_graphes(fichier_out = "./a_tej_2.txt", explicit_arg = False)
    pass