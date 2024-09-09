import sys
import pygraphviz as pgv
import random
import random
import re
from micro_serveur import EXEC_ADDRESS, ServeurReq, lancer_serveur


from lxml import etree as ET

from amr_utils.amr_readers import AMR_Reader
from examiner_framefiles import EXPLICITATION_AMR
#from amr_utils.propbank_frames import propbank_frames_dictionary
import json



if __name__ == "__main__":
    AMRtxt = """
    # ::snt Where is Homer Simpson when you need him ?
    (b / be-located-at-91
    :ARG0 (p / person
            :name ( h / name
                    :op1 “Homer”
                    :op2 “Simpson”))
    :ARG1 (a / amr-unknown)
    :time (n / need-01
                :polarity -
                :ARG0 (y / you)
                :ARG1 p))
    """

    AMRtxt2 = """
    # ::snt Most of the students want to visit New York when they graduate.
    (w / want-01
        :ARG0 (p / person
            :ARG0-of (s / study-01)
            :ARG1-of (i / include-91
                :ARG2 (p2 / person
                    :ARG0-of (s2 / study-01))
                :ARG3 (m / most)))
        :ARG1 (v / visit-01
            :ARG0 p
            :ARG1 (c / city
                :name ( n / name
                    :op1 "New"
                    :op2 "York"))
            :time (g / graduate-01
                :ARG0 p)))
    """
    
    AMRtxt3 = """
    (j / join-01
        :ARG0 (p / person
            :name (p2 / name
                :op1 (x0 / "Pierre")
                :op2 (x1 / "Vinken"))
            :age (t / temporal-quantity
                :quant (x3 / 61)
                :unit (y / year)))
        :ARG1 (b / board
                :ARG1-of (h / have-org-role-91
                    :ARG0 p
                    :ARG2 (d2 / director
                        :mod (e / executive
                            :polarity (x4 / -)))))
        :time (d / date-entity
            :month (x5 / 11)
            :day (x6 / 29)))
"""

    AMRtxt4 = """
    # ::snt the boy wants to go to the museum
    (w / want-01
        :ARG0 (b / boy)
        :ARG1 (g/ go-01
            :ARG0 b
            :polarity -
            :ARG1 (m / museum)))
      """


def make_propbank_frames_dictionary():
    global Explicit
    #global propbank_frames_dictionary

    Explicit = EXPLICITATION_AMR()
    Explicit.dicFrames = EXPLICITATION_AMR.transfo_pb2va_tsv()




class RECHERCHE_AMR:
    def __init__(self, nom_fichier):
        self.nom_fichier = nom_fichier
        dico = dict()
        self.offsets = []
        with open(nom_fichier, "rb") as F:
            chercher_id = True
            pos_fichier = 0
            idx_offset = 0
            for ligne in F:
                if chercher_id:
                    if ligne.startswith(b"# ::id "):
                        idSNT = ligne.split(maxsplit=3)[2]
                        idSNT = idSNT.decode("utf-8")
                        self.offsets.append(pos_fichier)
                        dico[idSNT] = idx_offset
                        idx_offset += 1
                        chercher_id = False
                elif len(ligne.strip()) == 0:
                    chercher_id = True
                pos_fichier += len(ligne)
        self.dico = dico
        self.N = len(self.offsets)
        self.reader = AMR_Reader()

    def __getitem__(self, idx):
        if type(idx) is str:
            idx = self.dico[idx] #Au besoin, émettre une erreur
        pos_fichier = self.offsets[idx] 
        with open(self.nom_fichier, "rb") as F:
            F.seek(pos_fichier)
            etape = 0
            amr_str = []
            jsn = []
            while True:
                ligne = F.readline()
                if len(ligne.strip()) == 0:
                    break
                if etape == 0:
                    if ligne.strip().startswith(b"{"):
                        etape = 1
                    else:
                        amr_str.append(ligne)
                if etape == 1:
                    jsn.append(ligne)
        amr_str = b"".join(amr_str).decode("utf-8")
        jsn = b"".join(jsn).decode("utf-8")
        jsn = json.loads(jsn)
        amr = self.reader.loads(amr_str, remove_wiki=True, output_alignments=False, link_string=True)
        return amr, jsn

    def alea(self):
        #print("self.N = %d"%self.N)
        idx = random.randint(0, (self.N)-1)
        return self.__getitem__(idx)
    
    def pick_ident(self):
        return random.choice(list(self.dico.keys()))





class GET_INDEX(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if chemin in ["/index.html", "/"]:
            return True
        else:
            return False
        
    def execute(self):
        self.sendFile("text/html; charset=utf-8", "./visu.html")


class GET_ALGEBRA(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if chemin == "/algebre_relationnelle.js":
            return True
        return False
    
    def execute(self):
        self.sendFile("text/javascript; charset=utf-8", "algebre_relationnelle.js")
  
class GET_VIVAGRAPH(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if chemin == "/vivagraph.min.js":
            return True
        return False
    
    def execute(self):
        self.sendFile("text/javascript; charset=utf-8", "vivagraph.min.js")

class GET_VIVAGRAPH_FACTORY(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if chemin == "/vivagraph_factory.js":
            return True
        return False
    
    def execute(self):
        self.sendFile("text/javascript; charset=utf-8", "vivagraph_factory.js")

def redresse_arete(r, s, t):
    if r.endswith('-of') and r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
        return r[:-len("-of")], t, s
    else:
        return (r, s, t)

class GET_LDC_2020_T02(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if (  chemin.startswith("/LDC_2020_T02/")
                and chemin.endswith(".html")
                #and chemin[-9] == "_"
                and chemin[-10] == "_"
                #and all(x in ["X", "O"] for x in chemin[-8:-5])
                and all(x in ["X", "O"] for x in chemin[-9:-5])
             ):
            return True
            # exemple de chemin : /Le_petit_prince/JAMR_125_XOO.html
        return False
    
    def execute(self):
        global recherche_amr
        global Explicit
        chemin = self.chemin
        chm = chemin[14:]
        
        #variables, voir_racine, reverse_roles = tuple(x == 'X' for x in chm[-8:-5])
        variables, voir_racine, reverse_roles, explct_roles = tuple(x == 'X' for x in chm[-9:-5])
        
        #iden = chm[:-9]
        iden = chm[:-10]
        if iden == "alea":
            identAMR = recherche_amr.pick_ident()
            amr, jsn = recherche_amr[identAMR]
            jsn["identAMR"] = identAMR
        else:
            amr, jsn = recherche_amr[iden]
        if (explct_roles):
            amr = Explicit.expliciter_AMR(amr)
        if not "snt" in amr.metadata:
            toks = amr.tokens
            amr.metadata["snt"] = " ".join(toks)
        phrase = amr.metadata["snt"]
        clusters = [C for C in jsn["dicTokens"] if len(C) > 1]
        clusters = tuple(tuple(sorted(C)) for C in clusters)
        clusters = list(set(clusters))

        G, prfx, dico_triplets = drawAMR(amr, variables, voir_racine, reverse_roles, clusters)
        G = drawSentGraph(prfx, jsn["tokens"], jsn["sommets"], jsn["aretes"], G)

        svg = grapheToSVG(amr, G)

               
        lprfx = len(prfx)
        jsn["prefixe"] = prfx
        dico_triplets = {k : v for k,v in dico_triplets.items() if k[0] != ":instance"}
        dico_triplets = [(v[0][lprfx:], v[1][lprfx:]) for v in dico_triplets.values()]
        dico_triplets = [[S1, S2] for S1, S2 in dico_triplets if not any((S1 in G and S2 in G) for G in jsn["dicTokens"])]
        jsn["triplets"] = dico_triplets
                
        svg = svg[1]
        
        arbo = ET.XML("<htmlfrag/>")
        
        arbo.append(svg)
        svg.tail = json.dumps(jsn)
        
        
        reponse = arbo
        reponse = ET.tostring(reponse, encoding="utf-8", method="xml")
        taille = len(reponse)
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", "%d"%taille)
        self.end_headers()
        self.wfile.write(reponse)
    

           
        
class PREFIXE():
    nb_cars = 1
    nb_max = 26
    prefixe_cur = 0
    prfx = ""
    @staticmethod
    def suivant():
        resu = ""
        N = PREFIXE.prefixe_cur
        for i in range(PREFIXE.nb_cars):
            resu = chr(65+32+(N%26)) + resu
            N = N//26
        PREFIXE.prefixe_cur += 1
        if PREFIXE.prefixe_cur >= PREFIXE.nb_max:
            PREFIXE.nb_max *= 26
            PREFIXE.nb_cars += 1
            PREFIXE.prefixe_cur = 0
        PREFIXE.prfx = resu
        return resu


dico_ponctuation = {",":"[virgule]", ".":"[point]", ":":"[deux points]", ";":"[point-virgule]" , "!":"[exclam]", "?":"[interrog]", "-":"[trait d’union]"};

def drawSentGraph(prefixe, toks, tk_utiles, aretes, G):
    tokens = []
    for t in toks:
        if t.startswith("¤"):
            t = t[1:]
            if t in dico_ponctuation:
                t = dico_ponctuation[t]
            else:
                t = "~"+t
        elif t in dico_ponctuation:
            t = dico_ponctuation[t]
        tokens.append(t)
    for i, s in enumerate(tk_utiles):
        G.add_node(
            "tk_%d"%i,
            label=tokens[s],
            id = "%s_tk_%d"%(prefixe, s),
            style = "filled",
            fillcolor="white",
            shape="rect"
        )
    for src, r, tgt in aretes:
        ident = "%s_tk_%d_tk_%d"%(prefixe, tk_utiles[src], tk_utiles[tgt])
        if r=="{groupe}":
            if src < tgt:
                G.add_edge("tk_%d"%src, "tk_%d"%tgt, id=ident, style="dashed", color="blue", dir="none")
        elif r == "{idem}":
            if src < tgt:
                G.add_edge("tk_%d"%src, "tk_%d"%tgt, id=ident, style="dashed", color="purple", dir="none")
        elif r == "{ne_pas_classer}":
            if src < tgt:
                G.add_edge("tk_%d"%src, "tk_%d"%tgt, id=ident, style="dashed", color="red", dir="none")
        else:
            edgL = r[1:] if r.startswith(":") else r
            G.add_edge("tk_%d"%src, "tk_%d"%tgt, label=edgL, id=ident)

    return G
    #G.draw("graphe_mot.svg", prog="dot")


def calc_tooltip_0(dicors):
    roles = [ (R if not R in Explicit.description_roles else Explicit.description_roles[R]) for R in dicors["roles"]]
    if roles[0] == None:
        start = 1
    else:
        start = 0
    roles = "\n".join( "ARG%d: %s"%(i, roles[i]) for i in range(start, len(roles)) )
    
    tooltip = [roles]
    
    rolesPB = dicors["roles_PB"]
    descr = dicors["descr"]
    rolesPB = "PropBank:\n" + "\n".join( "ARG%d: %s : %s"%(i, rolesPB[i], descr[i]) for i in range(start, len(rolesPB))   )
    tooltip.append(rolesPB)
    for K, rol in dicors.items():
        if K.startswith("roles_") and K != "roles_PB":
            K = K[6:]
            tooltip.append( K + ":\n" + "\n".join( "ARG%d: %s"%(i, rol[i]) for i in range(start, len(rol)) ) )
    tooltip = "\n-----\n".join(tooltip)
    return tooltip

def calc_tooltip(dicors):
    ARGn = [(("ARG%d"%x),r) for x,r in enumerate(dicors["ARGn"])]
    ARGn.extend((k,v) for k,v in dicors["special"].items())
    ARGn = ["%s: %s"%(x,r) for x,r in ARGn if x != r]
    tooltip = "\n".join(ARGn)
    return tooltip


def drawAMR(amr, voir_variables=False, voir_racine=False, reverse_roles=False, clusters=[]):
    clusters = [list(C) for C in clusters if len(C) > 1]
    G = pgv.AGraph(directed=True)
    prefixe = PREFIXE.suivant()
    G.graph_attr.update(id="GR_%s"%prefixe)
    G.node_attr["fontname"] = "helvetica"
    G.edge_attr["fontname"] = "helvetica"
    
    dico_triplets = dict()
    for iden in amr.variables:
        texte_rouge = False
        if voir_variables:
            argus = {"shape": "circle"}
        else:
            argus = dict()
        triplet = (":instance", iden, amr.nodes[iden])
        dico_triplets[triplet] = []
        label = amr.nodes[iden]
        argus["id"] = prefixe+iden
        
        if label in Explicit.dicFrames:
            dicors = Explicit.dicFrames[label]
            tooltip = calc_tooltip(dicors)
            if tooltip:
                argus["tooltip"] = tooltip
            else:
                texte_rouge = True
        elif re.search("-\d+$", label):
            texte_rouge = True
        if voir_variables:
            G.add_node(iden, label="",
                    style="filled",
                    fillcolor="white",
                    **argus)
        else:
            if texte_rouge:
                argus["class"] = "ROUGE"
                argus["fontcolor"] = "red"
            G.add_node(iden, label=label,
                        style="filled",
                        fillcolor="white",
                        **argus)
        dico_triplets[triplet].append(prefixe+iden)
        if voir_variables:
            id2 = "%s_concept_%s"%(prefixe, iden)
            for C in clusters:
                if iden in C:
                    C.append(id2)
            if texte_rouge:
                G.add_node( id2,
                        label=label,
                        id=id2,
                        style="filled",
                        fillcolor="white",
                        shape="rect",
                        fontcolor = "red")
            else:
                G.add_node( id2,
                            label=label,
                            id=id2,
                            style="filled",
                            fillcolor="white",
                            shape="rect")
            dico_triplets[triplet].append(id2)
            idarete = "%s→%s"%(prefixe+iden,id2)
            G.add_edge( iden,
                        id2,
                        style="dotted",
                        id=idarete)
            dico_triplets[triplet].append(idarete)
        
    if reverse_roles:
        aretes = amr.edges_redir()
    else:
        aretes = amr.edges
    for id1, rol, id2 in aretes:
        edgL = rol[1:] if rol.startswith(":") else rol
        texte_rouge = (rol.startswith(":ARG") or rol.startswith(":>ARG"))
        if id2 in amr.variables:
            triplet = (rol, id1, id2)
            idarete = "%s→%s"%(prefixe+id1,prefixe+id2)
            if texte_rouge:
                G.add_edge(id1, id2,
                           label=edgL,
                           id=idarete,
                           fontcolor="red",
                           **{"class":"ROUGE"})
            else:
                G.add_edge(id1, id2, label=edgL, id=idarete)
            dico_triplets[triplet] = [prefixe+id1, prefixe+id2, idarete]
        else:
            triplet = (rol, id1, amr.nodes[id2])
            G.add_node(id2, label=amr.nodes[id2], shape="rect", style="filled", fillcolor="white", id=prefixe+id2)
            idarete = "%s→%s"%(prefixe+id1,prefixe+id2)
            G.add_edge(id1, id2, label=edgL, id=idarete)
            dico_triplets[triplet] = [prefixe+id1, prefixe+id2, idarete]

    for i, C in enumerate(clusters):
        idclus = prefixe + "%s_clus_%d"%(prefixe, i)
        clus = G.subgraph(nbunch=C[:], name="cluster_%d"%i, color="lightblue")
    
    if voir_racine:
        triplet = ("TOP", amr.root, amr.nodes[amr.root])
        iden = "%s_TOP"%(prefixe)
        G.add_node(iden, label="", shape="house", style="filled", fillcolor="white", id=iden)
        idarete = "TOP→%s"%(prefixe+amr.root)
        G.add_edge(iden, amr.root, id=idarete)
        dico_triplets[triplet] = [prefixe+amr.root, iden, idarete]
        
    #G.layout(prog='dot')
    return G, prefixe, dico_triplets
    

def grapheToSVG(amr, G):
    G.layout(prog='dot')
    f = G.draw(path=None, format="svg")
    
    svg = ET.fromstring(f)   #.decode("utf-8"))
    largeur = svg.attrib["width"]
    hauteur = svg.attrib["height"]
    
    if largeur.endswith("pt"):
        largeur = int(largeur[:-2])*2//3
        svg.attrib["width"] = "%dpt"%largeur
    if hauteur.endswith("pt"):
        hauteur = int(hauteur[:-2])*2//3
        svg.attrib["height"] = "%dpt"%hauteur
    arbo = ET.XML("<htmlfrag/>")
    phrase = False
    if hasattr(amr, "metadata") and 'snt' in amr.metadata and len(amr.metadata["snt"]) > 0:
        phrase = amr.metadata["snt"]
    if not phrase and hasattr(amr, "tokens") and len(amr.tokens) > 0:
        tokens = amr.tokens
        while amr.tokens[0] in ("#", "::snt"):
            tokens = tokens[1:]
        if len(tokens) > 0:
            phrase = " ".join(tokens)
    if phrase:
        arbo.append(ET.XML("<h3>" + phrase + "</h3>"))
    arbo.append(svg)
    
    return arbo
 

def main(nom_fichier = "./AMR_et_graphes_phrases_2.txt"):
    global recherche_amr
    make_propbank_frames_dictionary()
    print("Ouverture du fichier %s pour dresser la table"%nom_fichier)
    recherche_amr = RECHERCHE_AMR(nom_fichier)
    print("Table dressée.")
    ServeurReq.add_get(GET_INDEX, GET_VIVAGRAPH,
                       GET_VIVAGRAPH_FACTORY,
                       GET_LDC_2020_T02,
                       GET_ALGEBRA)
    #ServeurReq.add_post(POST_CHOIX_VISU)
    lancer_serveur("localhost", 8081)


def main2():
    rech = RECHERCHE_AMR("./AMR_et_graphes_phrases.txt")
    print("voilà")
    amr, jsn = rech.alea()
    if not "snt" in amr.metadata:
        toks = amr.tokens
        amr.metadata["snt"] = " ".join(toks)
    phrase = amr.metadata["snt"]
    
    svg, pref, dico = convert2graph(amr,
                            variables=False,
                            voir_racine=False,
                            reverse_roles=False,
                            return_edges=True)
        

def main3():
    #amr = AMR_Reader().loads(AMRtxt2)
    #drawAMR(amr)
    global recherche_amr
    make_propbank_frames_dictionary()
    nom_fichier = "./AMR_et_graphes_phrases_2.txt"
    print("Ouverture du fichier %s pour dresser la table")
    recherche_amr = RECHERCHE_AMR(nom_fichier)
    print("Table dressée.")
    XX = GET_LDC_2020_T02(
        "/LDC_2020_T02/PROXY_AFP_ENG_20070201_0099.7_OOOX.html",
        None,None)
    XX.execute()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        nom_fichier = sys.argv[1]
        main(nom_fichier)
    else:
        main()
    