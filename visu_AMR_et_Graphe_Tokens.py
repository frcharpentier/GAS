import pygraphviz as pgv
import random
import re
from micro_serveur import EXEC_ADDRESS, ServeurReq, lancer_serveur


from lxml import etree as ET

from amr_utils.amr_readers import AMR_Reader
from amr_utils.propbank_frames import propbank_frames_dictionary
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
        idx = random.randint(0, self.N)
        return self.__getitem__(idx)





class GET_INDEX(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if chemin in ["/index.html", "/"]:
            return True
        else:
            return False
        
    def execute(self):
        self.sendFile("text/html; charset=utf-8", "./visu.html")

  
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
                and chemin[-9] == "_"
                and all(x in ["X", "O"] for x in chemin[-8:-5])
             ):
            return True
            # exemple de chemin : /Le_petit_prince/JAMR_125_XOO.html
        return False
    
    def execute(self):
        global recherche_amr
        chemin = self.chemin
        chm = chemin[14:]
        
        variables, voir_racine, reverse_roles = tuple(x == 'X' for x in chm[-8:-5])
        
        iden = chm[:-9]
        if iden == "alea":
            amr, jsn = recherche_amr.alea()
        else:
            amr, jsn = recherche_amr[iden]
        if not "snt" in amr.metadata:
            toks = amr.tokens
            amr.metadata["snt"] = " ".join(toks)
        phrase = amr.metadata["snt"]
        clusters = [C for C in jsn["dicTokens"] if len(C) > 1]
        clusters = tuple(tuple(sorted(C)) for C in clusters)
        clusters = list(set(clusters))

        svg, _, dico_triplets = convert2graph(amr,
                                variables=variables,
                                voir_racine=voir_racine,
                                reverse_roles=reverse_roles,
                                return_edges=True,
                                clusters = clusters)
        
        prfx = PREFIXE.prfx
        lprfx = len(prfx)
        jsn["prefixe"] = prfx
        dico_triplets = {k : v for k,v in dico_triplets.items() if k[0] != ":instance"}
        dico_triplets = [(v[0][lprfx:], v[1][lprfx:]) for v in dico_triplets.values()]
        dico_triplets = [[S1, S2] for S1, S2 in dico_triplets if not any((S1 in G and S2 in G) for G in jsn["dicTokens"])]
        jsn["triplets"] = dico_triplets
        print(repr(dico_triplets))
        
        #for clus in jsn["dicTokens"]:
        #    for i, C in enumerate(clus):
        #        clus[i] = prfx+C

        
        #dico = {redresse_arete(*k) : v for k, v in dico.items()}
        
        svg = svg[1]
        
        
        
        
        arbo = ET.XML("<htmlfrag/>")
        
        arbo.append(svg)
        jsns = json.dumps(jsn)
        jsns = re.sub("\]\]+", lambda x: " ]"*len(x[0]), jsns)
        svg.tail = ET.CDATA(json.dumps(jsn))
        
        
        reponse = arbo
        reponse = ET.tostring(reponse, encoding="utf-8", method="xml")
        taille = len(reponse)
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", "%d"%taille)
        self.end_headers()
        self.wfile.write(reponse)
    
class POST_CHOIX_VISU(EXEC_ADDRESS):
    @staticmethod
    def test(chemin):
        if (
                chemin.startswith("/")
                and chemin.endswith(".html")
                and len(chemin) == 9
                and all(x in ["X", "O"] for x in chemin[1:4])
            ):
            return True
        return False
    
    def execute(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        variables, voir_racine, reverse_roles = tuple(x == 'X' for x in self.chemin[1:4])
        reponse = convert2graph(post_data.decode('utf-8'),
                                        variables=variables,
                                        voir_racine=voir_racine,
                                        reverse_roles=reverse_roles)
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


def drawAMR(amr, voir_variables=False, voir_racine=False, reverse_roles=False, clusters=[]):
    clusters = [list(C) for C in clusters if len(C) > 1]
    G = pgv.AGraph(directed=True)
    prefixe = PREFIXE.suivant()
    G.graph_attr.update(id="GR_%s"%prefixe)
    G.node_attr["fontname"] = "helvetica"
    G.edge_attr["fontname"] = "helvetica"
    if voir_variables:
        argus = {"shape": "circle"}
    else:
        argus = dict()
    dico_triplets = dict()
    for iden in amr.variables:
        triplet = (":instance", iden, amr.nodes[iden])
        dico_triplets[triplet] = []
        label = amr.nodes[iden]
        argus["id"] = prefixe+iden
        if label in propbank_frames_dictionary:
            tooltip = propbank_frames_dictionary[label].replace("\t","\n")
            tooltip = iden + "\n" + tooltip
            G.add_node(iden, label="" if voir_variables else label,
                       style="filled",
                       fillcolor="white",
                       tooltip = tooltip,
                       **argus
                    )
        else:
            G.add_node(iden, label="" if voir_variables else label,
                       style="filled",
                       fillcolor="white",
                       **argus)
        dico_triplets[triplet].append(prefixe+iden)
        if voir_variables:
            id2 = "%s_concept_%s"%(prefixe, iden)
            for C in clusters:
                if iden in C:
                    C.append(id2)
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
        if id2 in amr.variables:
            triplet = (rol, id1, id2)
            idarete = "%s→%s"%(prefixe+id1,prefixe+id2)
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
        
    G.layout(prog='dot')
    return G, prefixe, dico_triplets
    

def convert2graph(amr, variables=False, voir_racine=False, reverse_roles=False, return_edges=False, clusters=[]):
    if type(amr) is str:
        reader = AMR_Reader()
        amr = reader.loads(amr)
    f, prefixe, dico_triplets = drawAMR(amr, variables, voir_racine, reverse_roles, clusters)
    f = f.draw(path=None, format="svg")
    
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
    
    if return_edges:
        return (arbo, prefixe, dico_triplets)
    else:
        return arbo
 

def main():
    global recherche_amr
    nom_fichier = "./AMR_et_graphes_phrases.txt"
    print("Ouverture du fichier %s pour dresser la table")
    recherche_amr = RECHERCHE_AMR(nom_fichier)
    print("Table dressée.")
    ServeurReq.add_get(GET_INDEX, GET_VIVAGRAPH, GET_VIVAGRAPH_FACTORY, GET_LDC_2020_T02)
    ServeurReq.add_post(POST_CHOIX_VISU)
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
    amr = AMR_Reader().loads(AMRtxt2)
    drawAMR(amr)


if __name__ == "__main__":
    main()
    