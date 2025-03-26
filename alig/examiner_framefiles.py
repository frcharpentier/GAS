import os
from lxml import etree as ET
import tqdm
import re
import csv
import json
import random
from outils.traiter_AMR_adjectifs import generateur_candidats_txt
from dependencies import PROPBANK_TO_VERBATLAS, UMR_91_ROLESETS, PROPBANK_DIRECTORY, AMR_UMR_91_ROLESETS_XML

def faire_roles_en_91():
    with open("C:/Users/fcharpentier/Documents/Boulot/visuAMR/propbank-frames/AMR-UMR-91-rolesets.xml",
              "r", encoding="utf-8") as F:
        arbo = ET.parse(F)
    with open("./UMR_91_rolesets.tsv", "w", encoding="utf-8") as F:
        rolesets = arbo.iterfind("//roleset")
        for roleset in rolesets:
            iden = roleset.attrib["id"]
            roles = roleset.findall("./roles/role")
            usages = roleset.findall("./usagenotes/usage")
            usages = [u.attrib["inuse"] for u in usages if u.attrib["resource"] == "AMR"]
            ARGn = [r.attrib["n"] for r in roles]
            descr = [r.attrib["f"] if (not "descr" in r.attrib) else r.attrib["descr"] for r in roles]
            lig = "\t".join([ "A%s>[%s]"%(A,d) for A,d in zip(ARGn, descr)])
            lig = "%s>va:????"%(iden) + "\t" + lig
            if all(u=="-" for u in usages):
                lig = "### " + lig
            print(lig, file=F)



#faire_roles_en_91()

class EXPLICITATION_AMR:
    description_roles = {
        "EXT":  "extent",
        "LOC":  "location",
        "DIR":  "destination", # "direction",
        "NEG":  "negation",#  (not in PREDITOR)
        "MOD":  "modif", #(general modification)
        "ADV":  "advrb_mod", #(adverbial modification)
        "MNR":  "manner",
        "PRD":  "attribute", # "predication" ?,
        "REC":  "reciprocal", #(eg herself, etc)
        "TMP":  "time", #"temporal",
        "PRP":  "purpose",
        "PNC":  "PNC",# (purpose_no_cause : no longer used)
        "CAU":  "cause",
        #"CXN":  "constructional_pattern",# (adjectival comparative marker)
        "ADJ":  "adjectival", #(nouns only)
        "COM":  "comitative",
        "CXN":  "constructional",# (for "compared to what" constructions for adjectivals)
        "DIS":  "discourse",
        "DSP":  "dir_speech", #(direct_speech)
        "GOL":  "goal",
        "PAG":  "agent", #(function tag for arg1)
        "PPT":  "patient", #(function tag for arg1)
        "RCL":  "rel_clause", #(no longer used)
        "SLC":  "SLC", #(Selectional Constraint Link)
        "VSP":  "verb_sp", #(Verb specific : function tag for numbered arguments)
        "LVB":  "lt_verb", #(Light verb : for nouns only)
        
        "SE1":  "SE1", #(UNVERIFIED)
        "SE2":  "SE2", #(UNVERIFIED)    
        "SE3":  "SE3", #(UNVERIFIED)
        "SE4":  "SE4", #(UNVERIFIED)
        "SE5":  "SE5", #(UNVERIFIED)    
        "SE6":  "SE6", #(UNVERIFIED) 

        "ANC":  "anchor",
        "ANC1": "anchor_1",
        "ANC2": "anchor_2",
        
        "ANG": "angle",

        "AXS":  "axis",
        "AXSp": "perp_axis",
        "AXSc": "cent_axis",
        "AXS1": "axis_1",
        "AXS2": "axis_2",

        "WHL":  "whole",
        "SEQ":  "sequence",
        "PSN":  "position"  
    }

    def __init__(self, fichier=None):
        if fichier:
            with open(fichier, "r", encoding="utf-8") as F:
                self.dicFrames = json.load(F)
        else:
            self.dicFrames = None
        self.dicAMRadj = None

    def expliciter(self, s, r, t):
        if r.startswith(":ARG") or r.startswith("?ARG"):
            reverse = (r.endswith("-of"))
            ND = t if reverse else s
            fff = re.search("ARG(\d+)", r[1:])
            if fff:
                ARGn = int(fff[1])
                fff = True
            else:
                ARGn = -1
                fff = False
            if ND in self.dicFrames:
                if fff:
                    roles = self.dicFrames[ND]["ARGn"]
                    lenr = len(roles)
                    if 0 <= ARGn < lenr :
                        fnd = True
                        numeriq = True
                    else:
                        fnd = False
                else:
                    ARGn = r[1:]
                    roles = self.dicFrames[ND]["special"]
                    if ARGn in roles:
                        fnd = True
                        numeriq = False
                    else:
                        fnd = False
                if fnd:
                    role = roles[ARGn].upper()
                    if re.match("ARG\d+$", role):
                        role = "?"+role
                        if reverse:
                            role += "-of"
                        return role
                    if numeriq:
                        assert 0 <= ARGn < 10
                        role = ":>%s(%d)"%(role, ARGn)
                    else:
                        role = ":>" + role
                    if reverse:
                        role += "-of"
                    return role
                else:
                    return "?" + r[1:]
            elif self.dicAMRadj and ND in self.dicAMRadj:
                argus = self.dicAMRadj[ND]
                try:
                    if reverse:
                        assert r[1:-3] == argus["domain"]
                    else:
                        assert r[1:] == argus["domain"]
                except:
                    return "?" + r[1:]
                else:
                    if fff:
                        if reverse:
                            return ":>THEME(%d)-of"%ARGn #3
                        else:
                            return ":>THEME(%d)"%ARGn    #3
                    elif reverse:
                        #return ":mod" #1
                        return ":>THEME-of" #3
                    else:
                        #return ":domain" #1
                        #return ":mod-of" #2
                        return ":>THEME"  #3
            else:
                return "?" + r[1:]
        return r

    def expliciter_AMR(self, amr):

        edges_t = []
        transfo = False
        for s,r,t in amr.edges:
            if r.startswith(":ARG")  or r.startswith("?ARG"):
                SS = amr.nodes[amr.isi_node_mapping[s]]
                TT = amr.nodes[amr.isi_node_mapping[t]]
                RR = self.expliciter(SS,r,TT)
                if RR != r:
                    transfo = True
                    r = RR
            edges_t.append((s,r,t))
        if transfo:
            amr.edges = edges_t
        return amr
    
    @staticmethod
    def transfo_pb2va_tsv(fichiers = [PROPBANK_TO_VERBATLAS, UMR_91_ROLESETS], fichierAdj="./adjectifs_AMR.txt"):
        prems = True
        resu = {}
        dicoAdj = {}
        f = lambda x: int(x[2]) if x[2] is not None else -1
        if fichierAdj is not None:
            for adj, numero, argus in generateur_candidats_txt(fichierAdj):
                k = "%s-%02d"%(adj, numero)
                dicoAdj[k] = argus
        for fichier in fichiers:
            print("ouverture de %s"%fichier)
            with open(fichier, "r", encoding="utf-8") as F:
                for ligne in F:
                    if prems:
                        prems=False
                        continue
                    ligne = ligne.strip()
                    if ligne.startswith("#"):
                        continue
                    if ligne.startswith("===FIN"):
                        break
                    speciaux = dict()
                    frame, *argus = ligne.split("\t")
                    pbframe, vaframe = frame.split(">")

                    fnd = re.search("^(.+)\.(\d+)$", pbframe)
                    if fnd:
                        pbframe = fnd[1] + "-" + fnd[2]

                    argus = [kv.split(">", maxsplit=1) for kv in argus]
                    
                    ARGn = [(re.search("^(A(\d+)|A.+)$", k),v) for k,v in argus]
                    if any(k is None for k,v in ARGn):
                        print(pbframe)
                    for k, v in ARGn:
                        if k[2] == None:
                            speciaux[k[1]] = v
                    ARGn = [(f(k),v) for k,v in ARGn if k[2] is not None]
                    if len(ARGn) == 0:
                        resu[pbframe] = {"vaframe": vaframe}
                        continue
                    maxi = max(k for k,v in ARGn)
                    mini = min(k for k,v in ARGn)
                    assert mini in (0,1)
                    argus = ["ARG%d"%x for x in range(1+maxi)]
                    for k,v in ARGn:
                        argus[k] = v
                    resu[pbframe] = {"vaframe" : vaframe,
                                    "ARGn" : argus,
                                    "special" : speciaux}
                    (vaframe,) + tuple(argus)
        return resu, dicoAdj

    @staticmethod
    def make_json_from_propbank_github(repertoire=None, fichier_91=AMR_UMR_91_ROLESETS_XML, fichier_corr = None, fichier_out=None):
        dico = dict()
        if fichier_corr == None:
            dico_corr = dict()
        else:
            with open(fichier_corr, "r", encoding="utf-8") as F:
                dico_corr = json.load(F)
        if repertoire == None:
            #repertoire = "C:/Users/fcharpentier/Documents/Boulot/visuAMR/propbank-frames"
            repertoire = PROPBANK_DIRECTORY
        fichiers_xml = [os.path.abspath(os.path.join(repertoire, "frames", f)) for f in os.listdir(os.path.join(repertoire, "frames"))]
        fichiers_xml.append(os.path.abspath(os.path.join(repertoire, fichier_91)))
        fichiers_xml = [X for X in fichiers_xml if X.endswith(".xml")]
        for fic in tqdm.tqdm(fichiers_xml):
            arbo = ET.parse(fic)
            predicates = arbo.iterfind("./predicate")
            for pred in predicates:
                rolesets = pred.iterfind("./roleset")
                for rs in rolesets:
                    usageAMR = [X.attrib["inuse"] for X in rs.findall("./usagenotes/usage[@resource='AMR']")]
                    if not "+" in usageAMR:
                        continue
                    dicors = dict()
                    iden = rs.attrib["id"]
                    fnd = re.search("^(.+)\.(\d+)$", iden)
                    if fnd:
                        iden = fnd[1] + "-" + fnd[2]
                    dicors["name"] = rs.attrib["name"]
                    dicors["fichier"] = os.path.basename(fic)
                    partes_ort = [X.attrib["pos"] for X in rs.findall("./aliases/alias")]
                    roles = rs.findall("./roles/role")
                    ARGn = [R.attrib["n"] for R in roles]
                    ARGn = [-1 if n in "mM" else int(n) for n in ARGn]
                    if len(ARGn) == 0:
                        dicors["roles"] = []
                        dico[iden] = dicors
                        continue
                    VN_roles = []
                    maxARGn = max(ARGn)
                    if -1 in ARGn:
                        il_y_a_ARGm = True
                        maxARGn += 1
                        ARGn = [maxARGn if n == -1 else n for n in ARGn]
                    
                    dicors["roles_PB"] = ["ARG%d"%ii for ii in range(1+maxARGn)]
                    dicors["descr"] = ["" for _ in range(1+maxARGn)]
                    if not 0 in ARGn:
                        dicors["roles_PB"][0] = None
                        dicors["descr"][0] = None
                    for i, R in zip(ARGn, roles):
                        if "descr" in R.attrib:
                            dicors["descr"][i] = R.attrib["descr"]
                        if "f" in R.attrib:
                            attribf = R.attrib["f"]
                            if len(attribf) > 0:
                                dicors["roles_PB"][i] = attribf.upper()
                            elif i == 0 and "agent" in descr.lower():
                                dicors["roles_PB"][i] = "PAG"
                        for vnrole in R.findall("./rolelinks/rolelink[@resource='VerbNet']"):
                            vnclass = vnrole.attrib["class"]
                            version = vnrole.attrib["version"]
                            if version == "UNK":
                                version = (0,)
                            else:
                                if version.startswith("verbnet"):
                                    version = version[7:]
                                version = tuple(int(x) for x in version.split("."))
                            rel = vnrole.text
                            if len(rel) == 0:
                                print(iden, os.path.basename(fic), i)
                            #clef = ((version + (random.randint(1,100),)), vnclass)
                            clef = version + (vnclass,)
                            truc = [X[0][:-1] for X in VN_roles]
                            if not clef in truc:
                                rlst = ["ARG%d"%ii for ii in range(1+maxARGn)]
                                if not 0 in ARGn:
                                    rlst[0] = None
                                VN_roles.append([clef + (random.randint(1,100),), rlst])
                                idxVN = -1
                            else:
                                idxVN = truc.index(clef)
                            VN_roles[idxVN][1][i] = rel
                    for K, roles in VN_roles:
                        K = "roles_" + K[-2] + "_v" + ".".join(str(x) for x in K[:-2])
                        dicors[K] = roles
                    if len(VN_roles) == 0:
                        if (len(ARGn) in (1,2)) and (not ("v" in partes_ort)):
                            dicors["roles"] = ["ARG%d"%ii for ii in range(1+maxARGn)]
                            if not 0 in ARGn:
                                dicors["roles"][0] = None
                            if len(ARGn) == 1:
                                dicors["roles"][ARGn[0]] = "theme"
                            else:
                                dicors["roles"][min(ARGn)] = "pivot"
                                dicors["roles"][max(ARGn)] = "theme"
                        else:
                            dicors["roles"] = dicors["roles_PB"][:]
                    else:
                        roles = max(VN_roles, key=lambda x: x[0][:-2] + (x[0][-1],))[1]
                        dicors["roles"] = roles[:]
                    if iden in dico_corr:
                        corr = dico_corr[iden]
                        if type(corr) in (list, dict):
                            if type(corr) is dict:
                                for i in range(len(dicors["roles"])):
                                    if str(i) in corr:
                                        dicors["roles"][i] = corr[str(i)]
                            else:
                                for n, rel in enumerate(corr):
                                    dicors["roles"][n] = rel
                    dico[iden] = dicors

        if fichier_out:
            with open(fichier_out, "w", encoding="utf-8") as F:
                json.dump(dico, F)
        return dico


def transfo_frame_arg_descr():
    resu = []
    fichier = "C:/Users/fcharpentier/Documents/Boulot/visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/frames"
    fichier = os.path.join(fichier, "propbank-amr-frame-arg-descr.txt")
    pattern = "ARG(\d)\s*:\s+"
    maxi = 0
    with open(fichier, "r", encoding="utf-8") as F:
        for ligne in F:
            courant = []
            ligne = ligne.strip()
            #print(ligne)
            sepa = re.split(pattern, ligne)
            sepa = [x.strip() for x in sepa]
            frame, descr = sepa[0], sepa[1:]
            assert "-" in frame
            ridx = frame.rindex("-")
            frame, numero = frame[0:ridx], frame[ridx+1:]

            if len(descr) == 0:
                lig = [frame, numero]
            else:
                if len(descr) %2 != 0:
                    descr.append("")
                descr = [(int(descr[i]), descr[i+1]) for i in range(0,len(descr),2)]
                maxN = max(d[0] for d in descr)
                maxi = max(maxi, maxN)
                descr2 = [""] * (1+maxN)
                for i, d in descr:
                    descr2[i] = d
            
                lig = [frame, numero] + descr2
            resu.append(lig)
    titre = ["FRAME", "NUM"] + ["ARG%d"%i for i in range(1+maxi)]
    resu.insert(0, titre)
    #print(resu)
    #print(len(resu))
    with open("propbank-amr-frame-arg-descr.csv", "w", encoding="utf-8", newline='') as F:
        wrtr = csv.writer(F, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        for lig in resu:
            wrtr.writerow(lig)
    print("TERMINÉ.")



            
            


#transfo_frame_arg_descr()
if __name__ == "__main__":
    #make_json_from_framefiles(None, "./RoleFrames.json")
    #EXPLICITATION_AMR.make_json_from_propbank_github()
    transfo_pb2va_tsv()