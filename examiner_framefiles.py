import os
from lxml import etree as ET
import tqdm
import re
import csv


description_roles = {
    "EXT":  "extent",
    "LOC":  "location",
    "DIR":  "direction",
    "NEG":  "negation",#  (not in PREDITOR)
    "MOD":  "modif", #(general modification)
    "ADV":  "advrb_mod", #(adverbial modification)
    "MNR":  "manner",
    "PRD":  "predication",
    "REC":  "reciprocal", #(eg herself, etc)
    "TMP":  "temporal",
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
}

def examiner_framefiles(repertoire=None):
    dico = dict()
    if repertoire == None:
        repertoire = "C:/Users/fcharpentier/Documents/Boulot/visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/frames/propbank-amr-frames-xml-2018-01-25"
    fichiers_xml = [os.path.abspath(os.path.join(repertoire, f)) for f in os.listdir(repertoire)]
    for fic in tqdm.tqdm(fichiers_xml):
        #print(os.path.basename(fic))
        arbo = ET.parse(fic)
        rolesets = arbo.iterfind("//roleset")
        for rs in rolesets:
            dicors = dict()
            iden = rs.attrib["id"]
            dicors["name"] = rs.attrib["name"]
            dicors["fichier"] = os.path.basename(fic)
            roles = rs.findall(".//role")
            ARGn = [int(R.attrib["n"]) for R in roles]
            maxARGn = max(ARGn) 
            dicors["roles"] = ["ARGxx"] * (1+maxARGn)
            for i, R in zip(ARGn, roles):
                if "f" in R.attrib:
                    dicors["roles"][i] = R.attrib["f"]
                for vnrole in R.findall("vnrole"):
                    if "vncls" in vnrole.attrib and "vntheta" in vnrole.attrib:
                        vncls = "roles_"+ vnrole.attrib["vncls"]
                        vntheta = vnrole.attrib["vntheta"]
                        if not vncls in dicors:
                            dicors[vncls] = ["ARGxx"] * (1+maxARGn)
                        dicors[vncls][i] = vntheta
            dico[iden] = dicors
            
    return dico

dico = examiner_framefiles()

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

transfo_frame_arg_descr()
    