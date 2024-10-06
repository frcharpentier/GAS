from enchainables import MAILLON
from collections import OrderedDict, defaultdict
import json
import re

@MAILLON
def sourcer_fichier_txt(SOURCE, nom_fichier):
    lbl_modname = "# ::model_name "
    lbl_id = "# ::id "
    etat = 0
    model_name=None
    with open(nom_fichier, "r", encoding="utf-8") as F:
        depart = True
        for ligne in F:
            ligne = ligne.strip()
            if depart and ligne.startswith(lbl_modname):
                model_name = ligne[len(lbl_modname):]
            elif etat == 0:
                if ligne.startswith(lbl_id):
                    ligne = ligne[len(lbl_id):]
                    idSNT = ligne.split()[0]
                    etat = 1
            elif etat == 1:
                if ligne.startswith('{"tokens": '):
                    jsn = json.loads(ligne)
                    jsn["idSNT"] = idSNT
                    jsn["model_name"] = model_name
                    yield jsn
                    etat  = 0
            depart = False


def cataloguer_roles(source):
    dico = defaultdict(lambda: 0)
    for jsn in source:
        for s, r, c in jsn["aretes"]:
            if not r.startswith("?"):
                if r.startswith(":>"):
                    fnd = re.search("^(:>\D+)\((\d+)\)$", r)
                    if fnd:
                        r = fnd[1]
                        ARGn = fnd[2]
                        dico[":ARG%s"%ARGn] += 1
                dico[r] += 1
    return dico

def liste_roles(nom_fichier):
    chaine = sourcer_fichier_txt(nom_fichier)
    dico = cataloguer_roles(chaine)
    cles = [k for k in dico]
    classer = lambda x: x[2:].lower() if x.startswith(":>") else (x[1:].lower() if x.startswith(":") else x.lower())
    cles.sort(key = classer)
    with open ("./liste_roles.py", "w", encoding="UTF-8") as F:
        print("from collections import OrderedDict, defaultdict", file=F)
        print(file=F)
        print("#Le dico suivant a été fait en exécutant la fonction liste_roles du module liste_tous_roles.py",
              file=F)
        print("dico_roles = OrderedDict( [", file=F)
        for k in cles:
            if not re.match(":ARG\d$", k):
                print("    (%s, %d),"%(repr(k), dico[k]), file=F)
        print("] )", file=F)
        print(file=F)
        print("#Le dico suivant a été fait en exécutant la fonction liste_roles du module liste_tous_roles.py",
              file=F)
        print("dico_ARGn = OrderedDict( [", file=F)
        ARGn = [k for k in cles if re.match(":ARG\d$", k)]
        for k in ARGn:
            print("    (%s, %d),"%(repr(k), dico[k]), file=F)
        print("] )", file=F)
    #print(dict(dico))



