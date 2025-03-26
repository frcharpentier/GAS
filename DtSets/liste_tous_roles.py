from outils.enchainables import MAILLON
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


def cataloguer_roles(source, dico=None):
    if dico is None:
        dico = defaultdict(lambda: 0)
    assert isinstance(dico, defaultdict)
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

def liste_roles(dico=None, nom_fichier=None, fichier_out = None):
    if dico is None:
        assert type(nom_fichier) is str
        chaine = sourcer_fichier_txt(nom_fichier)
        dico = cataloguer_roles(chaine)
    else:
        assert nom_fichier is None
        assert isinstance(dico, dict)
    cles = [k for k in dico]
    classer = lambda x: x[2:].lower() if x.startswith(":>") else (x[1:].lower() if x.startswith(":") else x.lower())
    cles.sort(key = classer)
    cles1 = [k for k in cles if not re.match(":ARG\d$", k)]
    cles_ARGn = [k for k in cles if re.match(":ARG\d+$", k)]
    dico_roles = OrderedDict([(k, dico[k]) for k in cles1])
    dico_ARGn = OrderedDict([(k, dico[k]) for k in cles_ARGn])
    if fichier_out is not None:
        with open (fichier_out, "w", encoding="UTF-8") as F:
            print("from collections import OrderedDict, defaultdict", file=F)
            print(file=F)
            print("#Le dico suivant a été fait en exécutant la fonction liste_roles du module liste_tous_roles.py",
                file=F)
            print("dico_roles = OrderedDict( [", file=F)
            for k, v in cles1.items():
                print("    (%s, %d),"%(repr(k), v), file=F)
            print("] )", file=F)
            print(file=F)
            print("#Le dico suivant a été fait en exécutant la fonction liste_roles du module liste_tous_roles.py",
                file=F)
            print("dico_ARGn = OrderedDict( [", file=F)
            for k, v in dico_ARGn.items():
                print("    (%s, %d),"%(repr(k), v), file=F)
            print("] )", file=F)
    return dico_roles, dico_ARGn



