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
                    fnd = re.search("^(:>\D+)\(\d+\)$", r)
                    if fnd:
                        r = fnd[1]
                dico[r] += 1
    return dico

def liste_roles(nom_fichier):
    chaine = sourcer_fichier_txt(nom_fichier)
    dico = cataloguer_roles(chaine)
    cles = [k for k in dico]
    classer = lambda x: x[2:].lower() if x.startswith(":>") else (x[1:].lower() if x.startswith(":") else x.lower())
    cles.sort(key = classer)
    for k in cles:
        print("%s: (%d,)"%(repr(k), dico[k]))
    #print(dict(dico))

#Le dico suivant a été fait en exécutant la fonction liste_roles
dico_roles = OrderedDict( [ (':accompanier', (569,)),
    (':age', (1135,)),
    (':>AGENT', (125945,)),
    (':>ARG0', (2,None)),
    (':>ARG2', (3,None)),
    (':>ASSET', (1283,)),
    (':>ATTRIBUTE', (14329,)),
    (':beneficiary', (1371,)),
    (':>BENEFICIARY', (7998,)),
    (':>CAUSE', (1573,)),
    (':>CO-AGENT', (2030,)),
    (':>CO-PATIENT', (513,)),
    (':>CO-THEME', (9280,)),
    (':>CONCESSION', (1140,)),
    (':concession', (1713,)),
    (':condition', (3670,)),
    (':>CONDITION', (1562,)),
    (':conj-as-if', (24,)),
    (':consist-of', (1386,)),
    (':day', (1,None)),
    (':dayperiod', (38,None)),
    (':decade', (1,None)),
    (':degree', (5623,)),
    (':>DESTINATION', (6178,)),
    (':destination', (575,)),
    (':direction', (1188,)),
    (':duration', (2922,)),
    (':example', (2914,)),
    (':>EXPERIENCER', (15572,)),
    (':extent', (455,)),
    (':>EXTENT', (1877,)),
    (':frequency', (1380,)),
    (':>FREQUENCY', (23,)),
    (':>GOAL', (7218,)),
    (':>IDIOM', (744,)),
    (':>INSTRUMENT', (2714,)),
    (':instrument', (787,)),
    (':li', (852,None)),
    (':location', (21913,)),
    (':>LOCATION', (5104,)),
    (':manner', (8095,)),
    (':>MANNER', (101,)),
    (':>MATERIAL', (322,)),
    (':medium', (2055,)),
    (':mod', (121338,)),
    (':>MOD', (413,)),
    (':mode', (1455,)),
    (':month', (34,)),
    (':name', (4569,)),
    (':op1', (18,None)),
    (':ord', (1254,)),
    (':part', (6322,)),
    (':>PART', (141,)),
    (':path', (401,)),
    (':>PATIENT', (29132,)),
    (':polarity', (18501,None)),
    (':>POLARITY', (96,None)),
    (':polite', (203,None)),
    (':poss', (16093,)),
    (':prep-against', (152,None)),
    (':prep-along-with', (9,None)),
    (':prep-amid', (5,None)),
    (':prep-among', (29,None)),
    (':prep-as', (178,None)),
    (':prep-at', (8,None)),
    (':prep-by', (21,None)),
    (':prep-for', (114,None)),
    (':prep-from', (31,None)),
    (':prep-in', (224,None)),
    (':prep-in-addition-to', (7,None)),
    (':prep-into', (24,None)),
    (':prep-on', (137,None)),
    (':prep-on-behalf-of', (65,None)),
    (':prep-out-of', (1,None)),
    (':prep-to', (140,None)),
    (':prep-toward', (8,None)),
    (':prep-under', (138,None)),
    (':prep-with', (295,None)),
    (':prep-without', (41,None)),
    (':>PRODUCT', (2754,)),
    (':purpose', (5417,)),
    (':>PURPOSE', (487,)),
    (':quant', (13804,)),
    (':range', (54,)),
    (':>RECIPIENT', (9893,)),
    (':>RESULT', (14792,)),
    (':scale', (101,)),
    (':>SOURCE', (5986,)),
    (':source', (2604,)),
    (':>STIMULUS', (11294,)),
    (':subevent', (350,)),
    (':subset', (3,)),
    (':>THEME', (121792,)),
    (':time', (30960,)),
    (':>TIME', (419,)),
    (':timezone', (326,None)),
    (':topic', (5436,)),
    (':>TOPIC', (14119,)),
    (':unit', (107,None)),
    (':>VALUE', (2168,)),
    (':value', (340,)),
    (':weekday', (26,None)),
    (':year', (9,None)),
    ('{and_or}', (2358,)),
    ('{and}', (140798,)),
    ('{groupe}', (1117630,)),
    ('{idem}', (41628,)),
    ('{inter}', (1214,)),
    ('{or}', (11339,)),
    ('{syntax}', (10258,))  ]
)

def traiter_dico_roles():
    dico = defaultdict(lambda :([], 0))
    for k, (n, id) in dico_roles.items():
        lis, cumul = dico[id]
        lis.append(k)
        cumul += n
        dico[id] = (lis, cumul)
    return dico