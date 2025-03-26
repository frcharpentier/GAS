import os
import re
import csv
import re
from outils.enchainables import MAILLON
from lxml import etree as ET


@MAILLON
def generateur_candidats_txt(SOURCE, fichier):
    with open(fichier, "r", encoding="utf-8") as F:
        for ligne in F:
            ligne = ligne.strip()
            if ligne.startswith("#"):
                continue
            mots = ligne.split()
            amr_lbl = mots[0]
            mtch = re.match("([a-zA-Z\-]+)-(\d+)$", amr_lbl)
            if mtch is None:
                print("### " + ligne)
                assert not mtch is None
            mot = mtch[1]
            numero = int(mtch[2])
            argus = dict()
            pos_argus = [i for i, m in enumerate(mots) if m.startswith("ARGM-") or (not re.match("ARG\d+#?:", m) is None)]
            if len(pos_argus) == 0:
                continue
            if not pos_argus[0] == 1:
                print("### " + ligne)
                continue
            assert pos_argus[0] == 1
            while len(pos_argus) > 0:
                pos = pos_argus[-1]
                k = mots[pos]
                k = k[:-1] # ôter le côlon final.
                if k.endswith("#"):
                    k = k[:-1]
                    argus["domain"] = k
                descr = " ".join(mots[pos+1:])
                argus[k] = descr
                pos_argus.pop()
                mots = mots[:pos]
            yield mot, numero, argus

@MAILLON
def generateur_candidats_xml(SOURCE, repertoire):
    #repertoire = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\AMR_de_chez_LDC\\LDC_2020_T02\\data\\frames\\propbank-amr-frames-xml-2018-01-25"
    fichiers = [os.path.abspath(os.path.join(repertoire, f)) for f in os.listdir(repertoire)]
    for fichier in fichiers:
        with open(fichier, "r", encoding="utf-8") as F:
            arbo = ET.parse(F)
        rolesets = arbo.iterfind("//roleset")
        for roleset in rolesets:
            iden = roleset.attrib["id"]
            srch = re.search("^([a-zA-Z\-_]+)\.(\d+)", iden)
            if srch is None:
                #print("### ", fichier)
                #print(iden)
                continue
            assert not srch is None
            mot, numero = srch[1], srch[2]
            numero = int(numero)
            aliases = roleset.findall("./aliases/alias")
            if any((al.text==mot) and al.attrib["pos"] == "j" for al in aliases):
                roles = roleset.findall("./roles/role")
                ARGn = [r.attrib["n"] for r in roles]
                descr = [r.attrib["f"] if (not "descr" in r.attrib) else r.attrib["descr"] for r in roles]
                #lig = "\t".join([ "A%s>[%s]"%(A,d) for A,d in zip(ARGn, descr)])
                argus = {"ARG%s"%(A) : d for A,d in zip(ARGn, descr)}
                yield mot, numero, argus
            else:
                pass



@MAILLON
def filtrer_adjectifs(SOURCE):
    nlp = spacy.load("en_core_web_sm")
    for mot, numero, argus in SOURCE:
        rspcy = nlp(mot) 
        if rspcy[0].pos_ == "ADJ":
            yield mot, numero, argus

@MAILLON
def classer_adjectifs(SOURCE):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    for mot, numero, argus in SOURCE:
        if len(argus) == 1:
            argus = {k+"#": v for k,v in argus.items()}
        else:
            clefs = ["ARG0", "ARG1"] #, "ARG2"]
            clefs = [k for k in clefs if k in argus]
            sentences = ["the subject is %s"%argus[k] for k in clefs]
            clefs = [None] + clefs
            sentences = ["the subject is %s"%mot] + sentences
            #calcul des plongements
            embeddings = model.encode(sentences)
            #calcul des similarités
            similarities = model.similarity(embeddings, embeddings)
            sim = similarities[0,:].numpy().tolist()
            sim[0] = 0. #Éliminer la similarité de la question avec elle-même
            sim = [(i,s) for i, s in enumerate(sim)]
            idx_maxi = max(sim, key=lambda x: x[1])[0]
            clef = clefs[idx_maxi]
            argus[clef+"#"] = argus[clef]
            del argus[clef]
        yield mot, numero, argus





@MAILLON
def afficher(SOURCE):
    for mot, numero, argus in SOURCE:
        ligne = "%s-%02d  "%(mot, numero)
        clefs = [k for k in argus]
        clefs.sort()
        ligne = ligne + "  " + "  ".join("%s: %s"%(k, argus[k]) for k in clefs)
        print(ligne)


        








if __name__ == "__main__":
    import spacy
    from sentence_transformers import SentenceTransformer
    def traitement(fin = "C:/Users/fcharpentier/Documents/Boulot/visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/frames/propbank-amr-frames-xml-2018-01-25"):
        #generateur_candidats(None, None)
        chaine = generateur_candidats_xml(fin) >> filtrer_adjectifs()
        chaine = chaine >> classer_adjectifs() >> afficher()
        chaine.enchainer()
    traitement()