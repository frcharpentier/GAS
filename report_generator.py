from bs4 import BeautifulSoup
import os
import sys
import random
from matplotlib import pyplot as plt
import numpy as np

fichier_vide = """<!DOCTYPE html>
<html>
  <head>
    <title>Report</title>
    <meta charset="UTF-8" />
	
    <style type="text/css" media="screen">
table, tr, td, th {
  border-collapse: collapse;
  border: 1px solid;
}
    </style>
    
    <!--
    <script src="https://d3js.org/d3.v7.min.js"></script>
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    -->
</head>
  <body>
  <h1>REPORT</h1>
  </body>
</html>
"""

class HTML_TABLE():
    def __init__(self, lignes = False, **kwargs):   
        cols = []
        self.liste_cols = []
        self.index = []
        self.gardeG = False
        if lignes:
            self.gardeG = True
            self.gardeH = False
        else:
            self.gardeG = False
            self.gardeH = True
        argus = {k: v if type(v) is list else [v] for k,v in kwargs.items() if k != "lignes"}
        lgmax = max(len(v) for v in argus.values())
        for k in argus.keys():
            if k == "index":
                liste_lbls = [str(x)[:30] for x in argus[k]]
                if len(liste_lbls) < lgmax:
                    liste_lbls += [""]*(lgmax-len(self.index))
                if lignes:
                    self.gardeH = True
                    self.liste_cols = liste_lbls
                else:
                    self.gardeG = True
                    self.index = liste_lbls
            else:
                if lignes:
                    self.index.append(k)
                else:
                    self.liste_cols.append(k)
                colonne = [str(x)[:30] for x in argus[k]]
                if len(colonne) < lgmax:
                    colonne += [""]*(lgmax-len(self.index))
                cols.append(colonne)
        if lignes:
            self.rangees = cols
        else:
            self.rangees = [x for x in zip(*cols)]
        #print(self.liste_cols)
        #print(self.rangees)
        #print("True" if self.gardeH else "False")
              
    def rendu(self, S):
        table = S.new_tag("table")
        
        if self.gardeH:
            rangee = S.new_tag("tr")
            if self.gardeG:
                rangee.append(S.new_tag("th"))
            for k in self.liste_cols:
                case = S.new_tag("th")
                case.append(k)
                rangee.append(case)
            table.append(rangee)
            table.append("\n")
            
        I = 0
        for lig in self.rangees:
            #print(lig)
            rangee = S.new_tag("tr")
            if self.gardeG:
                case = S.new_tag("th")
                case.append(self.index[I])
                I += 1
                rangee.append(case)
            for val in lig:
                case = S.new_tag("td")
                case.append(val)
                rangee.append(case)
            table.append(rangee)
            table.append("\n")
        return table
        
    #def rendu_T(self):
    #    colsT = [list(x) for x in zip(self.cols)
            
        
class NOM_FICHIER():
    def __init__(self, docu, extension):
        while extension.startswith("."):
            extension = extension[1:]
        self.format_image = extension
        trouve = False
        while not trouve:
            self.basename = "PJ_%d.%s"%(random.randint(1,1000000), extension)
            trouve = not os.path.isfile(os.path.join(docu.fullfigdirname, self.basename))
        self.fullname = os.path.join(docu.fullfigdirname, self.basename)
        self.figdirname = docu.figdirname
        self.S = docu.S
        
    def __enter__(self):
        return self
        
        
    def __exit__(self, typ, value, traceback):
        #print("Sortie du contexte")
        source = self.figdirname + "/" + self.basename
        image = self.S.new_tag("img", src = source)
        self.S.body.append(image)
        self.S.body.append("\n(Image stockée dans %s)\n"%source)
        if typ != None:
            return False
        return True 
    

class RESSOURCE():
    def __init__(self, docu, extension="bin"):
        while extension.startswith("."):
            extension = extension[1:]
        self.format_image = extension
        trouve = False
        while not trouve:
            self.basename = "RES_%d.%s"%(random.randint(1,1000000), extension)
            trouve = not os.path.isfile(os.path.join(docu.fullfigdirname, self.basename))
        self.fullname = os.path.join(docu.fullfigdirname, self.basename)
        self.figdirname = docu.figdirname
        self.S = docu.S
        
        
    def __enter__(self):
        return self
        
        
    def __exit__(self, typ, value, traceback):
        #print("Sortie du contexte")
        source = self.figdirname + "/" + self.basename
        self.S.body.append("\n(Ressource stockée dans %s)\n"%source)
        if typ != None:
            return False
        return True 

class HTML_IMAGE():
    def __init__(self, docu, extension):
        while extension.startswith("."):
            extension = extension[1:]
        self.format_image = extension
        trouve = False
        while not trouve:
            self.basename = "PJ_%d.%s"%(random.randint(1,1000000), extension)
            trouve = not os.path.isfile(os.path.join(docu.fullfigdirname, self.basename))
        self.fullname = os.path.join(docu.fullfigdirname, self.basename)
        self.figdirname = docu.figdirname
        self.S = docu.S
        
    def __enter__(self):
        self.F = open(self.fullname, "wb")
        return self
        
        
    def write(self, octets):
        self.F.write(octets)
        
        
    def __exit__(self, typ, value, traceback):
        #print("Sortie du contexte")
        self.F.close()
        source = self.figdirname + "/" + self.basename
        image = self.S.new_tag("img", src = source)
        self.S.body.append(image)
        self.S.body.append("\n(Image stockée dans %s)\n"%source)
        if typ != None:
            return False
        return True

class CLASSE_SKLEARN:
    def __init__(self, classe, report):
        self.classe = classe
        self.R = report

    def __call__(self, **kwargs):
        self.R.titre("Paramètres d’instanciation",2)
        self.R.table(
            **kwargs,
            colonnes=False
        )
        self.R.flush()
        instance = self.classe(**kwargs)
        return instance


class HTML_REPORT:
    def __init__(self, fichier):
        #print("init", file=sys.stderr)
        self.dirname = os.path.dirname(fichier)
        basename = os.path.basename(fichier)
        gauche, _ = os.path.splitext(basename)
        self.basename = gauche + ".html"
        self.fullname = os.path.join(self.dirname, self.basename)
        self.figdirname = gauche + "_PJ"
        self.fullfigdirname = os.path.join(self.dirname, self.figdirname)
        if os.path.exists(self.fullfigdirname):
            assert os.path.isdir(self.fullfigdirname)
        else:
            os.mkdir(self.fullfigdirname)
        if not os.path.isfile(self.fullname):
            with open(self.fullname, "w", encoding="utf-8") as F:
                F.write(fichier_vide)
        self.S = None
          
    def open(self):
        #print("open", file=sys.stderr)
        with open(self.fullname, "r", encoding="utf-8") as F:
            html_code = F.read()
        self.S = BeautifulSoup(html_code, "html.parser")
        
    def flush(self):
        if self.S == None:
            return
        #print("Écriture", file=sys.stderr)
        #print(str(self.S), file=sys.stderr)
        with open(self.fullname, "w",encoding="utf-8") as F:
            F.write(str(self.S))
            
    def __enter__(self):
        #print("Entrée dans le contexte", file=sys.stderr)
        self.open()
        return self
        
    def __exit__(self, typ, value, traceback):
        #print("Sortie du contexte")
        self.flush()
        if typ != None:
            return False
        return True
            
    def h(self, txt, niveau=1):
        assert 1 <= niveau <= 6
        titre = self.S.new_tag("h%d"%niveau)
        #print("h%d"%niveau, file=sys.stderr)
        titre.string = txt
        self.S.body.append(titre)
        self.S.body.append("\n")
        
    def titre(self, txt, niveau=1):
        self.h(txt, niveau)
        
    def text(self, txt):
        #print("txt", file=sys.stderr)
        par = self.S.new_tag("p")
        txt = txt.split("\n")
        if len(txt) == 1:
            par.append(txt[0])
        else:
            par.append(txt[0])
            for txt in txt[1:]:
                par.append(self.S.new_tag("br"))
                par.append("\n")
                par.append(txt)
        self.S.body.append(par)
        self.S.body.append("\n")
        
    def texte(self, txt):
        self.text(txt)
        
    def hr(self):
        #print("hr", file=sys.stderr)
        self.S.body.append(self.S.new_tag("hr"))
        self.S.body.append("\n")
        
    def ligne(self):
        self.hr()
    
    def line(self):
        self.hr()

    def skl(self, classe):
        return CLASSE_SKLEARN(classe, self)

    def new_img_with_format(self, format):
        return NOM_FICHIER(self, format)
    
    def new_ressource(self):
        return RESSOURCE(self)
            
    def new_img(self, extension):
        return HTML_IMAGE(self, extension)
        
    def new_image(self, extension):
        return HTML_IMAGE(self, extension)
        
    def img(self, extension):
        return HTML_IMAGE(self, extension)
        
    def image(self, extension):
        return HTML_IMAGE(self, extension)
        
    def table(self, **kwargs):
        argus = dict()
        lignes = False
        for k, v in kwargs.items():
            if k in ("lignes", "lines"):
                assert not any(X in ["columns", "colonnes"] for X in kwargs)
                lignes = False
                argus["index"] = v
            elif k in ("columns", "colonnes"):
                assert not any(X in ["lignes", "lines"] for X in kwargs)
                lignes = True
                if not v in [False, "", None]:
                    argus["index"] = v
            else:
                argus[k] = v
        tbl = HTML_TABLE(lignes=lignes, **argus)
                   
        self.S.body.append(tbl.rendu(self.S))
        self.S.body.append("\n")
            
            
def main():
    with HTML_REPORT("./rapport_essai.html") as R:
        #R.titre("Titre", 2)
        #R.text("Voici du texte\nSur plusieurs lignes.")
        #R.ligne()
        #R.titre("Figure", 2)
        #X = np.arange(0, 6.3, 0.01)
        #Y = np.sqrt(X)
        #fig, axes = plt.subplots()
        #axes.plot(X,Y)
        #with R.new_img("png") as F:
        #    fig.savefig(F)
        
        
        R.ligne()
        R.titre("Tableau", 2)
        R.table(colonne_A = [10,20,30], colonne_B = [10,20,40])
        R.texte("OK")
        R.table(lignes = ["L1", "L2", "L3"], colonne_A = [10,20,30], colonne_B = [10,20,40])
        R.texte("OK")
        R.table(colonnes = False, lig_A = [10,20,30], lig_B = [10,20,40])
        R.texte("OK")
        R.table(colonnes = ["C1", "C2", "C3"], lig_A = [10,20,30], lig_B = [10,20,40])
        R.texte("OK")
        R.table(colonnes = False, truc1 = 0.28, truc2=0.004)
        R.text("Fin du rapport")
        
    print("Terminé")
    
if __name__ == "__main__":
    main()