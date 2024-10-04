import os
from make_dataset import FusionElimination as FILT, AligDataset
from report_generator import HTML_REPORT

def batch_LM():
    nom_rapport="Rapport_modèle_linéaire"
    nom_dataset = "dataset_pipo"
    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Dataset : %s"%nom_dataset)
        R.texte("Effectifs avant filtrage :")
        R.table(relations=["un", "deux", "trois"], effectifs=[1,2,3])
