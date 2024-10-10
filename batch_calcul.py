import os
#from make_dataset import FusionElimination as FILT, AligDataset
import logging
from report_generator import HTML_REPORT, HTML_IMAGE
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Antisym



def plot_confusion_matrix(true_labels, pred_labels, label_names, imgfile=None):
    confusion_norm = confusion_matrix(true_labels,
                                      pred_labels, #.tolist(),
                                      labels=list(range(len(label_names))),
                                      normalize="true")
    confusion = confusion_matrix(true_labels,
                                 pred_labels, #.tolist(),
                                 labels=list(range(len(label_names)))
                                 )
    #print(confusion)

    fig, axes = plt.subplots(figsize=(16, 14))
    sbn.heatmap(
        confusion_norm,
        annot=confusion,
        cbar=False,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="viridis",
        ax=axes
    )
    axes.set_xlabel("Prédictions")
    axes.set_ylabel("Réalité")
    if imgfile == None:
        plt.show()
    else:
        if type(imgfile) is str:
            fig.savefig(imgfile)
        else:
            assert isinstance(imgfile, HTML_IMAGE) 
            fig.savefig(imgfile, format=imgfile.format)
    plt.close()
    return confusion



def batch_LM():
    nom_rapport="Rapport_modèle_linéaire.html"
    nom_dataset = "dataset_pipo"
    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Dataset : %s"%nom_dataset)
        R.texte("Effectifs avant filtrage :")
        R.table(relations=["un", "deux", "trois"], effectifs=[1,2,3])

    labels = ["un", "deux", "trois"]
    truth = np.random.randint(0,3, size=(500,))
    pred = np.random.randint(0,3, size=(500,))
    erreurs = np.random.choice((0,1), size=(500,), p=(0.2, 0.8))
    pred = (truth * erreurs) + (pred * (1-erreurs))

    with HTML_REPORT(nom_rapport) as R:
        R.titre("matrice de confusion", 2)
        with R.new_img("svg") as IMG:
            plot_confusion_matrix(truth, pred, labels, IMG)



if __name__ == "__main__" :
    batch_LM()