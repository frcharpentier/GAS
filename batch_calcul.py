import os
#from make_dataset import FusionElimination as FILT, AligDataset
from torch import optim, nn, utils, manual_seed
import random
import logging
from report_generator import HTML_REPORT, HTML_IMAGE
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np
from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Antisym
import lightning as LTN
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

os.environ['CUDA_VISIBLE_DEVICES']='1,4'


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

def pour_fusion(C, liste):
    if C.startswith(":") and C[1] != ">":
        CC = ":>" + C[1:].upper()
        if CC in liste:
            return CC
    return C

def faire_datasets_edges(train=True, dev=True, test=True):
    if train:
        DGRtr = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    if dev:
        DGRdv = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct", QscalK=True, split="dev")
    if test:
        DGRts = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct", QscalK=True, split="test")

    noms_classes = [k for k in DGRtr.filtre.alias]

    filtre = DGRtr.filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al, noms_classes))
    filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
    filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))
    filtre2 = filtre.eliminer(lambda x: x.al.startswith("{"))

    filtre2 = filtre.garder(":>AGENT", ":>BENEFICIARY", ":>CAUSE", ":>THEME",
                            ":>CONDITION", ":degree", ":>EXPERIENCER",
                            ":>LOCATION", ":>MANNER", ":>MOD", ":>PATIENT",
                            ":poss", ":>PURPOSE", ":>TIME", ":>TOPIC")

    if train:
        DGRtr_f2 = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="train")
    if dev:
        DGRdv_f2 = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="dev")
    if test:
        DGRts_f2 = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="test")
        
    datasets = ()
    if train:
        datasets += (EdgeDatasetMono(DGRtr_f2, "./edges_f_QK_train"),)
    if dev:
        datasets += (EdgeDatasetMono(DGRdv_f2, "./edges_f_QK_dev"),)
    if test:
        datasets += (EdgeDatasetMono(DGRts_f2, "./edges_f_QK_test"),)
    
    return datasets + (filtre2, )

def essai_train():
    DARtr, DARdv, DARts, filtre2 = faire_datasets_edges(True, True, True)

    dimension = 288
    nb_classes = len(filtre2.alias)
    freqs = filtre2.effectifs
    modele = Classif_Logist(dimension, nb_classes, freqs=freqs)

    arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    trainer = LTN.Trainer(max_epochs=100, devices=1, accelerator="gpu", callbacks=[arret_premat])

    print("Début de l’entrainement")
    train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
    valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
    trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("TERMINÉ.")
    modele.noms_classes = filtre2.alias # Pour étiqueter la matrice de confusion
    trainer.test(modele, dataloaders=utils.data.DataLoader(DARts, batch_size=32))

def essai_val():
    DGRtr = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    DGRdv = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct", QscalK=True, split="dev")
    DGRts = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct", QscalK=True, split="test")
    noms_classes = [k for k in DGRtr.filtre.alias]

    filtre = DGRtr.filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al, noms_classes))
    filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
    filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))
    filtre2 = filtre.eliminer(lambda x: x.al.startswith("{"))

    DGRtr_f2 = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="train")
    DGRdv_f2 = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="dev")
    DGRts_f2 = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="test")

    DARtr = EdgeDatasetMono(DGRtr_f2, "./edges_f_QK_train")
    DARts = EdgeDatasetMono(DGRts_f2, "./edges_f_QK_test")
    test_loader = utils.data.DataLoader(DARts)

    dimension = 288
    nb_classes = len(filtre2.alias)
    freqs = filtre2.effectifs
    modele = Classif_Logist.load_from_checkpoint("./lightning_logs/version_2/checkpoints/epoch=49-step=442950.ckpt",
                                                 dim=dimension, nb_classes=nb_classes, noms_classes=noms_classes, freqs = freqs)
    print("OK.")

                              



if __name__ == "__main__" :
    manual_seed(53)
    random.seed(53)
    #essai_val()
    essai_train()