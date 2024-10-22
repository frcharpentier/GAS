from make_dataset import FusionElimination as FILT, AligDataset, PermutEdgeDataset
import torch
from torch import optim, nn, utils, manual_seed
import random
import logging
from report_generator import HTML_REPORT, HTML_IMAGE
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import (confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay)

import numpy as np
from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Antisym
import lightning as LTN
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def filtre_defaut():
    ds = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    return ds.filtre


def faire_datasets_edges(filtre, train=True, dev=True, test=True, CLASSE = EdgeDatasetMono):
    if train:
        DGRtr_f2 = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct",
                            transform=filtre, QscalK=True, split="train")
    if dev:
        DGRdv_f2 = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct",
                            transform=filtre, QscalK=True, split="dev")
    if test:
        DGRts_f2 = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct",
                            transform=filtre, QscalK=True, split="test")
        
    datasets = ()
    if train:
        datasets += (CLASSE(DGRtr_f2, "./edges_f_QK_train"),)
    if dev:
        datasets += (CLASSE(DGRdv_f2, "./edges_f_QK_dev"),)
    if test:
        datasets += (CLASSE(DGRts_f2, "./edges_f_QK_test"),)
    
    return datasets


def main():
    ckpoint_model = "/home/frederic/projets/detection_aretes/lightning_logs/version_11/checkpoints/epoch=149-step=540300.ckpt"
    filtre = filtre_defaut()
    noms_classes = [k for k in filtre.alias]

    def pour_fusion(C):
        nonlocal noms_classes
        if C.startswith(":") and C[1] != ">":
            CC = ":>" + C[1:].upper()
            if CC in noms_classes:
                return CC
        return C
    
    filtre = filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al))
    filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
    filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))
    filtre2 = filtre.eliminer(lambda x: x.al.startswith("{"))

    filtre2 = filtre.garder(":>AGENT", ":>BENEFICIARY", ":>CAUSE", ":>THEME",
                            ":>CONDITION", ":degree", ":>EXPERIENCER",
                            ":>LOCATION", ":>MANNER", ":>MOD", ":>PATIENT",
                            ":poss", ":>PURPOSE", ":>TIME", ":>TOPIC")

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True, CLASSE=EdgeDataset)
    cible = "roles"
    modele = Classif_Bil_Sym.load_from_checkpoint(ckpoint_model).to("cpu")
    trainer = LTN.Trainer(accelerator="cpu")
    dld = utils.data.DataLoader(DARts, batch_size=32)
    roles_pred = trainer.predict(
        modele,
        dataloaders=dld,
        return_predictions=True
    )
    roles_pred_0 = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
    #truth = torch.concatenate([batch[cible] for batch in dld], axis=0)

    DARts.redresser_X(
        torch.randint(0,2,(DARts.Nadj,), dtype=torch.long),
        reshape=False)
    
    dld = utils.data.DataLoader(DARts, batch_size=32)
    roles_pred = trainer.predict(
        modele,
        dataloaders=dld,
        return_predictions=True
    )
    roles_pred_1 = torch.concatenate(roles_pred, axis=0)
    comparaison = (roles_pred_0 == roles_pred_1)
    comparaison = comparaison.all().item()
    return comparaison

main()


