import os

DEBUG = False
from interface_git import nettoyer_logs_lightning
from autoinspect import autoinspect
nettoyer_logs_lightning()

import os
import fire
#import inspect
from make_dataset import FusionElimination as FILT, AligDataset, PermutEdgeDataset
import torch
from torch import optim, nn, utils, manual_seed
import random
import logging
from report_generator import HTML_REPORT, HTML_IMAGE
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import (confusion_matrix,
    ConfusionMatrixDisplay)

import numpy as np
from make_dataset import ( AligDataset, EdgeDataset,
                          EdgeDatasetMono, EdgeDatasetRdmDir,
                          EdgeDatasetMonoEnvers)
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Sym_2, Classif_Bil_Antisym
import lightning as LTN
#from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, EarlyStopping

from batch_calcul import filtre_defaut, plot_confusion_matrix, get_ckpt, calculer_exactitudes

#os.environ['CUDA_VISIBLE_DEVICES']='1,4'
os.environ['CUDA_VISIBLE_DEVICES']='4'

def faire_datasets_grph(filtre="defaut", train=True, dev=True, test=True, CLASSE=AligDataset):
    if filtre == "defaut":
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
        filtre = filtre.eliminer("{syntax}")
        filtre = filtre.fusionner(lambda x: pour_fusion(x.al))
        filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
        filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))
        
        a_garder = (":>AGENT", ":>BENEFICIARY", ":>CAUSE", ":>THEME",
                                ":>CONDITION", ":degree", ":>EXPERIENCER",
                                ":>LOCATION", ":>MANNER", ":>MOD", ":>PATIENT",
                                ":poss", ":>PURPOSE", ":>TIME", ":>TOPIC")
        a_garder = a_garder + tuple(x for x in filtre.alias if x.startswith("{"))

        filtre = filtre.garder(*a_garder)
    datasets = ()
    if train:
        dsTRAIN = AligDataset("./dataset_QK_train",
                              "./AMR_et_graphes_phrases_explct",
                              transform=filtre,
                              QscalK=True, split="train")
        

        if CLASSE == EdgeDataset:
            dsTRAIN = CLASSE(dsTRAIN, "./edges_f_QK_train", masques="1")
        datasets += (dsTRAIN,)
    if dev:
        dsDEV = AligDataset("./dataset_QK_dev",
                            "./AMR_et_graphes_phrases_explct",
                            transform=filtre,
                            QscalK=True, split="dev")
        if CLASSE == EdgeDataset:
            dsDEV = CLASSE(dsDEV, "./edges_f_QK_dev", masques="1")
        datasets += (dsDEV,)
    if test:
        dsTEST = AligDataset("./dataset_QK_test",
                             "./AMR_et_graphes_phrases_explct",
                             transform=filtre,
                             QscalK=True, split="test")
        if CLASSE == EdgeDataset:
            dsTEST = CLASSE(dsTEST, "./edges_f_QK_test", masques="1")
        datasets += (dsTEST,)

    return datasets

@autoinspect
def batch_Bilin_tous_tokens(nom_rapport, rang=2, ckpoint_model=None, train=True, shuffle=False):
    DARtr, DARdv, DARts = faire_datasets_grph(train=True, dev=True, test=True, CLASSE = EdgeDataset)
    filtre = DARtr.filtre

    dimension = 144
    nb_classes = len(filtre.alias)
    freqs = filtre.effectifs
    cible = "roles"
    lr = 1.e-4
    if ckpoint_model:
        modele = Classif_Bil_Sym_2.load_from_checkpoint(ckpoint_model)
    else:
        modele = Classif_Bil_Sym_2(dimension, nb_classes, rang=rang, cible=cible, lr=lr, freqs=freqs)
    if train:
        arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        
        #trainer = LTN.Trainer(max_epochs=150, devices=1, accelerator="gpu", callbacks=[arret_premat])
        trainer = LTN.Trainer(max_epochs=5, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=2, accelerator="cpu")
    
        print("Début de l’entrainement")
        train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8, shuffle=shuffle)
        valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
        trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        print("TERMINÉ.")
    else:
        trainer = LTN.Trainer(devices=1, accelerator="gpu")

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.reexecution()
        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not chckpt and (not ckpoint_model is None):
            chckpt = ckpoint_model
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        R.table(colonnes=False,
                classe_modele=repr(modele.__class__),
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et effectifs)", 2)
        groupes = [" ".join(k for k in T) for T in filtre.noms_classes]
        R.table(relations=filtre.alias, groupes=groupes, effectifs=filtre.effectifs)
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([batch[cible] for batch in dld], axis=0)

        exactitudes = calculer_exactitudes(truth, roles_pred, freqs)
        #accuracy = accuracy_score(truth, roles_pred)
        #bal_accuracy = balanced_accuracy_score(truth, roles_pred)
        R.titre("Exactitude : %f, exactitude équilibrée : %f"%(exactitudes["acc"], exactitudes["bal_acc"]), 2)
        R.titre("Exactitude équilibrée rééchelonnée entre hasard et perfection : %f"%exactitudes["bal_acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard uniforme et perfection : %f"%exactitudes["acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard selon a priori et perfection : %f"%exactitudes["acc_adj2"], 2)

        with R.new_img_with_format("svg") as IMG:
            fig, matrix = plot_confusion_matrix(truth, roles_pred, DARts.liste_roles)
            fig.savefig(IMG.fullname)
        matrix = repr(matrix.tolist())
        R.texte_copiable(matrix, hidden=True, buttonText="Copier la matrice de confusion")
        R.ligne()




def test():
    (DS, ds) = faire_datasets_grph(train=True, dev=False, test=False, CLASSE=EdgeDataset)
    
    

if __name__ == "__main__":
    test()




