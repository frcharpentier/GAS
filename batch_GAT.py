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
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Antisym
import lightning as LTN
#from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, EarlyStopping

from batch_calcul import filtre_defaut, plot_confusion_matrix

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
def batch_GAT_classif():
    pass




def test():
    (DS, ds) = faire_datasets_grph(train=True, dev=False, test=False, CLASSE=EdgeDataset)
    
    

if __name__ == "__main__":
    test()




