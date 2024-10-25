from interface_git import nettoyer_logs_lightning
nettoyer_logs_lightning()
DEBUG = True

import os
import fire
import inspect
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

#os.environ['CUDA_VISIBLE_DEVICES']='1,4'
os.environ['CUDA_VISIBLE_DEVICES']='4'



#On stocke dans une variable globale, au tout début du programme.

def plot_confusion_matrix(y_true, y_pred, noms_classes=None):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels = noms_classes, normalize="true")
    matrix = confusion_matrix(y_true, y_pred)
    NN, _ = matrix.shape
    if NN < 7:
        NN = 7
    fig, ax = plt.subplots(figsize=(NN,NN))
    disp.plot(xticks_rotation="vertical", ax=ax)
    #pour calculer disp.figure_, qui est une figure matplotlib
    return disp.figure_, matrix


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



def get_ckpt(modele):
    logger = modele.logger
    if logger is None:
        return False
    version = logger.version
    if type(version) == int:
        version = "version_%d"%version
    repert = os.path.join(logger.save_dir, logger.name, version)
    if not os.path.exists(repert):
        return False
    repert = os.path.join(repert, "checkpoints")
    if not os.path.exists(repert):
        return False
    fichiers = [os.path.join(repert, f) for f in os.listdir(repert)]
    if len(fichiers) == 0:
        return False
    if len(fichiers) == 1:
        return fichiers[0]
    return fichiers

def calculer_exactitudes(truth, pred, freqs = None):
    M = confusion_matrix(truth, pred)
    effectifs = M.sum(axis=1)
    diago = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        rappels = diago / effectifs
    if np.any(np.isnan(rappels)):
        rappels = rappels[~np.isnan(rappels)]
        # élimination des valeurs nan
    bal_acc = np.mean(rappels)
    # bal_acc : exactitude équilibrée
    n_classes = len(rappels)
    hasard = 1 / n_classes
    bal_acc_adj = (bal_acc - hasard)/(1-hasard)
    # bal_acc_adj : exactitude équilibrée, et rééchelonnée entre hasard et perfection
    sum_diag = diago.sum()
    total = effectifs.sum()
    acc = sum_diag / total
    # acc : exactitude
    acc_adj = (acc - hasard) / (1-hasard)
    # acc_adj : exactitude rééchelonnée entre hasard (uniforme) et perfection
    if not freqs is None:
        if type(freqs) == list:
            freqs = np.array(freqs)
        freqs = freqs / freqs.sum()
        hasard2 = (freqs**2).sum()
        # hasard2 : exactitude d’un classificateur au hasard qui suit la distribution freq
        acc_adj2 = (acc - hasard2) / (1-hasard2)
        # acc_adj2 : exactitude rééchelonnée entre hasard2 et perfection
    else:
        acc_adj2 = None
    return {"acc":acc, "acc_adj":acc_adj, "acc_adj2":acc_adj2,
            "bal_acc": bal_acc, "bal_acc_adj": bal_acc_adj}


def get_appel_fonction():
    F2 = inspect.stack()[1]
    fonction = F2.function
    arguments = inspect.getargvalues(F2.frame).locals
    arguments = {k: v for k, v in arguments.items()}
    return fonction, arguments

def str_appel_fonction(fonction, arguments):
    return fonction + "(" + ", ".join("%s=%s"%(k,repr(v)) for k,v in arguments.items()) + ")"

def batch_LM(nom_rapport, ckpoint_model=None, train=True):
    fonction, arguments = get_appel_fonction()

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

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True)

    dimension = 288
    nb_classes = len(filtre2.alias)
    freqs = filtre2.effectifs
    cible = "roles"
    lr = 1.e-5
    if ckpoint_model:
        modele = Classif_Logist.load_from_checkpoint(ckpoint_model)
    else:
        modele = Classif_Logist(dimension, nb_classes, cible=cible, lr=lr, freqs=freqs)
    if train:
        arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        trainer = LTN.Trainer(max_epochs=100, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=5, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=2, accelerator="cpu")
    
        print("Début de l’entrainement")
        train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
        valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
        trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        print("TERMINÉ.")
    else:
        trainer = LTN.Trainer(devices=1, accelerator="gpu")

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Pour réexécuter :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)
        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not chckpt and (not ckpoint_model is None):
            chckpt = ckpoint_model
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        arguments["ckpoint_model"] = chckpt
        arguments["train"] = False
        R.table(colonnes=False,
                fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et effectifs)", 2)
        groupes = [" ".join(k for k in T) for T in filtre2.noms_classes]
        R.table(relations=filtre2.alias, groupes=groupes, effectifs=filtre2.effectifs)
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([batch[cible] for batch in dld], axis=0)

        R.titre("Pour recalculer ces statistiques :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        exactitudes = calculer_exactitudes(truth, roles_pred, freqs)
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


    #modele.noms_classes = filtre2.alias # Pour étiqueter la matrice de confusion
    #trainer.test(modele, dataloaders=utils.data.DataLoader(DARts, batch_size=32))

def rattraper():
    DARtr, DARdv, DARts, filtre2 = faire_datasets_edges(True, True, True)
    modele = Classif_Logist.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=99-step=360200.ckpt")
    trainer = LTN.Trainer(devices=1, accelerator="gpu")
    modele.noms_classes = filtre2.alias # Pour étiqueter la matrice de confusion
    trainer.test(modele, dataloaders=utils.data.DataLoader(DARts, batch_size=32))
    
def batch_LM_VerbAtlas_ARGn(nom_rapport = "Rapport_Logistique.html"):
    fonction, arguments = get_appel_fonction()
    ckpt = "/home/frederic/projets/detection_aretes/lightning_logs/version_3/checkpoints/epoch=49-step=180100.ckpt"
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

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True)
    cible = "roles"
    modele = Classif_Logist.load_from_checkpoint(ckpt)
    trainer = LTN.Trainer(devices=1, accelerator="gpu")

    ordre_classes = [ ':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5', ':ARG6',
                      ':>BENEFICIARY', ':>CONDITION', ':>LOCATION', ':>MANNER', 
                      ':>MOD', ':>PURPOSE', ':>TIME', ':>TOPIC', ':degree', ':poss',
                      ':>AGENT', ':>THEME', ':>PATIENT', ':>EXPERIENCER', ':>CAUSE'
                    ]
    # Construction de la permutation
    permut = [None] * len(ordre_classes)
    for i, C in enumerate(DARts.liste_rolARG):
        j = ordre_classes.index(C)
        permut[i] = j

    permut = torch.tensor(permut)

    dld = utils.data.DataLoader(DARts, batch_size=32)
    VerbAtlas_pred = trainer.predict(
        modele,
        dataloaders=dld,
        return_predictions=True
    )
    VerbAtlas_pred = torch.concatenate(VerbAtlas_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
    VerbAtlas_truth = torch.concatenate([batch["roles"] for batch in dld], axis=0)
    ARGn_truth = torch.concatenate([batch["ARGn"] for batch in dld], axis=0)
    # Construisons les prédictions pour ARGn : Chaque fois que le rôle VerbAtlas prédit est identique
    # au rôle VerbAtlas réel, on prend le rôle ARGn réel. Sinon, on garde le rôle VerbAtlas prédit.
    reussites = (VerbAtlas_pred == VerbAtlas_truth)
    echecs = ~reussites
    ARGn_pred = (reussites * ARGn_truth) + (echecs * VerbAtlas_pred)

    # Renumérotation des prédictions et des réalités
    ARGn_pred = permut[ARGn_pred]
    ARGn_truth = permut[ARGn_truth]
    
    freqs = [None] * len(DARtr.freqARGn)
    for i, frq in enumerate(DARtr.freqARGn.numpy().tolist()):
        freqs[permut[i].item()] = frq

    exactitudes = calculer_exactitudes(ARGn_truth, ARGn_pred, freqs)

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Note", 2)
        R.texte("""Cette expérience montre la capacité d’un classificateur linéaire entraîné sur l’étiquetage sémantique VerbAtlas
à détecter correctement les rôles étiquetés normalement selon PropBank (AMR).
Chaque fois que le classificateur détecte la bonne étiquette VerbAtlas, on remplace par l’étiquette AMR (puisqu’elle serait déterministe) pour calculer les scores.
Si l’étiquette n’est pas correctement détectée, on laisse l’étiquette telle quelle.
Certaines étiquettes VerbAtlas n’apparaissent jamais dans l’étiquetage PropBank (agent, patient...), ce qui explique que certaines lignes de la matrice de confusion
soient entièrement nulles. Pour le calcul de l’exactitude équilibrée (balanced accuracy), ces lignes sont omises.
(La nullité d’une ligne n’est pas du tout gênante pour le calcul de l’exactitude, en revanche.)""")
        R.titre("Informations de reproductibilité", 2)
        R.table(colonnes=False,
                fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = ckpt)
        R.titre("paramètres d’instanciation du modèle", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et effectifs)", 2)
        R.table(relations=DARtr.liste_rolARG, fréquences=DARts.freqARGn.numpy().tolist())
        
        R.titre("Pour recalculer ces statistiques :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)
        
        R.titre("Exactitude : %f, exactitude équilibrée : %f"%(exactitudes["acc"], exactitudes["bal_acc"]), 2)
        R.titre("Exactitude équilibrée rééchelonnée entre hasard et perfection : %f"%exactitudes["bal_acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard uniforme et perfection : %f"%exactitudes["acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard selon a priori et perfection : %f"%exactitudes["acc_adj2"], 2)

        with R.new_img_with_format("svg") as IMG:
            fig, matrix = plot_confusion_matrix(ARGn_truth, ARGn_pred, ordre_classes)
            fig.savefig(IMG.fullname)
        matrix = repr(matrix.tolist())
        R.texte_copiable(matrix, hidden=True, buttonText="Copier la matrice de confusion")
        R.ligne()




def batch_LM_ARGn(nom_rapport, ckpoint_model=None, train=True):
    fonction, arguments = get_appel_fonction()
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

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True)

    ordre_classes = [ ':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5', ':ARG6',
                      ':>BENEFICIARY', ':>CONDITION', ':>LOCATION', ':>MANNER', 
                      ':>MOD', ':>PURPOSE', ':>TIME', ':>TOPIC', ':degree', ':poss',
                      # Les classes listées après cette ligne ont un effectif nul pour l'étiquette ARGn
                      ':>AGENT', ':>THEME', ':>PATIENT', ':>EXPERIENCER', ':>CAUSE'
                    ]
    
    DARtr = PermutEdgeDataset(DARtr, ordre_classes)
    DARdv = PermutEdgeDataset(DARdv, ordre_classes)
    DARts = PermutEdgeDataset(DARts, ordre_classes)


    dimension = 288
    nb_classes = 17 # dix-sept classes aux effectifs non nuls
    assert len(DARtr.liste_rolARG) == len(ordre_classes)
    assert all(x==y for x, y in zip(DARtr.liste_rolARG, ordre_classes))
    liste_classes = DARtr.liste_rolARG[:nb_classes]
    freqs = DARtr.freqARGn[:nb_classes]
    cible = "ARGn"
    lr = 1.e-5

    modele = Classif_Logist(dimension, nb_classes, cible=cible, lr=lr, freqs=freqs)
    if ckpoint_model:
        modele = Classif_Logist.load_from_checkpoint(ckpoint_model)
    else:
        modele = Classif_Logist(dimension, nb_classes, cible=cible, lr=lr, freqs=freqs)

    if train:
        arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        trainer = LTN.Trainer(max_epochs=100, devices=1, accelerator="gpu", callbacks=[arret_premat])
    
        print("Début de l’entrainement")
        train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
        valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
        trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        print("TERMINÉ.")
    else:
        trainer = LTN.Trainer(devices=1, accelerator="gpu")

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Note", 2)
        R.texte("Expérience de classification sur les rôles PropBank sans passer par VerbAtlas")
        
        R.titre("Pour réexécuter :",2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not chckpt and (not ckpoint_model is None):
            chckpt = ckpoint_model
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        arguments["ckpoint_model"] = chckpt
        arguments["train"] = False
        R.table(colonnes=False,
                fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et fréquences)", 2)
        #groupes = [" ".join(k for k in T) for T in filtre2.noms_classes]
        R.table(relations=ordre_classes, fréquences=freqs.numpy().tolist() + [0]*(len(ordre_classes) - nb_classes))
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([batch[cible] for batch in dld], axis=0)
        
        R.titre("Pour recalculer ces statistiques :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        exactitudes = calculer_exactitudes(truth, roles_pred, freqs)
        #accuracy = accuracy_score(truth, roles_pred)
        #bal_accuracy = balanced_accuracy_score(truth, roles_pred)
        R.titre("Exactitude : %f, exactitude équilibrée : %f"%(exactitudes["acc"], exactitudes["bal_acc"]), 2)
        R.titre("Exactitude équilibrée rééchelonnée entre hasard et perfection : %f"%exactitudes["bal_acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard uniforme et perfection : %f"%exactitudes["acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard selon a priori et perfection : %f"%exactitudes["acc_adj2"], 2)

        with R.new_img_with_format("svg") as IMG:
            fig, matrix = plot_confusion_matrix(truth, roles_pred, liste_classes)
            fig.savefig(IMG.fullname)
        matrix = repr(matrix.tolist())
        R.texte_copiable(matrix, hidden=True, buttonText="Copier la matrice de confusion")
        R.ligne()

def batch_Antisym(nom_rapport, ckpoint_model=None, train=True):
    fonction, arguments = get_appel_fonction()
    filtre = filtre_defaut()

    filtre2 = filtre

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True, CLASSE = EdgeDataset)
    
    #Permutation aléatoire pour équilibrer les datasets
    DARtr.permuter_X1_X2(torch.randint(0,2,(DARtr.Nadj,), dtype=torch.long))
    DARdv.permuter_X1_X2(torch.randint(0,2,(DARdv.Nadj,), dtype=torch.long))
    DARts.permuter_X1_X2(torch.randint(0,2,(DARts.Nadj,), dtype=torch.long))


    dimension = 144
    lr = 1.e-4
    rang = 18 # Éviter les valeurs impaires : Une matrice antisymétrique d’ordre impair est toujours singulière

    if ckpoint_model:
        modele = Classif_Bil_Antisym.load_from_checkpoint(ckpoint_model)
    else:
        modele = Classif_Bil_Antisym(dimension, rang=rang, lr=lr)
    if train:
        arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        trainer = LTN.Trainer(max_epochs=30, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=5, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=2, accelerator="cpu")
    
        print("Début de l’entrainement")
        train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
        valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
        trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        print("TERMINÉ.")
    else:
        trainer = LTN.Trainer(devices=1, accelerator="gpu")

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Pour réexécuter :",2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not chckpt and (not ckpoint_model is None):
            chckpt = ckpoint_model
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        arguments["ckpoint_model"] = chckpt
        arguments["train"] = False
        R.table(colonnes=False,
                fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([batch["sens"] for batch in dld], axis=0)
        
        R.titre("Pour recalculer ces statistiques :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        exactitudes = calculer_exactitudes(truth, roles_pred)
        #accuracy = accuracy_score(truth, roles_pred)
        #bal_accuracy = balanced_accuracy_score(truth, roles_pred)
        R.titre("Exactitude : %f, exactitude équilibrée : %f"%(exactitudes["acc"], exactitudes["bal_acc"]), 2)
        R.titre("Exactitude équilibrée rééchelonnée entre hasard et perfection : %f"%exactitudes["bal_acc_adj"], 2)
        R.titre("Exactitude rééchelonnée entre hasard uniforme et perfection : %f"%exactitudes["acc_adj"], 2)

        with R.new_img_with_format("svg") as IMG:
            fig, matrix = plot_confusion_matrix(truth, roles_pred)
            fig.savefig(IMG.fullname)
        matrix = repr(matrix.tolist())
        R.texte_copiable(matrix, hidden=True, buttonText="Copier la matrice de confusion")
        R.ligne()


def batch_Bilin(nom_rapport, rang=2, ckpoint_model=None, train=True):
    fonction, arguments = get_appel_fonction()
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

    DARtr, DARdv, DARts = faire_datasets_edges(filtre2, True, True, True, CLASSE = EdgeDataset)

    dimension = 144
    nb_classes = len(filtre2.alias)
    freqs = filtre2.effectifs
    cible = "roles"
    lr = 1.e-4
    if ckpoint_model:
        modele = Classif_Bil_Sym.load_from_checkpoint(ckpoint_model)
    else:
        modele = Classif_Bil_Sym(dimension, nb_classes, rang=rang, cible=cible, lr=lr, freqs=freqs)
    if train:
        arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        trainer = LTN.Trainer(max_epochs=150, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=5, devices=1, accelerator="gpu", callbacks=[arret_premat])
        #trainer = LTN.Trainer(max_epochs=2, accelerator="cpu")
    
        print("Début de l’entrainement")
        train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
        valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
        trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        print("TERMINÉ.")
    else:
        trainer = LTN.Trainer(devices=1, accelerator="gpu")

    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Pour réexécuter :",2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not chckpt and (not ckpoint_model is None):
            chckpt = ckpoint_model
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        arguments["ckpoint_model"] = chckpt
        arguments["train"] = False
        R.table(colonnes=False,
                fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et effectifs)", 2)
        groupes = [" ".join(k for k in T) for T in filtre2.noms_classes]
        R.table(relations=filtre2.alias, groupes=groupes, effectifs=filtre2.effectifs)
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([batch[cible] for batch in dld], axis=0)

        R.titre("Pour recalculer ces statistiques :", 2)
        R.texte_copiable(str_appel_fonction(fonction, arguments), hidden=False)

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

# Pour refaire une expérience, le plus simple désormais est de faire ainsi :
# Sélectionner avec git checkout le bon instantané git, ouvrir une console, lancer python
# taper : from batch_calcul import batch_LM
# taper : batch_LM(nom_rapport="rejeu.html",
#           ckpoint_model="/home/frederic/projets/detection_aretes/lightning_logs/version_3/checkpoints/epoch=49-step=180100.ckpt",
#           train=False)
                              



if __name__ == "__main__" :
    manual_seed(53)
    random.seed(53)
    if DEBUG:
        print("""
DDD   EEEE  BBB   U   U   GGG
D  D  E     B  B  U   U  G
D  D  EEE   BBB   U   U  G  GG
D  D  E     B  B  U   U  G   G
DDD   EEEE  BBB    UUU    GGG
""")
        #batch_LM(nom_rapport="Rapport_Logistique.html")
        #batch_LM_VerbAtlas_ARGn()
        #batch_LM_ARGn(nom_rapport="logistiq_ARGn.html")
        #batch_LM_ARGn(nom_rapport="logistiq_ARGn.html",
        #              ckpoint_model="/home/frederic/projets/detection_aretes/lightning_logs/version_8/checkpoints/epoch=99-step=360200.ckpt",
        #              train=False)

        #batch_Bilin(nom_rapport = "Rapport_Bilin_Sym.html")

        batch_Antisym(nom_rapport = "Rapport_Bilin_Antisym.html")

        #batch_LM(nom_rapport="rejeu.html",
        #         ckpoint_model="/home/frederic/projets/detection_aretes/lightning_logs/version_3/checkpoints/epoch=49-step=180100.ckpt",
        #         train=False)
        
        #rattraper()
        #essai_train()
    else:
        fire.Fire()