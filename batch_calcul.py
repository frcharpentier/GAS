import os
import inspect
#from make_dataset import FusionElimination as FILT, AligDataset
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
    ConfusionMatrixDisplay)

import numpy as np
from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir
from modeles import Classif_Logist, Classif_Bil_Sym, Classif_Bil_Antisym
import lightning as LTN
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

os.environ['CUDA_VISIBLE_DEVICES']='1,4'


class GitException(Exception):
    def __init__(self):
        super().__init__("Il y a des fichiers modifiés dans le repo git. Veillez à les soumettre, puis relancez.")
    
def git_get_commit():
    # Provoque une exception salutaire s’il y a des fichiers
    # modifiés dans le répo git. Dans le cas contraire,
    # renvoie le hash md5 du dernier instantané git.
    import subprocess
    
    cmd = "git status --porcelain"
    retour = subprocess.check_output(cmd, shell=True)
    if type(retour) == bytes:
        retour = retour.decode("utf-8")
    lignes = retour.split("\n")
    lignes = [lig for lig in lignes if len(lig) > 0]
    if any(not lig.startswith("?? ") for lig in lignes):
        raise GitException
    cmd = 'git log -1 --format=format:"%H"'
    retour = subprocess.check_output(cmd, shell=True)
    if type(retour) == bytes:
        retour = retour.decode("utf-8")
    retour = retour.strip()
    return retour

GLOBAL_HASH_GIT = git_get_commit()
#On stocke dans une variable globale, au tout début du programme.

def plot_confusion_matrix(y_true, y_pred, noms_classes=None):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels = noms_classes, normalize="true")
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(xticks_rotation="vertical", ax=ax)
    #pour calculer disp.figure_, qui est une figure matplotlib
    return disp.figure_


def filtre_defaut():
    ds = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    return ds.filtre

def faire_datasets_edges(filtre, train=True, dev=True, test=True):
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
        datasets += (EdgeDatasetMono(DGRtr_f2, "./edges_f_QK_train"),)
    if dev:
        datasets += (EdgeDatasetMono(DGRdv_f2, "./edges_f_QK_dev"),)
    if test:
        datasets += (EdgeDatasetMono(DGRts_f2, "./edges_f_QK_test"),)
    
    return datasets

def get_ckpt(modele):
    logger = modele.logger
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

def batch_LM():
    filtre = filtre_defaut()
    def pour_fusion(C, liste):
        if C.startswith(":") and C[1] != ">":
            CC = ":>" + C[1:].upper()
            if CC in liste:
                return CC
        return C
    noms_classes = [k for k in filtre.alias]
    filtre = filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al, noms_classes))
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
    modele = Classif_Logist(dimension, nb_classes, cible="roles", freqs=freqs)

    arret_premat = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    #trainer = LTN.Trainer(max_epochs=100, devices=1, accelerator="gpu", callbacks=[arret_premat])
    #trainer = LTN.Trainer(max_epochs=5, devices=1, accelerator="gpu", callbacks=[arret_premat])
    trainer = LTN.Trainer(max_epochs=2, accelerator="cpu")

    print("Début de l’entrainement")
    train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
    valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
    trainer.fit(model=modele, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("TERMINÉ.")

    nom_rapport="Rapport_Logistique.html"
    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre("Informations de reproductibilité", 2)
        chckpt = get_ckpt(modele)
        if not type(chckpt) == str:
            chckpt = repr(chckpt)
        R.table(fonction=str(inspect.stack()[0][3]),
                classe_modele=repr(modele.__class__),
                MD5_git=GLOBAL_HASH_GIT, 
                chkpt_model = chckpt)
        R.titre("paramètres d’instanciation", 3)
        hparams = {k: str(v) for k, v in modele.hparams.items()}
        R.table(**hparams, colonnes=False)
        
        R.titre("Dataset (classe et effectifs)", 2)
        R.table(relations=filtre2.alias, effectifs=filtre2.effectifs)
        dld = utils.data.DataLoader(DARts, batch_size=32)
        roles_pred = trainer.predict(
            modele,
            dataloaders=dld,
            return_predictions=True
        )
        roles_pred = torch.concatenate(roles_pred, axis=0) #On a obtenu une liste de tenseurs (un par batch)
        truth = torch.concatenate([roles.to(dtype=torch.long) for _,roles,_,_ in dld], axis=0)
        accuracy = accuracy_score(truth, roles_pred)
        bal_accuracy = balanced_accuracy_score(truth, roles_pred)
        R.titre("Accuracy : %f, balanced accuracy : %f"%(accuracy, bal_accuracy), 2)
        with R.new_img_with_format("svg") as IMG:
            fig = plot_confusion_matrix(truth, roles_pred, DARts.liste_roles)
            fig.savefig(IMG.fullname)
        R.ligne()


    #modele.noms_classes = filtre2.alias # Pour étiqueter la matrice de confusion
    #trainer.test(modele, dataloaders=utils.data.DataLoader(DARts, batch_size=32))

def rattraper():
    DARtr, DARdv, DARts, filtre2 = faire_datasets_edges(True, True, True)
    modele = Classif_Logist.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=99-step=360200.ckpt")
    trainer = LTN.Trainer(devices=1, accelerator="gpu")
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
    batch_LM()
    #rattraper()
    #essai_train()