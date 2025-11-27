import os
import wandb
import random
import torch
from torch_geometric.data import Batch
from train_tacred import make_datasets, make_dataloaders
from tacred.models import tacred_GAT_MODEL
import pprint
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
#from PIL import Image as PILImage
#import io
from tacred.DataLoader import tacredDataLoader
from accelerate import Accelerator



def sauver_modele():
    with wandb.init(project="EXPE_20_Nov_2025_10H01_25787521", id="7x7qi3jo", resume="must") as run:
        config = run.config
        pprint.pprint(config)
        run.log_model("./tacred_checkpoints/EXPE_20_Nov_2025_10H01_25787521_epoch_047_BEST", aliases = ["best_epoq_47"])
        run.finish()

@torch.no_grad
def compute_true_preds(modele, dataldr, device="cpu"):
    iter0 =  (X for X in tqdm(dataldr))
    iterRef, iter1 = itertools.tee(iter0, 2)
    iterRef = (b.y1[0].item() for b in iterRef)
    iterPred = (modele.predict_batch(b) for b in iter1) 
    iterPred = (p[0].item() for p, _ in iterPred)

    y_true = []
    y_preds = []
    for r, p in zip(iterRef, iterPred):
        y_true.append(r)
        y_preds.append(p)
    
    return y_true, y_preds


def confus_matrix():
    modele = tacred_GAT_MODEL.from_pretrained("./tacred_checkpoints/EXPE_20_Nov_2025_10H01_25787521_epoch_047_BEST")
    #modele = modele.to(device="cuda:0")
    (dtset,) = make_datasets("robertaBase", False, False, True)
    dtldr = tacredDataLoader(dtset, batch_size=1, shuffle=False, follow_batch=["msk"])
    accelerator = Accelerator(mixed_precision="bf16")
    modele_mod, dtldr_mod = accelerator.prepare(modele, dtldr)
    print("PARCOURS DU DATASET")
    y_true, y_pred = compute_true_preds(modele_mod, dtldr_mod, device="cuda:0")
    accelerator.end_training()
    class_names = dtset.liste_relations
    print("CALCUL DE LA MATRICE DE CONFUSION")
    matrice = confusion_matrix(y_true, y_pred)
    matrice = json.dumps(matrice.tolist())
    fig, ax = plt.subplots(figsize=(20,20))
    dsp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, normalize="true")
    dsp.plot(ax=ax, xticks_rotation="vertical")

    with wandb.init(project="EXPE_20_Nov_2025_10H01_25787521", name="confus_test_set") as run:
        nom_sauvegarde = os.path.join("./rapports_tacred", run.project, run.name + "___" + run.id, "confusion.svg")
        os.makedirs(os.path.dirname(nom_sauvegarde), exist_ok=True)
        dsp.figure_.savefig(nom_sauvegarde, format="svg")   
        run.log({
            "matrice_confusion": dsp.figure_,
            "confus": matrice,
        })
        run.finish()


def essai2():
    #y_true = [0,1,2,0,1,2,0,1,2,0,1,2]
    #y_pred = [0,2,1,0,1,1,1,1,2,2,1,2]
    #class_names = ["classe A", "classe B", "classe C"]

    y_true = [random.randint(0,41) for _ in range(500)]
    y_pred = [random.randint(0,41) for _ in range(500)]
    class_names = ["classe %d"%i for i in range(42)]

    matrice = confusion_matrix(y_true, y_pred)
    matrice = json.dumps(matrice.tolist())
    fig, ax = plt.subplots(figsize=(20,20))
    dsp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, normalize="true")
    dsp.plot(ax=ax, xticks_rotation="vertical")
    #print(matrice)


    # Initialize a run
    with wandb.init(project="essai_confusion") as run:
        nom_sauvegarde = os.path.join("./rapports_tacred", run.project, run.name + "___" + run.id, "confusion.svg")
        os.makedirs(os.path.dirname(nom_sauvegarde), exist_ok=True)
        dsp.figure_.savefig(nom_sauvegarde, format="svg")   
        run.log({
            "matrice_confusion": dsp.figure_,
            "confus": matrice,
        })

if __name__ == "__main__":
    wandb.login()
    #sauver_modele()
    confus_matrix()