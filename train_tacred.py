DEBUG = False

import os
import numpy as np
import torch
from tqdm import tqdm
from tacred.make_dataset import make_aligs, transfo_to_filenames, TacredDataset
#from DtSets.make_dataset import FusionElimination as FILT, AligDataset
from tacred.DataLoader import tacredGraphSampler, tacredStratifiedGraphSampler, tacredDataLoader, tacredGPUDataLoader
from tacred.models import tacred_GAT_MODEL, tacredTRAINER, tacredTRAINER_hparam_search

from outils.report_generator import HTML_REPORT, HTML_IMAGE
#from outils.interface_git import nettoyer_logs_lightning
from outils.autoinspect import autoinspect

import datetime
import fire
import wandb
import pprint




def batch_essai(rep):
    alig_file, rep_ds_grph, id_model = transfo_to_filenames("robertaBase", True)
    print("rep_ds_grph:", rep_ds_grph)

    dsTEST = TacredDataset(rep_ds_grph + "test",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="test",
                            device="cpu",
                            tokenHF=None)
    
    #dsDEV = TacredDataset(rep_ds_grph + "dev",
    #                        alig_file,
    #                        transform=None,
    #                        QscalK=True,
    #                        split="dev",
    #                        device="cpu",
    #                        tokenHF=None)

    #dsTRAIN = TacredDataset(rep_ds_grph + "train",
    #                        alig_file,
    #                        transform=None,
    #                        QscalK=True,
    #                        split="train",
    #                        device="cpu",
    #                        tokenHF=None)
    
    #index, cumul = dsTRAIN.get_strates()
    #print("shape de index:", index.shape)
    #print("cumul:", cumul)

    sampler = tacredGraphSampler(dsTEST, avg_num=5000, shuffle=False)
    
    #sampler = tacredStratifiedGraphSampler(dsTEST, avg_num=5000)
    
    ldTRAIN = tacredDataLoader(dsTEST, batch_sampler=sampler, follow_batch=["msk"], num_workers=0)
    return ldTRAIN
    print("Parcours d’une époch du dataset dans le dataloader avec remise et stratification:")
    visites = np.zeros((42,), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(ldTRAIN):
            #idx = batch.msk
            #selection = batch.x[idx,:,:]
            #moyenne = selection.mean()
            classes = batch.y1
            visites[classes] = visites[classes]+1
    print("Parcours terminé.")
    print(visites)
    
def make_datasets(transfo_name, train, dev, test):
    alig_file, rep_ds_grph, id_model = transfo_to_filenames(transfo_name, True)
    datasets = ()
    if train:
        dsTRAIN = TacredDataset(rep_ds_grph + "train",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="train",
                            device="cpu",
                            tokenHF=None)
        datasets += (dsTRAIN,)

    if dev:
        dsDEV = TacredDataset(rep_ds_grph + "dev",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="dev",
                            device="cpu",
                            tokenHF=None)
        datasets += (dsDEV,)
    if test:
        dsTEST = TacredDataset(rep_ds_grph + "test",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="test",
                            device="cpu",
                            tokenHF=None)
        datasets += (dsTEST,)

    return datasets


def make_dataloaders(transfo_name, train, dev, test, forceGPU=False, shuffleDev=False):
    datasets = make_datasets(transfo_name, train, dev, test)
    dataloaders = ()
    if forceGPU:
        CLASSE = tacredGPUDataLoader
    else:
        CLASSE = tacredDataLoader
    if train:
        dsTRAIN = datasets[0]
        datasets = datasets[1:]
        stratSampler = tacredStratifiedGraphSampler(dsTRAIN, avg_num=5000)
        ldTRAIN = CLASSE(
            dsTRAIN,
            batch_sampler=stratSampler,
            follow_batch=["msk"],
            num_workers=1, 
            forceGPU=forceGPU
        )
        dataloaders += (ldTRAIN,)
    if dev or test:
        for ds in datasets:
            regSampler = tacredGraphSampler(ds, avg_num=5000, shuffle=shuffleDev)
            loader = CLASSE(
                ds,
                batch_sampler=regSampler,
                follow_batch=["msk"],
                num_workers=0,
                forceGPU=forceGPU
            )
            dataloaders += (loader,)

    return dataloaders



def eval_checkpoint(chkpoint):
    #(dsDEV,) = make_datasets("robertaBase", False, True, False)
    #regSampler   = tacredGraphSampler(dsDEV, avg_num=5000, shuffle=False)
    #ldDEV   = tacredGPUDataLoader(
    #    dsDEV,
    #    batch_sampler=regSampler,
    #    follow_batch=["msk"],
    #    num_workers=0, 
    #)
    (ldDEV,) = make_dataloaders("robertaBase", False, True, False, forceGPU=True)
    model = tacred_GAT_MODEL.from_pretrained(chkpoint)
    model.to(device="cuda:0")
    model.eval()
    nomMetriq="hyperml/balanced_accuracy"
    metriq, lossEval = model.evaluer(nomMetriq, ldDEV)
    print("%s : %f"%(nomMetriq, metriq))
    print("Perte eval : %f"%lossEval)



@autoinspect
def batch_train_1(
        transfo_name = "robertaBase",
        lr=5.e-4,
        epochs=20,
        weight_decay=0.,
        metriq="custom_Fscore",
        a_surveiller = "F_score(macro)",
        patience = 5):
    
    ldTRAIN, ldDEV = make_dataloaders(transfo_name, True, True, False, forceGPU=False)

    print("CRÉATION DU MODÈLE")
    model = tacred_GAT_MODEL(dim_in = 144,
                 dim_h1 = 144,
                 dim_h2 = 144,
                 heads = 1,
                 nb_couches = 4,
                 rang_sim = 8,
                 dropout_p = 0.3,
                 nb_classes = 42)
    
    print("CRÉATION DE L'ENTRAÎNEUR")
    trainer = tacredTRAINER(model=model,
                            lr=lr,
                            epochs=epochs,
                            weight_decay=weight_decay,
                            metriq=metriq,
                            a_surveiller=a_surveiller,
                            patience=patience
                            )
    expe = trainer.rep_expe
    nom_rapport = os.path.join("rapports_tacred", expe)
    with HTML_REPORT(nom_rapport) as R:
        R.ligne()
        R.titre(expe, 2)
        R.reexecution()
        num_params = model.count_params()
        print("Nombre de paramètres: %d"%num_params)
        R.texte("Nombre de paramètres: %d"%num_params)

        config = trainer.get_config()
        R.table(**config, colonnes=False)
    
        print("DÉMARRAGE de L'ENTRAÎNEMENT")
        trainer.train(ldTRAIN, ldDEV)
        print("TERMINÉ")
        duree = trainer.elapsed_time()
        print("Durée de l’entrainement : %s"%str(duree))
        R.texte("Durée de l’entrainement : %s"%str(duree))
    
    



def essai2():
    from tacred.make_dataset import sourcer_tacred_alig
    alig_file, rep_ds_grph, id_model = transfo_to_filenames("robertaBase", True)
    alig_file += "_test.json"
    print("alig_file:", alig_file)
    iter0 = sourcer_tacred_alig(alig_file)
    for N in range(5):
        TT = next(iter0)
        print(repr(TT))


def essai_train_tacred():
    #from tacred.HFTrainer_train import entrainement
    from tacred.HFAccelerate_train import entrainement
    entrainement()

def wandb_train_from_config(config):
    transfo_name="robertaBase"
    print("CRÉATION DE L'ENTRAÎNEUR")
    trainer = tacredTRAINER_hparam_search.apartir_config(config)
    #model = trainer.model
    ldTRAIN, ldDEV = make_dataloaders(transfo_name, True, True, False, forceGPU=False)
    print("DÉMARRAGE de L'ENTRAÎNEMENT")
    trainer.train(ldTRAIN, ldDEV)
    print("TERMINÉ")
    duree = trainer.elapsed_time()
    print("Durée de l’entrainement : %s"%str(duree))

def wandb_essai_train():
    wandb.login()
    config = {
        "model_dim_in": 144,
        "model_dim_h1": 144,
        "model_dim_h2": 144,
        "model_heads": 1,
        "model_nb_couches": 4,
        "model_rang_sim": 8,
        "model_dropout_p": 0.3,
        "model_add_LN": False,
        "model_option_couche_0": False,
        "model_nb_classes":42,
        "trainer_lr": 5.e-4,
        "trainer_epochs": 1,
        "trainer_weight_decay": 1.e-4,
        "trainer_metriq": "custom_Fscore"
    }
    wandb_train_from_config(config)


# Fonction pour faire les sweep. Elle ne doit pas prendre de paramètres
def sweep_function():
    wandb.init()
    config = wandb.config
    transfo_name="robertaBase"
    print("CONFIGURATION:")
    pprint.pprint(config)
    print("CRÉATION DE L'ENTRAÎNEUR")
    trainer = tacredTRAINER_hparam_search.apartir_config(config)
    #model = trainer.model
    ldTRAIN, ldDEV = make_dataloaders(transfo_name, True, True, False, forceGPU=False)
    print("DÉMARRAGE de L'ENTRAÎNEMENT")
    trainer.train(ldTRAIN, ldDEV, wandb_initialized = True)
    print("TERMINÉ")
    duree = trainer.elapsed_time()
    print("Durée de l’entrainement : %s"%str(duree))






# Fonction à exécuter depuis la ligne de commande pour tester
# plein d’hyperparamètres par "sweep" grâce à WandB
def run_sweep(sweep_id, count):
    wandb.login()
    wandb.agent(sweep_id, function=sweep_function, count=count)



    

if __name__ == "__main__":
    if DEBUG:
        print("""
DDD   EEEE  BBB   U   U   GGG
D  D  E     B  B  U   U  G
D  D  EEE   BBB   U   U  G  GG
D  D  E     B  B  U   U  G   G
DDD   EEEE  BBB    UUU    GGG
""")
        #rep = "/home/charpentier/projets/json_tacred/"
        #essai(rep)
        batch_train_1(DEBUG=True)
        #essai_train_tacred()
        #essai2()


    else:
        fire.Fire()
