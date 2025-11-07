import os
import numpy as np
import torch
from tqdm import tqdm
from tacred.make_dataset import make_aligs, transfo_to_filenames, TacredDataset
#from DtSets.make_dataset import FusionElimination as FILT, AligDataset
from tacred.DataLoader import tacredGraphSampler, tacredStratifiedGraphSampler, tacredDataLoader, tacredGPUDataLoader
from tacred.models import tacred_GAT_MODEL, tacredTRAINER


def essai(rep):
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


def make_dataloaders(transfo_name, train, dev, test, forceGPU=False):
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
            regSampler = tacredGraphSampler(ds, avg_num=5000, shuffle=False)
            loader = CLASSE(
                ds,
                batch_sampler=regSampler,
                follow_batch=["msk"],
                num_workers=0,
                forceGPU=forceGPU
            )
            dataloaders += (loader,)

    return dataloaders




def essai_train():
    alig_file, rep_ds_grph, id_model = transfo_to_filenames("robertaBase", True)
    print("rep_ds_grph:", rep_ds_grph)

    dsTRAIN = TacredDataset(rep_ds_grph + "train",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="train",
                            device="cpu",
                            tokenHF=None)
    d48125 = dsTRAIN[48125]
    dsDEV = TacredDataset(rep_ds_grph + "dev",
                            alig_file,
                            transform=None,
                            QscalK=True,
                            split="dev",
                            device="cpu",
                            tokenHF=None)
    
    #dsTEST = TacredDataset(rep_ds_grph + "test",
    #                        alig_file,
    #                        transform=None,
    #                        QscalK=True,
    #                        split="test",
    #                        device="cpu",
    #                        tokenHF=None)
    

    stratSampler = tacredStratifiedGraphSampler(dsTRAIN, avg_num=5000)
    regSampler   = tacredGraphSampler(dsDEV, avg_num=5000, shuffle=False)
    ldTRAIN = tacredDataLoader(
        dsTRAIN,
        batch_sampler=stratSampler,
        follow_batch=["msk"],
        num_workers=1
    )
    ldDEV   = tacredDataLoader(
        dsDEV,
        batch_sampler=regSampler,
        follow_batch=["msk"],
        num_workers=0
    )
    print("CRÉATION DU MODÈLE")
    model = tacred_GAT_MODEL(dim_in = 144,
                 dim_h1 = 144,
                 dim_h2 = 144,
                 heads = 1,
                 nb_couches = 4,
                 rang_sim = 8,
                 dropout_p = 0.3,
                 nb_classes = 42)
    
    print("Nombre de paramètres: %d"%model.count_params())
    
        
    print("CRÉATION DE L'ENTRAÎNEUR")
    
    trainer = tacredTRAINER(model=model, epochs=20)
    print("DÉMARRAGE de L'ENTRAÎNEMENT")
    trainer.train(ldTRAIN, ldDEV)
    print("TERMINÉ")
    
    



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


    

if __name__ == "__main__":
    #rep = "/home/charpentier/projets/json_tacred/"
    #essai(rep)
    essai_train()
    #essai_train_tacred()
    #essai2()
