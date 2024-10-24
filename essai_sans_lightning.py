import torch
import torch.nn as nn
from torch import utils
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from make_dataset import AligDataset, PermutEdgeDataset, EdgeDataset, EdgeDatasetMono

from torch.utils.tensorboard import SummaryWriter

class Classif_Logist(nn.Module):
    # modèle linéaire à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR. 
    def __init__(self, dim, nb_classes, lr, freqs=None):
        super(Classif_Logist, self).__init__()
        if not freqs is None:
            if type(freqs) is list:
                assert len(freqs) == nb_classes
                freqs = torch.Tensor(freqs)
            assert freqs.shape == (nb_classes,)
        self.lin = nn.Linear(dim, nb_classes, bias=True)
        self.freqs = freqs
        self.lr = lr
        self.pondus = freqs.max() / freqs
        #Calcul de la pondération. La classe majoritaire prendra la pondération 1,0.
        self.loss = nn.CrossEntropyLoss(weight = self.pondus, reduction="mean")

    def forward(self, X):
        # Ni softmax ni log softmax.
        # utiliser la perte "Cross entropy à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return self.lin(X)
    

def trainModel(modele, train_ld, val_ld, lr, nb_epochs):
    cible = "roles"
    writer = SummaryWriter('./logs_sans_lightning/XP_1')
    optimizer = optim.Adam(modele.parameters(), lr=lr)
    step = 0
    for epc in range(nb_epochs):
        print("Epoch n° %d"%epc)
        for batch in tqdm(train_ld):
            optimizer.zero_grad()
            logits = modele.forward(batch["X"])
            perte = modele.loss(logits, batch[cible])
            perte.backward()
            optimizer.step()

            writer.add_scalar("perte_entrainement", perte.cpu().item(), step)
            step += 1

        optimizer.zero_grad()
        print("Validation")
        with torch.no_grad():
            cumul_perte = []
            for batch in tqdm(val_ld):
                logits = modele.forward(batch["X"])
                perte = modele.loss(logits, batch[cible])
                cumul_perte.append(perte.cpu().item())
            perte = np.array(cumul_perte).mean()
            print("perte validation : %f"%perte)
        writer.add_scalar("perte_validation", perte, step)
        writer.flush()
            

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
    lr = 1.e-5

    modele = Classif_Logist(dimension, nb_classes, lr=lr, freqs=freqs)

    print("Début de l’entrainement")
    train_loader = utils.data.DataLoader(DARtr, batch_size=64, num_workers=8)
    valid_loader = utils.data.DataLoader(DARdv, batch_size=32, num_workers=8)
    trainModel(modele, train_loader, valid_loader, lr, nb_epochs=50)
    print("TERMINÉ.")
    
if __name__ == "__main__":
    main()
