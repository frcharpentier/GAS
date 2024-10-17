import torch
import matplotlib.pyplot as plt
from torch import optim, nn, utils
import torch.nn.functional as NNF
import lightning as LTN
import random
from torchmetrics.aggregation import CatMetric

from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir



class Classif_Logist(LTN.LightningModule):
    # modèle linéaire à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR. 
    def __init__(self, dim, nb_classes, cible, freqs=None):
        super(Classif_Logist, self).__init__()
        if not freqs is None:
            if type(freqs) is list:
                assert len(freqs) == nb_classes
                freqs = torch.Tensor(freqs)
            assert freqs.shape == (nb_classes,)
        self.lin = nn.Linear(dim, nb_classes, bias=True)
        self.freqs = freqs
        self.cible = cible
        self.pondus = freqs.max() / freqs
        #Calcul de la pondération. La classe majoritaire prendra la pondération 1,0.
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss(weight = self.pondus, reduction="mean")

    def forward(self, X):
        # Ni softmax ni log softmax.
        # utiliser la perte "Cross entropy à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        #X = X.to(dtype=torch.float32)
        return self.lin(X)
    
    def predict_step(self, batch, batch_idx):
        #X, roles, sens, ARGn = batch
        logits = self.forward(batch["X"])
        return logits.argmax(axis=1).to(device="cpu")
    
    def training_step(self, batch, batch_idx):
        #X, roles, sens, ARGn = batch
        logits = self.forward(batch["X"])
        perte = self.loss(logits, batch[self.cible])
        #perte = self.loss(logits, roles.to(torch.long))
        self.log("train_loss", perte)
        return perte
    
    def validation_step(self, batch, batch_idx):
        #Boucle de validation
        #X, roles, sens, ARGn = batch
        logits = self.forward(batch["X"])
        #perte = self.loss(logits, roles.to(torch.long))
        perte = self.loss(logits, batch[self.cible])
        # Lu dans la documentation :
        #Si on l'appelle depuis la fonction validation_step, la fonction log
        #accumule les valeurs pour toute l'époché
        self.log("val_loss", perte)
        

    def on_test_start(self):
        self.accuPred = CatMetric()
        self.accuTrue = CatMetric()
    
    def test_step(self, batch, batch_idx):
        #X, roles, sens, ARGn = batch
        logits = self.forward(batch["X"])
        roles_pred = logits.argmax(axis=1).to(device="cpu")
        self.accuPred.update(roles_pred)
        self.accuTrue.update(batch[self.cible])

    def on_test_end(self):
        roles = self.accuTrue.compute().cpu().numpy()
        roles_pred = self.accuPred.compute().cpu().numpy()
        #fig = dessin_matrice_conf(roles, roles_pred, self.noms_classes)
        #self.logger.experiment.add_figure("Matrice de Confusion", fig, self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-5)
        #optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    
class Classif_Bil_Sym(nn.Module):
    # modèle bilinéaire symétrique à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR.

    def __init__(self, dim, nb_classes, rang=2, freqs = None):
        super(Classif_Bil_Sym, self).__init__()
        if freqs:
            assert freqs.shape == (nb_classes,)
        self.weight = nn.Parameter(torch.empty(nb_classes, rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(nb_classes, rang))
        nn.init.xavier_normal_(self.diag)
        self.bias = nn.Parameter(torch.empty(nb_classes, ))
        nn.init.normal_(self.bias)
        self.freqs = freqs
        self.pondus = freqs.max() / freqs
        #Calcul de la pondération. La classe majoritaire prendra la pondération 1,0.
        self.nb_classes = nb_classes

    def forward(self, X):
        # X est un tenseur de format (b, dim, 2)
        # On commence par le transformer en un tenseur (b, 1, dim, 2)
        # Qui s’ajustera (par conduplication) à un tenseur (b, nb_classes, dim, 2)

        X = X.unsqueeze(-3)
        M = self.weight.unsqueeze(0)
        # X est désormais un tenseur de format (b, 1, dim, 2)
        # et M est un tenseur de format (1, nb_classes, rang, dim)
        # Si on l’ajuste (par conduplication) à un tenseur (b, nb_classes, rang, dim)
        # alors M et X sont envisageables comme des tenseurs de matrices (b, nb_classes)
        # Si on multiple (matriciellement) point à point les éléments de ces tenseurs,
        # on obtient un tenseur (b, nb_classes) de matrices (rang, 2),
        # C’est-à-dire un tenseur (b, nb_classes, rang, 2).

        
        Y = torch.matmul(M, X)
        # Y est un tenseur de format (b, nb_classes, rang, 2)
        
        Y0 = (Y[...,0] * self.diag.unsqueeze(0)).unsqueeze(-2)
        # Y0 est un tenseur (b, nb_classes, 1, rang)
        Y1 = Y[...,1].unsqueeze(-1)
        # Y1 est un tenseur (b, nb_classes, rang, 1)
        Y = torch.matmul(Y0,Y1)
        # Y est un tenseur (b, nb_classes,1,1)
        return Y.reshape(-1, self.nb_classes)


class Classif_Bil_Sym_0(nn.Module):
    # modèle bilinéaire symétrique à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR.

    def __init__(self, dim, nb_classes, rang=2, freqs = None):
        super(Classif_Bil_Sym_0, self).__init__()
        if freqs:
            assert freqs.shape == (nb_classes,)
        self.weight = nn.Parameter(torch.empty(nb_classes, dim, rang))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(nb_classes, rang))
        nn.init.xavier_normal_(self.diag)
        self.bias = nn.Parameter(torch.empty(nb_classes, ))
        nn.init.normal_(self.bias)
        self.freqs = freqs
        
    def forward(self, X):
        # X est un tenseur de format (b, dim, 2)
        # weight est défini dans __init__.
        X1 = X[:,:,0]
        X2 = X[:,:,1]
        Z = torch.einsum("bk, nkr -> bnr", X1, self.weight)
        # Z est un tenseur de format (b, nb_classes, rang)
        Z1 = Z * torch.unsqueeze(self.diag, 0)
        # multiplication par la matrice diagonale
        
        Z2 = torch.einsum("bk, nkr -> bnr", X2, self.weight)
        # Z2 est un tenseur de format (b, nb_classes, rang)
        Y = (Z1 * Z2).sum(axis=2)
        # Équivalent (mais sans doute plus rapide) à
        # torch.einsum("bnr, bnr -> bn", Z1, Z2)
        Y = Y + self.bias.view(1,-1)
        return Y
    

    
#class AntiSymetrique(nn.Module):
#    def forward(self, X):
#        A = X.triu(1)
#        return A - (A.transpose(-1,-2))

class Classif_Bil_Antisym(nn.Module):
    # modèle bilinéaire symétrique à utiliser avec le dataset d’arêtes
    # étiquettes = sens

    def __init__(self, dim, rang=2, freqs = None):
        # On n’a que deux classes, a priori.
        super(Classif_Bil_Antisym, self).__init__()
        if freqs:
            assert freqs.shape == (2,)
        self.weight = nn.Parameter(torch.empty(rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.antisym = nn.Parameter(torch.empty(rang,rang))
        nn.init.xavier_normal_(self.antisym)
        self.freqs = freqs

    def forward(self, X):
        # X : (b, dim, 2)
        # w_unsq : (1, r, dim)
        Y = torch.matmul(self.weight.unsqueeze(0), X)
        # Y : (b, r, 2)
        A = self.antisym.triu(1)
        A = A - (A.T)
        Y1 = Y[...,1]
        Y1 = torch.matmul(A.unsqueeze(0), Y1.unsqueeze(-1))
        Y0 = Y[...,0].unsqueeze(-2)
        Y = torch.matmul(Y0, Y1)
        return Y.reshape(-1)

class Classif_Bil_Antisym_0(nn.Module):
    # modèle bilinéaire symétrique à utiliser avec le dataset d’arêtes
    # étiquettes = sens

    def __init__(self, dim, rang=2, freqs = None):
        # On n’a que deux classes, a priori.
        super(Classif_Bil_Antisym_0, self).__init__()
        if freqs:
            assert freqs.shape == (2,)
        self.weight = nn.Parameter(torch.empty(dim, rang))
        nn.init.xavier_normal_(self.weight)
        self.antisym = nn.Parameter(torch.empty(rang,rang))
        nn.init.xavier_normal_(self.antisym)
        self.freqs = freqs

    def forward(self, X):
        # X est un tenseur de format (b, dim, 2)
        # weight est défini dans __init__.
        X1 = X[:,:,0]
        X2 = X[:,:,1]
        Z = X1 @ self.weight
        # Équivalent plus rapide de Z = torch.einsum("bk, kr -> br", X1, self.weight)
        # Z est un tenseur de format (b, rang)

        A = self.antisym.triu(1)
        A = A - (A.T)

        Z1 = Z @ A
        # Multiplication par la matrice antisymétrique de rang R.

        Z2 = X2 @ self.weight
        # Équivalent plus rapide de Z2 = torch.einsum("bk, kr -> br", X1, self.weight)
        # Z2 est un tenseur de format (b, rang)
        Y = (Z1 * Z2).sum(axis=1)
        # Équivalent (mais sans doute plus rapide) à
        # torch.einsum("br, br -> b", Z1, Z2)
        
        return Y.reshape(-1)
    

def test_sym_antisym():
    dim = 5
    rang = 2
    batch = 97
    nb_classes = 3

    def changer_sens_X(X, direction):
        # Direction est un tenseur qui ne contient que des zéros et des uns.
        assert X.shape[0] == direction.shape[0]
        if not direction.dtype == torch.long:
            direction = direction.to(dtype = torch.long)
        indices = torch.column_stack((direction, 1-direction)).view(-1,1,2)
        #assert all(indices[i, 0, 0] == T[i] for i in range(self.Nadj))
        #assert all(indices[i, 0, 1] == 1-T[i] for i in range(self.Nadj))
        X = torch.take_along_dim(X, indices, dim=2)
        return X

    X = torch.randn((batch, dim, 2))
    print("Test symétrique :")
    modele = Classif_Bil_Sym(dim, nb_classes, rang)
    #modele0 = Classif_Bil_Sym_0(dim, nb_classes, rang)
    with torch.no_grad():
        Yref = modele.forward(X)
    for _ in range(20):
        index = torch.randint(0,2, (batch,), dtype=torch.long)
        Xs = changer_sens_X(X, index)
        with torch.no_grad():
            Y = modele.forward(Xs)
            #Y0 = modele0.forward(Xs)
        #assert torch.allclose(Y,Y0)
        assert torch.allclose(Y, Yref) #(Y == Yref).all().item()
    print("Test Symétrique OK.")
    print()
    print("Test antisymétrique :")
    modele = Classif_Bil_Antisym(dim, rang)
    with torch.no_grad():
        Yref = modele.forward(X)
    for _ in range(20):
        index = torch.randint(0,2, (batch,), dtype=torch.long)
        signe = -(2*index) + 1
        Xs = changer_sens_X(X, index)
        with torch.no_grad():
            Y = modele.forward(Xs)
        Ys = Y*signe
        assert torch.allclose(Ys, Yref) #(Y == Yref).all().item()
    print("Test Antisymétrique OK.")


def calcul():
    ds_train = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    noms_classes = [k for k in ds_train.filtre.alias]
    filtre = ds_train.filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al, noms_classes))
    filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
    filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))

if __name__ == "__main__":
    torch.manual_seed(53)
    random.seed(53)
    #test_sym_antisym()
    dim = 5
    rang = 2
    batch = 97
    nb_classes = 3
    modele = Classif_Bil_Sym(dim, nb_classes, rang)
    for N, T in modele.named_parameters():
        print(N, T)


    