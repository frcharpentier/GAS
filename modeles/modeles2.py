import torch
import matplotlib.pyplot as plt
from torch import optim, nn, utils
import torch.nn.functional as NNF
import lightning as LTN
import random
from torchmetrics.aggregation import CatMetric
from torchmetrics import Accuracy as TMAccuracy
from torch_geometric.nn import GATv2Conv

from DtSets.make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir

class torchmodule_Classif_Lin(torch.nn.Module):
    def __init__(self, dim, nb_classes):
        super(torchmodule_Classif_Lin, self).__init__()
        self.dim = dim
        self.nb_classes = nb_classes
        self.hparams = {"dim": dim, "nb_classes": nb_classes}
        self.lin = nn.Linear(dim, nb_classes, bias=True)

    def forward(self, X):
        # Ni softmax ni log softmax.
        # utiliser la perte "Cross entropy à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return self.lin(X)

class torchmodule_Classif_Bil_Sym(torch.nn.Module):
    def __init__(self, dim, nb_classes, rang=2):
        super(torchmodule_Classif_Bil_Sym, self).__init__()
        self.dim = dim
        self.nb_classes = nb_classes
        self.rang = rang
        self.hparams = {"dim": dim, "nb_classes": nb_classes, "rang": rang}

        self.weight = nn.Parameter(torch.empty(nb_classes, rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(nb_classes, rang))
        nn.init.xavier_normal_(self.diag)
        self.bias = nn.Parameter(torch.empty(nb_classes, ))
        nn.init.normal_(self.bias)

    def forward(self, X):
        # X est un tenseur de format (b, dim, 2)
        # On commence par le transformer en un tenseur (b, 1, dim, 2)
        # Qui s’ajustera (par conduplication) à un tenseur (b, nb_classes, dim, 2)

        X = X.unsqueeze(-3)
        M = self.weight.unsqueeze(0)
        B = self.bias.unsqueeze(0)
        # X est désormais un tenseur de format (b, 1, dim, 2)
        # et M est un tenseur de format (1, nb_classes, rang, dim)
        # Si on l’ajuste (par conduplication) à un tenseur (b, nb_classes, rang, dim)
        # alors M et X sont envisageables comme des tenseurs (b, nb_classes)
        # de matrices (rang, dim) pour M et (dim, 2) pour X.
        # Si on multiple (matriciellement) point à point les éléments de ces tenseurs,
        # on obtient un tenseur (b, nb_classes) de matrices (rang, 2),
        # C’est-à-dire un tenseur (b, nb_classes, rang, 2).

        Y = torch.matmul(M, X)
        # Y est un tenseur de format (b, nb_classes, rang, 2)
        
        Y0 = Y[...,0] * (self.diag.unsqueeze(0))
        # Extraction de la colonne zéro, et multiplication point à point par diag,
        # (Ce qui revient à multiplier matriciellement par une matrice diagonale)
        # Y0 est un tenseur (b, nb_classes, rang)
        Y1 = Y[...,1]
        # Extraction de la colonne 1
        # Y1 est un tenseur (b, nb_classes, rang)
        Y = (Y0*Y1).sum(axis=-1)
        # Y est un tenseur (b, nb_classes)
        Y = Y+B # ajout du biais

        # Ni softmax ni log softmax.
        # Utiliser la perte "Entropie croisée à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return Y
    
class torchmodule_Classif_Bil_Sym_2(torch.nn.Module):
    def __init__(self, dim, nb_classes, rang=2):
        super(torchmodule_Classif_Bil_Sym_2, self).__init__()
        self.dim = dim
        self.nb_classes = nb_classes
        self.rang = rang
        self.hparams = {"dim": dim, "nb_classes": nb_classes, "rang": rang}
        self.weight = nn.Parameter(torch.empty(nb_classes, rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(nb_classes, rang))
        nn.init.xavier_normal_(self.diag)
        self.bias_vecto = nn.Parameter(torch.empty(nb_classes, dim))
        nn.init.xavier_normal_(self.bias_vecto)
        self.bias0 = nn.Parameter(torch.empty(nb_classes, ))
        nn.init.normal_(self.bias0)

    def forward(self, X):
        # X est un tenseur de format (b, dim, 2)
        # On commence par le transformer en un tenseur (b, 1, dim, 2)
        # Qui s’ajustera (par conduplication) à un tenseur (b, nb_classes, dim, 2)

        X = X.unsqueeze(-3)
        # bias_vecto est un tenseur de format (nb_classes, dim)
        # On en faait un tenseur (1, nb_clases, dim, 1)
        # Pour le soustraire à X (Par conduplication)
        B = self.bias_vecto.unsqueeze(0).unsqueeze(-1)
        Xb = X-B

        M = self.weight.unsqueeze(0)
        # Xb est un tenseur de format (b, nb_classes, dim, 2)
        # et M est un tenseur de format (1, nb_classes, rang, dim)
        # Si on l’ajuste (par conduplication) à un tenseur (b, nb_classes, rang, dim)
        # alors M et Xb sont envisageables comme des tenseurs (b, nb_classes)
        # de matrices (rang, dim) pour M et (dim, 2) pour X.
        # Si on multiple (matriciellement) point à point les éléments de ces tenseurs,
        # on obtient un tenseur (b, nb_classes) de matrices (rang, 2),
        # C’est-à-dire un tenseur (b, nb_classes, rang, 2).

        Y = torch.matmul(M, Xb)
        # Y est un tenseur de format (b, nb_classes, rang, 2)
        
        Y0 = Y[...,0] * (self.diag.unsqueeze(0))
        # Extraction de la colonne zéro, et multiplication point à point par diag,
        # (Ce qui revient à multiplier matriciellement par une matrice diagonale)
        # Y0 est un tenseur (b, nb_classes, rang)
        Y1 = Y[...,1]
        # Extraction de la colonne 1
        # Y1 est un tenseur (b, nb_classes, rang)
        Y = (Y0*Y1).sum(axis=-1)
        # Y est un tenseur (b, nb_classes)

        #ajout du biais
        Y = Y + (self.bias0.unsqueeze(0))
        

        # Ni softmax ni log softmax.
        # Utiliser la perte "Entropie croisée à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return Y

class torchmodule_Classif_Bil_Antisym(torch.nn.Module):
    def __init__(self, dim, rang=2):
        # On n’a que deux classes, a priori.
        super(torchmodule_Classif_Bil_Antisym, self).__init__()
        self.dim = dim
        self.rang = rang
        self.hparams = {"dim": dim, "rang": rang}
        self.nb_classes = "binary"
        self.weight = nn.Parameter(torch.empty(rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.antisym = nn.Parameter(torch.empty(rang,rang))
        nn.init.xavier_normal_(self.antisym)
        self.bias_vecto = nn.Parameter(torch.empty(dim,))
        nn.init.normal_(self.bias_vecto)
        
        #self.save_hyperparameters()
        

    def forward(self, X):
        # X : (b, dim, 2)
        W = self.weight.unsqueeze(0)
        # w : (1, r, dim)
        B = self.bias_vecto.unsqueeze(0).unsqueeze(-1)
        # B : (1, dim, 1) et X-B : (b, dim, 2)
        Y = torch.matmul(W, (X-B)) # Y = W·(X-B)
        # Y : (b, r, 2)
        B1 = torch.matmul(W, B) # shape: (1, r, 1) B1=W·B
        A = self.antisym.triu(1)
        A = A - (A.T)
        A = A.unsqueeze(0) # shape: (1, r, r)
        Y1 = Y[...,1].unsqueeze(-1) # shape: (b, r, 1) Y1 = W·(X1-B)
        Y1 = torch.matmul(A, Y1)    # shape: (b, r, 1) Y1 = A·W·(X1-B)
        B2 = torch.matmul(A, B1) # shape: (1, r, 1)  B2 = A·W·B
        tBMB = (B1.squeeze() * B2.squeeze()).sum() # shape: ()
        # tBMB = tB·tW·A·W·B

        Y0 = Y[...,0].unsqueeze(-2)  # shape: (b, 1, r) Y2 = t(W·(X0-B))
        Y = torch.matmul(Y0, Y1)     # shape: (b, 1, 1) Y = t(X0-B)·tW·A·W·(X1-B)
        Y = Y.reshape(-1)            # shape: (b,)
        Y = Y + tBMB.unsqueeze(0)    # shape: (b,) Y = t(X0-B)·tW·A·W·(X1-B) + tB·tW·A·W·B
        return Y
    
class torchmodule_Classif_Bil_Antisym_2(torch.nn.Module):
    def __init__(self, dim, rang=2):
        # On n’a que deux classes, a priori.
        super(torchmodule_Classif_Bil_Antisym_2, self).__init__()
        assert (rang % 2) == 0
        self.dim = dim
        self.rang = rang
        self.hparams = {"dim": dim, "rang": rang}
        self.nb_classes = "binary"
        self.weight = nn.Parameter(torch.empty(rang, dim))
        nn.init.xavier_normal_(self.weight)
        self.antisym = nn.Parameter(torch.empty(rang//2,))
        nn.init.normal_(self.antisym)
        self.bias_vecto = nn.Parameter(torch.empty(dim,))
        nn.init.normal_(self.bias_vecto)

        idx = torch.LongTensor(
            [2*(i//2) + 1-(i%2) for i in range(rang)]
            )
        #[1,0,3,2,5,4,...] (indexation pour intervertir les lignes deux à deux)

        self.register_buffer("idx", idx)

        
        #self.save_hyperparameters()
        

    def forward(self, X):
        # X : (b, dim, 2)

        Av = self.antisym.unsqueeze(-1) * torch.tensor([[1, -1]], requires_grad=False)
        # Av : (r/2, 2), la deuxième colonne est l’opposé de la première.
        Av = Av.reshape(self.rang).unsqueeze(-1) # shape: (r, 1)

        W = self.weight
        # W : (r, dim)
        Wperm = torch.index_select(W, 0, self.idx)
        # Permutation des lignes deux à deux.
        AW = Av*Wperm # shape : (r, dim) AW = A·W
        AW = AW.unsqueeze(0) # shape : (1, r, dim)

        B = self.bias_vecto.unsqueeze(0).unsqueeze(-1)
        # B : (1, dim, 1)
        Xb = X-B # X-B : (b, dim, 2)
        X1b = Xb[...,1].unsqueeze(-1) # shape : (b, dim, 1) X1b = X1 - B
        X0b = Xb[...,0].unsqueeze(-1) # shape : (b, dim, 1) X0b = X0 - B
        WX0b = torch.matmul(W.unsqueeze(0), X0b) #shape : (b, r, 1) WX0b = W·(X0-B)

        AWX1b = torch.matmul(AW, X1b) # shape: (b, r, 1) AWX1b = A·W·(X1-B)
        Y = (WX0b * AWX1b).reshape(-1, self.rang).sum(axis=1) # shape : (b,) Y = t(X0-B)·tW·A·W·(X1-B)

        AWB = torch.matmul(AW, B) #shape : (1, r, 1) AWB = A·W·B
        WB = torch.matmul(W.unsqueeze(0), B) #shape : (1, r, 1) WB = W·B
        tBMB = (WB*AWB).reshape(-1, self.rang).sum(axis=1) # shape : (1,) tBMB = tB·tW·A·W·B

        Y = Y + tBMB # shape : (b,) Y = t(X0-B)·tW·A·W·(X1-B) + tB·tW·A·W·B

        return Y

class torchmodule_GAT_role_classif(torch.nn.Module):
    def __init__(self, dim_in: int,
                 dim_h1:int,
                 dim_h2: int,
                 heads: int,
                 nb_couches: int,
                 rang_sim: int,
                 dropout_p: float,
                 nb_classes: int):
                 
        # dim_in : Les nœuds du graphe adjoint sont décrits par une
        #          paire de vecteurs descripteurs. dim_in est leur dimension.
        # dim_h1 : dimension de la sortie de la première couche symétrique
        # rang_sim :rang pour approximation des matrices symétriques
        #
        # nb_couches : nombre de couches GATv2
        # heads : nombre de tête (param commun à toutes les couches GATv2)
        # La dimension de sortie de chaque couche est nb_têtes * dim_h2
        # On termine par une couche de classification linéaire.
        super(torchmodule_GAT_role_classif, self).__init__()

        self.hparams = {"dim_in": dim_in,
                        "dim_h1" : dim_h1,
                        "dim_h2" : dim_h2,
                        "heads" : heads,
                        "nb_couches": nb_couches,
                        "rang_sim": rang_sim,
                        "dropout_p": dropout_p,
                        "nb_classes": nb_classes}

        #if not freqs is None:
        #    if type(freqs) is list:
        #        assert len(freqs) == nb_classes
        #        freqs = torch.Tensor(freqs)
        #    assert freqs.shape == (nb_classes,)

        
            
        self.weight = nn.Parameter(torch.empty(dim_h1, rang_sim, dim_in))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(dim_h1, rang_sim))
        nn.init.xavier_normal_(self.diag)
        self.bias_vecto = nn.Parameter(torch.empty(dim_h1, dim_in))
        nn.init.xavier_normal_(self.bias_vecto)
        self.bias0 = nn.Parameter(torch.empty(dim_h1, ))
        nn.init.normal_(self.bias0)
        self.nb_couches = nb_couches

        self.dim_h1 = dim_h1
        self.dim_h2 = dim_h2
        self.nb_classes = nb_classes

        self.gat_layers = nn.ModuleList()
        if dropout_p is None:
            self.dropout_p = 0
            #dropout_p = 0
        else:
            self.dropout_p = dropout_p

        if nb_couches > 0:
        
            self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = dim_h2, heads=heads))
            for _ in range(1, nb_couches):
                self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = dim_h2, heads=heads))
        else:
            pass

        # Ajout d’une classification linéaire finale
        self.lin = nn.Linear(heads*dim_h2, self.nb_classes, bias=True)


    def forward(self, grph):
        X, edge_idx = grph
        # X est un tenseur de format (b, dim_in, 2)
        # On commence par le transformer en un tenseur (b, 1, dim_in, 2)
        # Qui s’ajustera (par distributivité) à un tenseur (b, dim_h1, dim_in, 2)

        X = X.unsqueeze(-3)
        # bias_vecto est un tenseur de format (dim_h1, dim_in)
        # On en fait un tenseur (1, dim_h1, dim_in, 1)
        # Pour le soustraire à X (Par distributivité)
        B = self.bias_vecto.unsqueeze(0).unsqueeze(-1)
        Xb = X-B # shape : (b, dim_h1, dim_in, 2)

        M = self.weight.unsqueeze(0)
        # Xb est désormais un tenseur de format (b, dim_h1, dim_in, 2)
        # et M est un tenseur de format (1, dim_h1, rang, dim_in)
        # Si on l’ajuste (par distributivité) à un tenseur (b, dim_h1, rang, dim_in)
        # alors M et Xb sont envisageables comme des tenseurs (b, dim_h1)
        # de matrices (rang, dim_in) pour M, ou (dim_in, 2) pour X.
        # Si on multiple (matriciellement) point à point les éléments de ces tenseurs,
        # on obtient un tenseur (b, dim_h1) de matrices (rang, 2),
        # C’est-à-dire un tenseur (b, dim_h1, rang, 2).

        Y = torch.matmul(M, Xb)
        # Y est un tenseur de format (b, dim_h1, rang, 2)
        
        Y0: torch.Tensor = Y[...,0] * (self.diag.unsqueeze(0))
        # Extraction de la colonne zéro, et multiplication point à point par diag,
        # (Ce qui revient à multiplier matriciellement par une matrice diagonale)
        # Y0 est un tenseur (b, dim_h1, rang)
        Y1: torch.Tensor = Y[...,1]
        # Y1 est un tenseur (b, dim_h1, rang)

        H: torch.Tensor = torch.sum((Y0*Y1), dim=-1)
        # H est un tenseur (b, dim_h1)
        # ajout du biais :
        H = H + (self.bias0.unsqueeze(0))
        
        if self.nb_couches == 0:
            H = nn.functional.relu(H)
        elif self.dropout_p:
            for gat in self.gat_layers:
                Hgat = H
                Hgat = nn.functional.dropout(Hgat, self.dropout_p, training=self.training)
                Hgat = gat(Hgat, edge_idx)
                H = H + Hgat # connexion résiduelle
        else:
            for gat in self.gat_layers:
                Hgat = H
                Hgat = gat(Hgat, edge_idx)
                H = H + Hgat # connexion résiduelle

        # Classification finale
        Y = self.lin(H) # shape: (b, nb_classes)

        # Ni softmax ni log softmax.
        # Utiliser la perte "Entropie croisée à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return Y
    
class torchmodule_GAT_sans_GAT(torchmodule_GAT_role_classif):
    def __init__(self, dim_in: int,
                 dim_h1:int,
                 #dim_h2: int,
                 #heads: int,
                 #nb_couches: int,
                 rang_sim: int,
                 #dropout_p: float,
                 nb_classes: int,
                 ):
    
        super(torchmodule_GAT_sans_GAT, self).__init__(
                dim_in = dim_in,
                dim_h1 = dim_h1,
                dim_h2 = dim_h1,
                heads = 1,
                nb_couches = 0,
                rang_sim = rang_sim,
                dropout_p = 0.,
                nb_classes = nb_classes,
        )
    
def make_GAT_model(nom_modele, **kwargs):
    assert nom_modele in ["tm_GAT", "tm_GAT_sans_GAT",
                          "torchmodule_GAT_role_classif",
                          "torchmodule_GAT_sans_GAT"]
    if nom_modele in ["tm_GAT", "torchmodule_GAT_role_classif"]:
        dim_in = kwargs["dim_in"]
        dim_h1 = kwargs["dim_h1"]
        dim_h2 = kwargs["dim_h2"]
        heads = kwargs["heads"]
        nb_couches = kwargs["nb_couches"]
        rang_sim   = kwargs["rang_sim"]
        dropout_p = kwargs["dropout_p"]
        nb_classes = kwargs["nb_classes"]
        return torchmodule_GAT_role_classif(dim_in=dim_in,
            dim_h1 = dim_h1,
            dim_h2 = dim_h2,
            heads = heads,
            nb_couches = nb_couches,
            rang_sim = rang_sim,
            dropout_p = dropout_p,
            nb_classes = nb_classes)
    else:
        dim_in = kwargs["dim_in"]
        dim_h1 = kwargs["dim_h1"]
        rang_sim   = kwargs["rang_sim"]
        nb_classes = kwargs["nb_classes"]
        return torchmodule_GAT_sans_GAT(dim_in = dim_in,
            dim_h1 = dim_h1,
            rang_sim = rang_sim,
            nb_classes = nb_classes)

class INFERENCE(LTN.LightningModule):
    def __init__(self, modele,
                 f_features = 'lambda b: b["X"]',
                 f_target='lambda b: b["roles"]',
                 f_msk = "",
                 lr=1.e-5,
                 perte = "CrossEntropyLoss",
                 freqs=None,
                 no_bal_acc=False):
        super(INFERENCE, self).__init__()
        nb_classes = modele.nb_classes
        if nb_classes in (None, "binaire", "binary"):
            nb_classes = 2
            self.binary = True
        else:
            self.binary = False
        if not freqs is None:
            if type(freqs) is list:
                assert len(freqs) == nb_classes
                freqs = torch.Tensor(freqs)
            assert freqs.shape == (nb_classes,)
        self.add_module("modele", modele)
        # Il est nécessaire d’appeler la fonction add_module plutôt que la 
        # simple assignation self.modele = modele, parce que le constructeur
        # n’est pas appelé à la volée.
        assert f_features.startswith("lambda b:")
        assert f_target.startswith("lambda b:")
        assert f_msk == "" or f_msk.startswith("lambda b:")
        self.f_features = eval(f_features)
        self.f_target = eval(f_target)
        if f_msk == "":
            self.f_msk = False
        else:
            self.f_msk = eval(f_msk)
        self.lr = lr
        self.freqs = freqs
        self.save_hyperparameters(ignore="modele")
        assert perte in nn.__dict__
        if freqs is None:
            self.loss = (nn.__dict__[perte])(reduction="mean")
        else:
            self.pondus = freqs.max() / freqs
            self.loss = (nn.__dict__[perte])(weight = self.pondus, reduction="mean")
        self.acc_metric = TMAccuracy(task="multiclass", num_classes=nb_classes)
        self.compute_bal_acc = not(no_bal_acc)
        if self.compute_bal_acc:
            self.balacc_metric = TMAccuracy(task="multiclass", num_classes=nb_classes, average="macro")
        
    def forward(self, *args, **kwargs):
        return self.modele.forward(*args, **kwargs)
    
    def predict_step(self, batch, batch_idx):
        logits = self.forward(self.f_features(batch))
        if self.f_msk:
            msk = self.f_msk(batch)
            logits = logits[msk]
        if self.binary:
            return ((logits > 0)*1).to(device="cpu")
        else:
            return logits.argmax(axis=1).to(device="cpu")
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(self.f_features(batch))
        truth = self.f_target(batch)
        if self.f_msk:
            msk = self.f_msk(batch)
            logits = logits[msk]
            truth = truth[msk].to(dtype=torch.int64)
        perte = self.loss(logits, truth)
        self.log("train_loss", perte)
        return perte
    
    def validation_step(self, batch, batch_idx):
        #Boucle de validation
        logits = self.forward(self.f_features(batch))
        truth = self.f_target(batch)
        if self.f_msk:
            msk = self.f_msk(batch)
            logits = logits[msk]
            truth = truth[msk].to(dtype=torch.int64)

        (bs,) = truth.shape

        perte = self.loss(logits, truth)
        # Lu dans la documentation :
        #Si on l'appelle depuis la fonction validation_step, la fonction log
        #accumule les valeurs pour toute l'époché
        self.log("val_loss", perte, batch_size=bs, on_step=False, on_epoch=True)

        if self.binary:
            preds = (logits > 0) * 1
        else:
            preds = logits.argmax(axis=1)
        self.acc_metric(preds, truth)
        self.log("val_acc", self.acc_metric, on_step=False, on_epoch=True)
        if self.compute_bal_acc:
            self.balacc_metric(preds, truth)
            self.log("val_bal_acc", self.balacc_metric, on_step=False, on_epoch=True)
        
        
        

    #def on_test_start(self):
    #    self.accuPred = CatMetric()
    #    self.accuTrue = CatMetric()
    
    #def test_step(self, batch, batch_idx):
    #    logits = self.forward(self.f_features(batch))
    #    if self.binary:
    #        roles_pred = ((logits > 0)*1).to(device="cpu")
    #    else:
    #        roles_pred = logits.argmax(axis=1).to(device="cpu")
    #    self.accuPred.update(roles_pred)
    #    self.accuTrue.update(self.f_target(batch))

    #def on_test_end(self):
    #    roles = self.accuTrue.compute().cpu().numpy()
    #    roles_pred = self.accuPred.compute().cpu().numpy()
        

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    

