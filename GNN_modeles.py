import torch
from torch import optim, nn
import torch.nn.functional as NNF
import lightning as LTN
import random
from torchmetrics.aggregation import CatMetric
from torch_geometric.nn import GATv2Conv

from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir


class GAT_role_classif(LTN.LightningModule):
    def __init__(self, dim_in, dim_h1, dim_h2, heads, nb_couches, rang_sim, dropout_p, nb_classes, lr, freqs=None):
        # dim_in : Les nœuds du graphe adjoint sont décrits par une
        #          paire de vecteurs descripteurs. dim_in est leur dimension.
        # dim_h1 : dimension de la sortie de la première couche symétrique
        # rang_sim :rang pour approximation des matrices symétriques
        #
        # nb_couches : nombre de couches GATv2
        # heads : nombre de tête (param commun à toutes les couches GATv2)
        # La dimension de sortie de chaque couche est nb_têtes * dim_h2
        # On termine par une couche de classification linéaire.
        super(GAT_role_classif, self).__init__()

        if not freqs is None:
            if type(freqs) is list:
                assert len(freqs) == nb_classes
                freqs = torch.Tensor(freqs)
            assert freqs.shape == (nb_classes,)
            
        self.weight = nn.Parameter(torch.empty(dim_h1, rang_sim, dim_in))
        nn.init.xavier_normal_(self.weight)
        self.diag = nn.Parameter(torch.empty(dim_h1, rang_sim))
        nn.init.xavier_normal_(self.diag)
        self.bias_vecto = nn.Parameter(torch.empty(dim_h1, dim_in))
        nn.init.xavier_normal_(self.bias_vecto)
        self.bias0 = nn.Parameter(torch.empty(dim_h1, ))
        nn.init.normal_(self.bias0)

        self.lr = lr
        self.dim_h1 = dim_h1
        self.dim_h2 = dim_h2
        self.freqs = freqs
        self.nb_classes = nb_classes

        self.gat_layers = nn.ModuleList()
        if dropout_p is None:
            self.dropout_p = 0
            #dropout_p = 0
        else:
            self.dropout_p = dropout_p

        assert nb_couches > 0
        
        #    if nb_couches == 1:
        #        self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = nb_classes, heads=1))
        #    else:
        #        self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = dim_h2, heads=heads))
        #        for _ in range(1, nb_couches-1):
        #            self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = dim_h2, heads=heads))
        #        self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = nb_classes, heads=1))
        
        self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = dim_h2, heads=heads))
        for _ in range(1, nb_couches):
            self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = dim_h2, heads=heads))

        # Ajout d’une classification linéaire finale
        self.lin = nn.Linear(heads*dim_h2, self.nb_classes, bias=True)

        self.pondus = freqs.max() / freqs
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss(weight = self.pondus, reduction="mean")

    def forward(self, X, edge_idx):
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
        
        Y0 = Y[...,0] * (self.diag.unsqueeze(0))
        # Extraction de la colonne zéro, et multiplication point à point par diag,
        # (Ce qui revient à multiplier matriciellement par une matrice diagonale)
        # Y0 est un tenseur (b, dim_h1, rang)
        Y1 = Y[...,1]
        # Y1 est un tenseur (b, dim_h1, rang)
        H = (Y0*Y1).sum(axis=-1)
        # H est un tenseur (b, dim_h1)
        # ajout du biais :
        H = H + (self.bias0.unsqueeze(0))
        
        if self.dropout_p:
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
    
    def predict_step(self, batch, batch_idx):
        logits = self.forward(batch.x, batch.edge_index)
        logits = logits[batch.msk1]
        return logits.argmax(axis=1).to(device="cpu")
    
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch.x, batch.edge_index)
        logits = logits[batch.msk1]
        y1 = batch.y1[batch.msk1].to(dtype=torch.int64)
        (bs,) = y1.shape
        # À tester : Cette sélection d’indices propage-t-elle correctement
        # le gradient ?
        # Réponse : On dirait bien que oui.
        perte = self.loss(logits, y1)
        self.log("train_loss", perte, batch_size=bs)
        return perte
    
    def validation_step(self, batch, batch_idx):
        #Boucle de validation
        logits = self.forward(batch.x, batch.edge_index)
        logits = logits[batch.msk1]
        y1 = batch.y1[batch.msk1].to(dtype=torch.int64)
        (bs,) = y1.shape
        perte = self.loss(logits, y1)
        # Lu dans la documentation :
        #Si on l'appelle depuis la fonction validation_step, la fonction log
        #accumule les valeurs pour toute l'époché
        self.log("val_loss", perte, batch_size=bs)
        

    def on_test_start(self):
        self.accuPred = CatMetric()
        self.accuTrue = CatMetric()
    
    def test_step(self, batch, batch_idx):
        logits = self.forward(batch.x)
        logits = logits[batch.msk1]
        roles_pred = logits.argmax(axis=1).to(device="cpu")
        self.accuPred.update(roles_pred)
        y1 = batch.y1[batch.msk1]
        self.accuTrue.update(y1)

    def on_test_end(self):
        roles = self.accuTrue.compute().cpu().numpy()
        roles_pred = self.accuPred.compute().cpu().numpy()
        

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

