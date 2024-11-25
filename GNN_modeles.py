import torch
from torch import optim, nn
import torch.nn.functional as NNF
import lightning as LTN
import random
from torchmetrics.aggregation import CatMetric
from torch_geometric.nn import GATv2Conv

from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir


class GAT_role_classif(LTN.LightningModule):
    def __init__(self, dim_in, dim_h1, dim_h2, heads, nb_couches, rang_sim, dropout_p, nb_classes, cible, lr, freqs=None):
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
        self.bias = nn.Parameter(torch.empty(dim_h1, ))
        nn.init.normal_(self.bias)
        self.cible = cible
        self.lr = lr
        self.dim_h1 = dim_h1
        self.dim_h2 = dim_h2
        self.freqs = freqs
        self.nb_classes = nb_classes

        self.gat_layers = nn.ModuleList()
        if self.dropout_p is None:
            self.dropout_p = 0
            #dropout_p = 0
        else:
            self.dropout_p = dropout_p

        assert nb_couches > 0
        if nb_couches == 1:
            self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = nb_classes, heads=1))
        else:
            self.gat_layers.append(GATv2Conv(in_channels = dim_h1, out_channels = dim_h2, heads=heads))
            for _ in range(1, nb_couches-1):
                self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = dim_h2, heads=heads))
            self.gat_layers.append(GATv2Conv(in_channels = heads*dim_h2, out_channels = nb_classes, heads=1))

        self.pondus = freqs.max() / freqs
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss(weight = self.pondus, reduction="mean")

        def forward(self, X, edge_idx):
            # X est un tenseur de format (b, dim_in, 2)
            # On commence par le transformer en un tenseur (b, 1, dim_in, 2)
            # Qui s’ajustera (par distributivité) à un tenseur (b, dim_h1, dim_in, 2)

            X = X.unsqueeze(-3)
            M = self.weight.unsqueeze(0)
            B = self.bias.unsqueeze(0)
            # X est désormais un tenseur de format (b, 1, dim_in, 2)
            # et M est un tenseur de format (1, dim_h1, rang, dim_in)
            # Si on l’ajuste (par distributivité) à un tenseur (b, dim_h1, rang, dim_in)
            # alors M et X sont envisageables comme des tenseurs (b, dim_h1)
            # de matrices (rang, dim_in) pour M, ou (dim_in, 2) pour X.
            # Si on multiple (matriciellement) point à point les éléments de ces tenseurs,
            # on obtient un tenseur (b, dim_h1) de matrices (rang, 2),
            # C’est-à-dire un tenseur (b, dim_h1, rang, 2).

            Y = torch.matmul(M, X)
            # Y est un tenseur de format (b, dim_h1, rang, 2)
            
            Y0 = Y[...,0] * (self.diag.unsqueeze(0))
            # Y0 est un tenseur (b, dim_h1, rang)
            Y1 = Y[...,1]
            # Y1 est un tenseur (b, dim_h1, rang)
            H = (Y0*Y1).sum(axis=-1)
            # H est un tenseur (b, dim_h1)
            H = H + B #(ajout du biais)
            

            if self.dropout_p:
                for gat in self.gat_layers:
                    H = nn.functional.dropout(H, self.dropout_p, training=self.training)
                    H = gat(H,edge_idx)
            else:
                for gat in self.gat_layers:
                    H = gat(H, edge_idx)

            # Ni softmax ni log softmax.
            # Utiliser la perte "Entropie croisée à partir des logits"
            # (nn.CrossEntropyLoss ou NNF.cross_entropy)
            return H


