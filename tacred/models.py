import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from huggingface_hub import PyTorchModelHubMixin
import datetime
from accelerate import Accelerator
import evaluate, itertools
from torch.optim import Adam, AdamW
from tqdm.auto import tqdm
import math


class Fscore:
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.TinterP = [0] * nb_classes
        self.T       = [0] * nb_classes
        self.P       = [0] * nb_classes

    def add_sample(self, trueClass, predClass):
        if trueClass==predClass:
            self.TinterP[trueClass] += 1
        self.T[trueClass] += 1
        self.P[predClass] += 1

    def add_batch(self, references, predictions):
        for t_class, p_class in zip(references, predictions):
            self.add_sample(t_class, p_class)

    def compute(self, β=1):
        S = sum(self.T)
        # micro:
        intersect = sum(self.TinterP)
        prec = intersect / sum(self.P)
        recl = intersect / sum(self.T)
        iprec = float("inf") if prec == 0. else 1/prec
        irecl = float("inf") if recl == 0. else 1/recl
        iF    = (iprec + irecl*(β**2))/(1+β**2)
        F     = 1/iF
        micro = F

        # macro:
        F = [0.] * self.nb_classes
        for i in range(self.nb_classes):
            prec = 0. if self.P[i] == 0 else self.TinterP[i] / self.P[i]
            recl = 0. if self.T[i] == 0 else self.TinterP[i] / self.T[i]
            iprec = float("inf") if prec == 0. else 1/prec
            irecl = float("inf") if recl == 0. else 1/recl
            iF    = (iprec + irecl*(β**2))/(1+β**2)
            F[i] = 2. if iF == 0. else 1/iF
            macro = sum(F) / self.nb_classes
        # weighted:
        weighted = sum(yl*f for yl, f in zip(self.T, F)) / S
        
        #No average
        self.F_no_average = F

        return {"F_score(micro)": micro, "F_score(macro)": macro, "F_score(weighted)": weighted}
        

        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class tacred_GAT_MODEL(nn.Module,PyTorchModelHubMixin):
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
        super().__init__()

        #self.hparams = AttrDict(**{"dim_in": dim_in,
        #                "dim_h1" : dim_h1,
        #                "dim_h2" : dim_h2,
        #                "heads" : heads,
        #                "nb_couches": nb_couches,
        #                "rang_sim": rang_sim,
        #                "dropout_p": dropout_p,
        #                "nb_classes": nb_classes})

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

    def forward(self, batch):
        X, edge_idx = batch.x, batch.edge_index
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
        logits = Y[batch.msk]
        output = (logits, batch.msk_ptr)
        return output
    
    def count_params(self):
        return sum(math.prod(t.shape) for t in self.parameters())
    
    @torch.no_grad()
    def predict_batch(self, batch):
        # Sauvegarde du mode (training ou eval)
        trmode = self.training
        # On force le mode eval
        self.eval()
        logits, ptr = self.forward(batch)
        Ytrue = batch.y1[batch.msk_batch].to(dtype=torch.int64)
        loss = F.cross_entropy(logits, Ytrue)
        sftmax = logits.softmax(axis=-1)
        index = batch.msk_batch
        index = index.unsqueeze(-1)
        index = index.repeat(1,self.nb_classes)
        R = torch.zeros((batch.y1.shape[0],self.nb_classes), dtype=sftmax.dtype, device=index.device)
        R.scatter_reduce_(0, index, sftmax, reduce="sum")
        R = R.argmax(axis=-1)
        # On rétablit le mode (training ou eval) tel qu’il était auparavant.
        self.train(trmode)
        return R, loss
    
    @torch.no_grad()
    def evaluer(self, nom_metrique, loader):
        # Sauvegarde du mode (training ou eval)
        trmode = self.training
        # On force le mode eval
        self.eval()
        if nom_metrique == "custom_Fscore":
            metriq = Fscore(nb_classes = self.nb_classes)
        else:
            metriq = evaluate.load(nom_metrique)
        iter0 = (b for b in tqdm(loader))
        iterRef, iter1 = itertools.tee(iter0, 2)
        iterRef = (b.y1 for b in iterRef)
        iterPred, iterLoss = itertools.tee((self.predict_batch(b) for b in iter1) , 2)
        iterPred = (p for p, _ in iterPred)
        iterLoss = (L.item() for _, L in iterLoss)
        eval_loss=MOYENNE()
        for b_ref, b_pred, b_loss in zip(iterRef, iterPred, iterLoss):
            eval_loss.cumuler(b_loss)
            metriq.add_batch(predictions=b_pred, references=b_ref)
        resu = metriq.compute()
        # pour la métrique "hyperml/balanced_accuracy",
        # le résultat de metriq.compute() est un dictionnaire qui contient l’unique clé
        # "balanced_accuracy", dont la valeur est un simple scalaire de type float
        resu["loss(eval)"] = eval_loss.m
        # On rétablit le mode (training ou eval) tel qu’il était auparavant.
        self.train(trmode)
        return resu

    


    


class MOYENNE:
    def __init__(self):
        self.N = 0
        self.m = 0.

    def cumuler(self, v):
        self.N += 1
        self.m += (v-self.m)/self.N
        return self.m

    def RaZ(self):
        self.N = 0
        return self.m

class tacredTRAINER:
    def __init__(self,
        model,
        lr=5.e-4,
        epochs=20,
        weight_decay=0.,
        metriq="hyperml/balanced_accuracy",
        a_surveiller = "balanced_accuracy",
        patience = 3,
    ):
        self.model = model
        self.hparams = AttrDict()
        self.hparams.nb_classes = model.nb_classes
        self.hparams.lr = lr
        self.hparams.epochs = epochs
        self.hparams.weight_decay = weight_decay
        self.hparams.metriq = metriq
        self.hparams.a_surveiller = a_surveiller
        self.hparams.patience = patience

        self.best_ckpt=None
        self.last_ckpt=None

        self.tensorboard_dir="./results_Tacred"
        self.checkpoints_dir = "./tacred_checkpoints"
        self.date_debut = datetime.datetime.now()
        self.rep_expe = "EXPE_" + self.date_debut.strftime("%d_%b_%Y_%HH%M_%S%f")

    def get_config(self):
        cfg = {}
        for t in self.hparams:
            cfg[t] = self.hparams[t]
        for k, v in self.model._hub_mixin_config.items():
            cfg["model_"+k] = v
        return cfg
    
    def preparer_accel(self, *args):
        pass

    def save_model(self, num_epoch, which="BEST"):
        assert which in ["BEST", "LAST"]
        name = self.rep_expe + "_epoch_%03d_%s"%(num_epoch, which)
        wPath = os.path.join(self.checkpoints_dir, name)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        print("Création du répertoire %s"%wPath)
        unwrapped_model.save_pretrained(
            wPath,
            is_main_process = self.accelerator.is_main_process,
            save_function = self.accelerator.save,
        )
        deldir = []
        if which == "BEST":
            # Si on enregistre un modèle meilleur, on efface le précédent
            # meilleur modèle.
            if type(self.best_ckpt) == str and os.path.exists(self.best_ckpt):
                deldir.append(self.best_ckpt)
            self.best_ckpt = wPath
        # Dans tous les cas, on efface le dernier modèle.
        if type(self.last_ckpt) == str and os.path.exists(self.last_ckpt):
            deldir.append(self.last_ckpt)
        if which == "LAST":
            self.last_ckpt = wPath

        if len(deldir) > 0:
            for D in deldir:
                print ("Effacement du répertoire %s"%D)
                for f in os.listdir(D):
                    fp = os.path.join(D, f)
                    os.remove(fp)
                os.rmdir(D)

    def elapsed_time(self):
        return datetime.datetime.now() - self.date_debut

    def train(self, loadTRAIN, loadEVAL):
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=self.tensorboard_dir)  #(fp16=True)
        config = self.get_config()

        self.accelerator.init_trackers(self.rep_expe, config=config)

        optimizer = AdamW(self.model.parameters(),
                          lr = self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)

        #Transformation du modèle et des trucs
        model, optimizer, loadTRAIN, loadEVAL = self.accelerator.prepare(
            self.model, optimizer, loadTRAIN, loadEVAL
        )
        self.loadTRAIN = loadTRAIN
        self.loadEVAL  = loadEVAL

        num_train_epochs = self.hparams.epochs

        model.train()
        global_step=0
        train_loss=MOYENNE()
        best_metriq = 0.
        early_stop_counter = 0
        for epoch in range(num_train_epochs):
            model.train()
            for step, batch in enumerate(tqdm(loadTRAIN)):
                with self.accelerator.accumulate(self.model):
                    optimizer.zero_grad()
                    logits, logits_ptr = model(batch)
                    
                    Ytrue = batch.y1[batch.msk_batch].to(dtype=torch.int64)
                    #unique, counts = torch.unique(batch.msk_batch, return_counts=True)
                    #weights = torch.ones((self.hparams.nb_classes,))
                    #weights[unique] = 1/counts
                    #loss = F.cross_entropy(logits, Ytrue, weights)
                    loss = F.cross_entropy(logits, Ytrue)
                    self.accelerator.backward(loss)
                    optimizer.step()
                    train_loss.cumuler(loss.item())
                    if step % 100 == 0:
                        self.accelerator.log({"loss(train)": train_loss.m}, step=global_step)
                        train_loss.RaZ()
                global_step += 1
            
            aLogger = self.model.evaluer(self.hparams.metriq, loadEVAL)
            metriq = aLogger[self.hparams.a_surveiller]
            self.accelerator.log(aLogger, step=global_step)
            if metriq > best_metriq:
                best_metriq = metriq
                early_stop_counter = 0
                self.save_model(epoch, "BEST")
            else:
                self.save_model(epoch, "LAST")
                early_stop_counter += 1
                if early_stop_counter >= self.hparams.patience:
                    break
            if os.path.exists("./stop_train"):
                # Pour gérer arrêt manuel
                os.remove("./stop_train")
                break

        self.accelerator.end_training()


    

    




