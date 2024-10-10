import torch
from torch import nn
import torch.nn.functional as NNF

class Classif_Logist(nn.Module):
    # modèle linéaire à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR. 
    def __init__(self, dim, nb_classes, freqs=None):
        super(Classif_Logist, self).__init__()
        if freqs:
            assert freqs.shape == (nb_classes,)
        self.lin = nn.Linear(dim, nb_classes, bias=True)
        self.freqs = freqs

    def forward(self, X):
        # Ni softmax ni log softmax.
        # utiliser la perte "Cross entropy à partir des logits"
        # (nn.CrossEntropyLoss ou NNF.cross_entropy)
        return self.lin(X)
    

class Classif_Bil_Sym(nn.Module):
    # modèle bilinéaire symétrique à utiliser avec le dataset d’arêtes
    # étiquettes = roles VerbAtlas ou étiquettes = roles AMR.

    def __init__(self, dim, nb_classes, rang=2, freqs = None):
        super(Classif_Bil_Sym, self).__init__()
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
    with torch.no_grad():
        Yref = modele.forward(X)
    for _ in range(20):
        index = torch.randint(0,2, (batch,), dtype=torch.long)
        Xs = changer_sens_X(X, index)
        with torch.no_grad():
            Y = modele.forward(Xs)
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
if __name__ == "__main__":
    test_sym_antisym()


    