import fire
from interface_git import git_get_commit
import sys
from report_generator import HTML_REPORT
import inspect


def autoinspect(fonction):
    sig = inspect.signature(fonction)
    paires = [(nom, par) for nom, par in sig.parameters.items() ]
    assert all(par.kind in (inspect._ParameterKind.POSITIONAL_OR_KEYWORD, inspect._ParameterKind.VAR_KEYWORD) for _, par in paires), "Wrong function signature %s"%(fonction.__name__)
    paires = [(nom, par) for nom, par in paires if not par.kind == inspect._ParameterKind.VAR_KEYWORD]
    assert all(nom == par.name for (nom, par) in paires)
    dico_defaut = {nom: par.default for nom, par in paires if not par.default == inspect.Parameter.empty}
    noms_argus = [nom for nom, par in paires]
    #sans_defaut = [nom for nom, par in paires if not hasattr(par, "default")]
    nom_fonction = fonction.__name__

    def interne(*args, **kwargs):
        debug = False
        if "DEBUG" in kwargs:
            if kwargs["DEBUG"] == True:
                debug = True
            del kwargs["DEBUG"]
        if not debug:
            HTML_REPORT.MD5_GIT = git_get_commit()
            pile = inspect.stack()
            pile = [elt.function for elt in pile]
            print(pile)
            if len(pile) == 2:
                assert pile[-1] == "<module>", "La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire"
            elif len(pile) == 5:
                assert pile[1] == '_CallAndUpdateTrace', "La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire"
                assert pile[2] == '_Fire', "La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire"
                assert pile[3] == 'Fire', "La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire"
                assert pile[4] == '<module>', "La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire"
            else:
                raise Exception("La fonction %s doit être appelée depuis le premier niveau d’exécution ou par le module Fire")
            
        assert all(type(a) in [str, int, float, bool] for a in args), "Les arguments de la fonction %s doivent être de type str, int, float ou bool"%nom_fonction
        assert all(type(v) in [str, int, float, bool] for v in kwargs.values()),"Les arguments de la fonction %s doivent être de type str, int, float ou bool"%nom_fonction
        assert len(args) < len(noms_argus), "Trop d’arguments positionnels donnés en paramètre à la fonction %s"%nom_fonction
        

        dico_argus = {k: v for k, v in dico_defaut.items()}
        kwargus = {}
        interdits = []
        for i, v in enumerate(args):
            nom = noms_argus[i]
            dico_argus[nom] = v
            interdits.append(nom)

        for k, v in kwargs.items():
            assert not k in interdits, "L’argument %s de la fonction %s a été défini plus d’une fois"%(k, nom_fonction)
            if k in noms_argus:
                dico_argus[k] = v
            else:
                kwargus[k] = v
        
        assert len(dico_argus) == len(noms_argus), "Trop ou pas assez d’arguments donnés à la fonction %s"%(nom_fonction)
        assert all(A in dico_argus for A in noms_argus), "Trop ou pas assez d’arguments donnés à la fonction %s"%(nom_fonction)
        assert all(A in noms_argus for A in dico_argus), "Trop ou pas assez d’arguments donnés à la fonction %s"%(nom_fonction)

        listeargs = [(A, dico_argus[A]) for A in noms_argus if not A in dico_defaut or dico_argus[A] != dico_defaut[A]]
        listeargs.extend([(k, v) for k, v in kwargus.items()])
        exe_python = nom_fonction + "(" + ", ".join("%s=%s"%(A, repr(v)) for A, v in listeargs) + ")"
        commande = [nom_fonction]
        for k, v in listeargs:
            if type(v) is str:
                assert not '"' in v, "Les chaines contenant le caractère guillemet sont interdites en guise d’arguments à la fonction %s"%nom_fonction
                if " " in v:
                    v = '"' + v + '"'
            else:
                v = repr(v)
            commande.append("--%s"%k)
            commande.append(v)
        commande = " ".join(commande)
        #print(exe_python)
        #print(commande)
        if not debug:
            HTML_REPORT.exe_python = exe_python
            HTML_REPORT.commande = commande
        return fonction(*args, **kwargs)
    
    return interne


if __name__ == "__main__":
    def pipo(a, b, c=0, **kwargs):
        print(a+b+c)
        if "coucou" in kwargs:
            print("coucou y est")
        return a+b+c
    
    pipopipo = autoinspect(pipo)
    pipopipo(2,3, DEBUG=True, coucou=5)
            