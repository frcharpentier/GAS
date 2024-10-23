import subprocess
from pathlib import Path
import os

def detruire_fichier(P):
    if P.is_dir():
        for p in P.iterdir():
            detruire_fichier(p)
        print("destruction du répertoire %s"%str(P.absolute()))
        P.rmdir()
    else:
        print("destruction du fichier %s"%str(P.absolute()))
        P.unlink()

def nettoyer_logs_lightning():
    P = Path("./lightning_logs")
    if P.exists() and P.is_dir():
        liste = [p for p in P.iterdir() if p.name.startswith("version_")]
        for li in liste:
            if not any((p.name == "checkpoints") and (p.is_dir()) for p in li.iterdir()):
                detruire_fichier(li)



class GitException(Exception):
    def __init__(self):
        super().__init__("Il y a des fichiers modifiés dans le repo git. Veillez à les soumettre, puis relancez.")
    
def git_get_commit(debug = False):
    # Provoque une exception salutaire s’il y a des fichiers
    # modifiés dans le répo git. Dans le cas contraire,
    # renvoie le hash md5 du dernier instantané git.
    
    cmd = "git status --porcelain"
    retour = subprocess.check_output(cmd, shell=True)
    if type(retour) == bytes:
        retour = retour.decode("utf-8")
    lignes = retour.split("\n")
    lignes = [lig for lig in lignes if len(lig) > 0]
    if any(not lig.startswith("?? ") for lig in lignes):
        if debug:
            return "000000000000"
        else:
            raise GitException
    cmd = 'git log -1 --format=format:"%H"'
    retour = subprocess.check_output(cmd, shell=True)
    if type(retour) == bytes:
        retour = retour.decode("utf-8")
    retour = retour.strip()
    return retour

