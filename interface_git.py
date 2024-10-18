import subprocess

class GitException(Exception):
    def __init__(self):
        super().__init__("Il y a des fichiers modifiés dans le repo git. Veillez à les soumettre, puis relancez.")
    
def git_get_commit():
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
        raise GitException
    cmd = 'git log -1 --format=format:"%H"'
    retour = subprocess.check_output(cmd, shell=True)
    if type(retour) == bytes:
        retour = retour.decode("utf-8")
    retour = retour.strip()
    return retour

GLOBAL_HASH_GIT = git_get_commit()