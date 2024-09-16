import types


# La classe Maillon est un décorateur qu’on peut ajouter à une fonction pour faire des
# filtres enchainables. Le premier argument de la fonction est une source itérable
# sur laquelle la fonction fait une boucle for, et fait des traitement. La fonction
# possède l’instruction yield.

class MAILLON(object):
    def __init__(self, fonction):
        self.fonction = fonction
        self.nom = fonction.__name__
        

    def __call__(self, *args, **kwargs):
        self.argus = [None] + [t for t in args]
        self.kwargus = kwargs
        return self

    def __rshift__(self, C2):
        # Cette fonction sert à la surcharge de l’opérateur >>
        C2.argus[0]=self
        return C2

    def __iter__(self):
        self.itrt = self.fonction(*self.argus, **self.kwargus)
        if isinstance(self.itrt, types.GeneratorType):
            pass
        else:
            self.itrt = None
        return self
    
    def __next__(self):
        if self.itrt is None:
            raise StopIteration
        else:
            return next(self.itrt)
        
    def enchainer(self):
         for _ in self:
              pass
          

if __name__ == "__main__":
# Voici un exemple pour jouer au jeu du "fizz buzz" :
# Le premier maillon de la chaine n’utilise pas son entrée, 
# et fournit simplement tous les nombres de zéro à cent.

    @MAILLON
    def source(S):
        for i in range(100):
            #print("source %d"%i)
            yield i

# Le deuxième maillon de la chaine itère son entrée
# et teste si le nombre est un multiple de cinq (test fizz)
# elle fournit un couple dont le premier élt est le
# nombre et le deuxième un booléen qui est vrai si
# on est en présence d’un "fizz"

    @MAILLON
    def filtre5(S):
        for i in S:
            if i%5 == 0:
                fizz = True
            else:
                fizz = False
            yield (i, fizz)

# Le troisième maillon itère des couples nombre, bool
# de son entrée et test si l’écriture du nombre contient
# un sept (test buzz). Elle fournit un triplet dont le premier élt est le
# nombre le deuxième est le booléen "fizz", et le troisième le 
# booléen "buzz"
    @MAILLON
    def filtre7(S):
        for i, fizz in S:
            if "7" in str(i):
                buzz = True
            else:
                buzz = False
            yield (i, fizz, buzz)

# Le dernier maillon de la chaine itère les triplets
# en entrée et affiche le nombre, le mot "fizz", le mot "buzz"
# ou les deux mots "fizz buzz". La fonction ne "yield" rien.
# c’est donc forcément le dernier maillon de la chaine.

    @MAILLON
    def affiche(S, mot_fizz="FIZZ", mot_buzz="BUZZ"):
        for i, fizz, buzz in S:
            if fizz or buzz:
                if fizz and buzz:
                    print("%s %s"%(mot_fizz, mot_buzz))
                elif fizz:
                    print("%s"%mot_fizz)
                elif buzz:
                    print("%s"%mot_buzz)
            else:
                print(i)

# On construit la chaine avec l’opérateur >>. Remarquer que le premier argument
# de chaque fonction est toujours la source, et doit toujours être omis lors de l’appel
    chaine = source() >> filtre5() >> filtre7() >> affiche("crac", "boum")
    
# Et on lance toute la chaine
    chaine.enchainer()
