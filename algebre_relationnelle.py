import types
from collections import namedtuple


class RELATION:
    def __init__(self, *sort):
        assert type(sort) is tuple
        assert all(type(s) is str for s in sort)
        assert len(set(sort)) == len(sort)
        self.sort = sort
        self.arite = len(sort)
        self.table = []
        self.nuplet = namedtuple("nuplet", " ".join(self.sort))

    def __len__(self):
        return len(self.table)
    
    def __iter__(self):
        return iter(self.table)
    
    #def __next__(self):
    #    if self.idx_table < len(self.table):
    #        t = self.table[self.idx_table]
    #        self.idx_table += 1
    #        return t
    #    else:
    #        raise StopIteration

    def ren(self, *sort):
        #Pour changer le nom des sortes. (Utile avant une jointure)
        assert len(sort) == len(self.sort)
        resu = RELATION(*sort)
        resu.table = [resu.nuplet(*t) for t in self.table]
        return resu
    
    def d(self, *sort):
        return self.ren(*sort)
    
    def copy(self):
        resu = RELATION(*self.sort[:])
        resu.table = [resu.nuplet(*(t for t in tt)) for tt in self.table]
        return resu
    
    def rmdup(self):
        self.table = list(set(self.table))
        return self

    def add(self, *ajout):
        assert all(len(t)==self.arite for t in ajout)
        self.table.extend([self.nuplet(*t) for t in ajout])
        return self
    
    def __add__(self, rel2):
        assert rel2.sort == self.sort
        resu = RELATION(*self.sort)
        table = [resu.nuplet(*t) for t in self.table]
        table.extend([resu.nuplet(*t) for t in rel2.table])
        resu.table = list(set(table))
        return resu
    
    def __sub__(self, rel2):
        assert rel2.sort == self.sort
        ens = set(self.table)
        ens = ens - set(rel2.table)
        resu = RELATION(*self.sort)
        resu.table = [resu.nuplet(*t) for t in ens]
        return resu
    
    def __mul__(self, rel2):
        return self.join(rel2)

    def proj(self, *sort):
        assert all(s in self.sort for s in sort)
        resu = RELATION(*sort)
        indices = [self.sort.index(s) for s in sort]
        table = set(resu.nuplet(*(t[i] for i in indices)) for t in self.table)
        resu.table = list(table)
        return resu
    
    def p(self, *sort):
        return self.proj(*sort)
    
    def select(self, f):
        resu = RELATION(*self.sort)
        table = [resu.nuplet(*t) for t in self.table if f(t)]
        resu.table = table
        return resu
    
    def s(self, f):
        return self.select(f)
    
    #def index(self, col):
    #    sort = (col,) + self.sort
    #    resu = RELATION(*sort)
    #    resu.add(*[(i,) + t in enumerate(set(T.t for T in self.table))])
    #    return resu

    
    def join(self, rel2):
        sort2 = rel2.sort
        sort_ = tuple(s for s in sort2 if not s in self.sort) # colonnes de rel2\self
        sort_i = tuple(s for s in sort2 if s in self.sort)    # colonnes de rel2 inter self
        sort = self.sort + sort_                              # d’abord les colonnes de self, puis celles de rel2\self.
        resu = RELATION(*sort)
        if len(sort_i) == 0:
            # S’il n’y a aucune colonnes dans l’intersection, il s’agit d’un produit cartésien
            table = [resu.nuplet(*(t+t2)) for t in self.table for t2 in rel2.table]
            resu.table = table
            return resu
        # Sinon (si on a une intersection non vide entre les colonnes)
        idxexter = [i for i,s in enumerate(sort2) if not (s in self.sort)] # indice dans rel2 des colonnes de rel2\self
        idxinter = [i for i,s in enumerate(sort2) if s in self.sort]       # indice dans rel2 des colonnes de rel2 inter self.
        idxinter1 = [self.sort.index(s) for s in sort_i]                   # indice dans self des colonnes de rel2 inter self.
        dic2 = {}
        for t in rel2.table:
            clef = tuple(t[i] for i in idxinter)
            if clef in dic2:
                dic2[clef].append(tuple(t[i] for i in idxexter))
            else:
                dic2[clef] = [tuple(t[i] for i in idxexter)]
        table = []
        for t in self.table:
            clef = tuple(t[i] for i in idxinter1)
            try:
                val = dic2[clef]
                for v in val:
                    table.append(resu.nuplet(*(t + v)))
            except:
                pass
        resu.table = table
        return resu
    


    
def test2():
    rel_mg = RELATION("mot", "groupe")
    rel_sg = RELATION("sommet", "groupe")
    arcs = RELATION("sommet_s", "sommet_c", "rel")
    arcs.add(("s1", "s3", "A0"), ("s1", "s5", "A1"), ("s1", "s2", "LOC"), ("s3", "s5", "pipo"))

    rel_mg.add(("m1", "g1"), ("m2", "g1"), ("m3", "g3"), ("m5", "g5"))
    rel_sg.add(("s1", "g1"), ("s2", "g1"), ("s3", "g3"), ("s5", "g5"))

    rel_gg = rel_sg.ren("sommet_s", "groupe_s").join(arcs).join(rel_sg.ren("sommet_c", "groupe_c")).proj("groupe_s", "groupe_c", "rel").rmdup().select(lambda x: x.groupe_s != x.groupe_c)
    print(rel_gg.table)
    rel = rel_mg.ren("mot_s", "groupe_s").join(rel_gg).proj("mot_s", "groupe_c", "rel").join(rel_mg.ren("mot_c", "groupe_c")).proj("mot_s", "mot_c", "rel").rmdup()
    print(rel.table)
    #rel = rel_ms.ren("mot_s", "sommet_s").join(arcs).proj("mot_s", "sommet_c", "rel").join(rel_ms.ren("mot_c", "sommet_c")).proj("mot_s", "mot_c", "rel").rmdup()
    #print(rel.sort)
    #print(rel.table)

    
    

    #sommets = (arcs.proj("sommet_s").ren("sommet") + arcs.proj("sommet_c").ren("sommet"))
    #GRP = sommets.join(rel1)
    #print(sommets.table)
    #sommets = sommets.rmdup()
    #print(sommets.table)

    #GRP.add(("s1", "s2", "{groupe}"))

    #rel = arcs.join(GRP).proj("sommet_s", "sommet_c", "rel")
    #print(rel.table)

    #groupes = rel1.join(rel1.ren("mot2", "sommet")).proj("mot", "mot2").join(RELATION("rel").add(("{groupe}",)))
    #groupes.rmdup()
    #print(groupes.table)

    #tble = rel1.join(rel2).proj("mot", "mot2").rmdup().ren("source", "cible").select(lambda x: x.source != x.cible)
    #tble = tble.join(RELATION("relation").add(("{groupe}",)))
    #print(tble.table)

def test1():
    rel1 = RELATION("tok", "mot")
    rel2 = RELATION("mot", "sommet")
    rel1.add(("t1", "m1"), ("t2", "m1"), ("t3", "m2"))
    rel2.add(("m1", "s1"), ("m2", "s2"), ("m1", "s2"), ("m5", "s10"))

    rel = rel1.join(rel2).proj("tok", "sommet")
    print(rel.table)

    print(rel.select(lambda x: x.sommet == "s2").table)


def test3():
    SG_mg = RELATION("mot", "groupe")
    SG_sg = RELATION("sommet", "groupe")
    REN_mg = RELATION("mot", "type", "groupe")
    REN_ag = RELATION("cible", "type", "groupe")

    SG_mg.add(("Jean", "G1"),("Valjean", "G1"),("était", "G2"),
              ("gentil", "G3"),
              ("etait", "G5"), ("bon", "G6"))
    
    SG_sg.add(("S1", "G1"), ("S2","G1"), ("S3","G2"),
              ("S4", "G3"), ("S5", "G5"), ("S6", "G6"))
    
    REN_mg.add(("cet", "_", "G4"), ("homme", "_", "G4"))
    REN_ag.add(("S1", "_", "G4"), ("S2", "_", "G4"))

    rel1 = SG_mg + REN_mg.p("mot", "groupe")
    rel2 = (rel1.ren("mot_s", "groupe") * (rel1.ren("mot_c", "groupe"))).p("mot_s", "mot_c")
    rel2 = rel2.s(lambda x: x.mot_s != x.mot_c)
    groupes = rel2 * (RELATION("rel").add(("{groupe}",)))

    rel3 = SG_sg + (REN_ag.p("cible", "groupe").ren("sommet", "groupe"))
    rel3 = (rel1 * rel3).ren("mot", "g1", "sommet")
    rel3 = (rel3 * SG_sg).p("mot", "groupe")
    rel4 = (rel3.ren("mot_s", "groupe") * rel3.ren("mot_c", "groupe")).p("mot_s", "mot_c")
    rel4 = rel4.s(lambda x: x.mot_s != x.mot_c)
    idems = (rel4-rel2)*(RELATION("rel").add(("{idem}", )))

    print((groupes+idems).table)

if __name__ == "__main__":
    test3()                          
