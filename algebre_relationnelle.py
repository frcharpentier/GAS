import os

class RELATION:
    def __init__(self, sort):
        assert type(sort) in (list, tuple)
        assert all(type(s) is str for s in sort)
        assert len(set(sort)) == len(sort)
        self.sort = sort
        self.arite = len(sort)
        self.table = []

    def add_1(self, ajout):
        assert(len(ajout) == self.arite)
        self.table.append(tuple(ajout))

    def add_n(self, ajout):
        assert all(len(t)==self.arite for t in ajout)
        self.table.extend(ajout)

    def proj(self, sort):
        assert all(s in self.sort for s in sort)
        indices = [self.sort.index(s) for s in sort]
        table = [tuple(t[i] for i in indices) for t in self.table]
        resu = RELATION(sort)
        resu.table = table
        return resu
    
    def select_tv(self, s, val):
        assert s in self.sort
        idx = self.sort.index(s)
        table = [t for t in self.table if t[idx] == val]
        resu = RELATION(self.sort)
        resu.table = table
        return resu
    
    def jointure(self, rel2):
        sort2 = rel2.sort
        sort_ = [s for s in sort2 if not s in self.sort]
        sort_i = [s for s in sort2 if s in self.sort]
        idxexter = [i for i,s in enumerate(sort2) if not (s in self.sort)]
        sort = self.sort[:]
        sort.extend(sort_)
        idxinter = [i for i,s in enumerate(sort2) if s in self.sort]
        idxinter1 = [self.sort.index(s) for s in sort_i]
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
                    table.append(t + v)
            except:
                pass
        resu = RELATION(sort)
        resu.table = table
        return resu

if __name__ == "__main__":
    rel1 = RELATION(["tok", "mot"])
    rel2 = RELATION(["mot", "sommet"])
    rel1.add_n([("t1", "m1"), ("t2", "m1"), ("t3", "m2")])
    rel2.add_n([("m1", "s1"), ("m2", "s2"), ("m1", "s2"), ("m5", "s10")])

    rel = rel1.jointure(rel2).proj(("tok", "sommet"))
    print(rel.table)

    print(rel.select_tv("sommet", "s2").table)
                                         
