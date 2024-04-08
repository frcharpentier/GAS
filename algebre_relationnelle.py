import types

class NUPLET:
    def __init__(self, rel, t):
        if isinstance(t, NUPLET):
            self.t = t.t
        else: 
            self.t = tuple(t)
        self.rel = rel

    def __getattr__(self, s):
        idx = self.rel.sort.index(s)
        return self.t[idx]
    
    def __getitem__(self, idx):
        return self.t[idx]
    
    def __hash__(self):
        return hash(self.t + self.rel.sort)
    
    def __eq__(self, n2):
        return (self.rel.sort == n2.rel.sort) and (self.t == n2.t)
    
    def __len__(self):
        return len(self.t)
    
    def __str__(self):
        return str(self.t)
    
    def __repr__(self):
        return repr(self.t)

class RELATION:
    def __init__(self, *sort):
        assert type(sort) is tuple
        assert all(type(s) is str for s in sort)
        assert len(set(sort)) == len(sort)
        self.sort = sort
        self.arite = len(sort)
        self.table = []

    def __len__(self):
        return len(self.table)
    
    def enum(self):
        for t in self.table:
            yield t.t

    def ren(self, *sort):
        #Pour changer le nom des sortes. (Utile avant une jointure)
        assert len(sort) == len(self.sort)
        resu = RELATION(*sort)
        resu.table = [NUPLET(resu, t) for t in self.table]
        return resu
    
    def copy(self):
        resu = RELATION(*self.sort[:])
        resu.table = [NUPLET(resu, (t for t in tt)) for tt in self.table]
        return resu
    
    def rmdup(self):
        self.table = list(set(self.table))
        return self

    def add(self, *ajout):
        assert all(len(t)==self.arite for t in ajout)
        self.table.extend([NUPLET(self, t) for t in ajout])
        return self
    
    def __add__(self, rel2):
        assert rel2.sort == self.sort
        resu = RELATION(*self.sort)
        table = [NUPLET(resu, t) for t in self.table]
        table.extend([NUPLET(resu, t) for t in rel2.table])
        resu.table = table
        return resu
    
    def __sub__(self, rel2):
        assert rel2.sort == self.sort
        ens = set(self.table)
        ens = ens - set(rel2.table)
        resu = RELATION(*self.sort)
        resu.table = [NUPLET(resu, t) for t in ens]
        return resu

    def proj(self, *sort):
        assert all(s in self.sort for s in sort)
        resu = RELATION(*sort)
        indices = [self.sort.index(s) for s in sort]
        table = [NUPLET(resu, (t[i] for i in indices)) for t in self.table]
        resu.table = table
        return resu
    
    def select(self, f):
        resu = RELATION(*self.sort)
        table = [NUPLET(resu, t) for t in self.table if f(t)]
        resu.table = table
        return resu
    
    def join(self, rel2):
        sort2 = rel2.sort
        sort_ = tuple(s for s in sort2 if not s in self.sort)
        sort_i = tuple(s for s in sort2 if s in self.sort)
        idxexter = [i for i,s in enumerate(sort2) if not (s in self.sort)]
        sort = self.sort + sort_
        idxinter = [i for i,s in enumerate(sort2) if s in self.sort]
        idxinter1 = [self.sort.index(s) for s in sort_i]
        resu = RELATION(*sort)
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
                    table.append(NUPLET(resu, (t.t + v)))
            except:
                pass
        resu.table = table
        return resu
    

def transfo_AMR(amr):
    sommets = RELATION("sommet", "variable", "constante")
    arcs = RELATION("source", "cible", "relation")
    arcs_redir = RELATION("source", "cible", "rel_redir")

    for k, v in amr.nodes.items():
        if k in amr.variables:
            sommets.add((amr.isi_node_mapping[k], v, None))
        else:
            sommets.add((amr.isi_node_mapping[k], None, v))

    for s, r, t in amr.edges:
        s = amr.isi_node_mapping[s]
        t = amr.isi_node_mapping[t]
        arcs.add((s, t, r))

    for s, r, t in amr.edges_redir():
        s = amr.isi_node_mapping[s]
        t = amr.isi_node_mapping[t]
        arcs.add((s, t, r))

    return {"sommets":sommets, "arcs": arcs, "arcs_redir": arcs_redir}

    
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

if __name__ == "__main__":
    test2()                          
