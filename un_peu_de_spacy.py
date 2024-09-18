import os
import re
import csv
import spacy
from aligneur_seq import aligneur_seq
from algebre_relationnelle import RELATION
import json
#L’environnement TORCH_et_TALN contient le package spacy, par exemple





class Metadata_Parser:

    token_range_re = re.compile(r'^(\d-\d|\d(,\d)+)$')
    metadata_re = re.compile(r'(?<=[^#]) ::')

    def __init__(self):
        pass

    
    @staticmethod
    def get_token_range(string):
        if '-' in string:
            start = int(string.split('-')[0])
            end = int(string.split('-')[-1])
            return [i for i in range(start, end)]
        else:
            return [int(i) for i in string.split(',')]

    
    @staticmethod
    def readlines(lines):
        #lines = self.metadata_re.sub('\n# ::', lines)
        lines = Metadata_Parser.metadata_re.sub('\n# ::', lines)
        metadata = {}
        graph_metadata = {}
        #rows = [self.readline_(line) for line in lines.split('\n')]
        rows = [Metadata_Parser.readline_(line) for line in lines.split('\n')]
        labels = {label for label,_ in rows}
        for label in labels:
            if label in ['root','node','edge']:
                graph_metadata[label] = [val for l,val in rows if label==l]
            else:
                metadata[label] = [val for l,val in rows if label==l][0]
        if 'snt' not in metadata and 'tok' not in metadata:
            metadata['snt'] = ['']
        return metadata, graph_metadata

    
    @staticmethod
    def readline_(line):
        if not line.startswith('#'):
            label = 'snt'
            metadata = line.strip()
        elif line.startswith('# ::id'):
            label = 'id'
            metadata = line[len('# ::id '):].strip()
        elif line.startswith("# ::tok"):
            label = 'tok'
            metadata = line[len('# ::tok '):].strip().split()
        elif line.startswith('# ::snt '):
            label = 'snt'
            metadata = line[len('# ::snt '):].strip()
        elif line.startswith('# ::alignments'):
            label = 'alignments'
            metadata = line[len('# ::alignments '):].strip()
        elif line.startswith('# ::node') or line.startswith('# ::root') or line.startswith('# ::edge'):
            label = line[len('# ::'):].split()[0]
            line = line[len(f'# ::{label} '):]
            rows = [row for row in csv.reader([line], delimiter='\t', quotechar='|')]
            metadata = rows[0]
            for i, s in enumerate(metadata):
                #if self.token_range_re.match(s):
                #    metadata[i] = self.get_token_range(s)
                if Metadata_Parser.token_range_re.match(s):
                    metadata[i] = Metadata_Parser.get_token_range(s)
        elif line.startswith('# ::'):
            label = line[len('# ::'):].split()[0]
            line = line[len(f'# ::{label} '):]
            metadata = line
        else:
            label = 'snt'
            metadata = line[len('# '):].strip()
        return label, metadata

def yield_prefix(nf):
    etat = 0
    lignes = []
    with open(nf, "r", encoding="utf-8") as FICHIER:
        for ligne in FICHIER:
            ligne = ligne.strip()
            if ligne.startswith("#"):
                lignes.append(ligne)
                etat = 1
            elif etat == 1:
                yield "\n".join(lignes)
                lignes = []
                etat = 2
        if etat == 1:
            yield "\n".join(lignes)
            lignes = []


def lister_phrases(liste_fichiers):
    for sntfile in liste_fichiers:
        print(sntfile)
        for pfx in yield_prefix(sntfile):
            metadata, _ = Metadata_Parser.readlines(pfx)
            if "id" in metadata and ("snt" in metadata or "tok" in metadata):
                idSNT = metadata["id"]
                dico = dict()
                if "snt" in metadata:
                    dico["snt"] = metadata["snt"]
                if "tok" in metadata:
                    dico["tok"] = metadata["tok"]
                yield idSNT, dico


def main(fichier_out="./POS_phrases.json"):
    snt_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/unsplit"
    amr_rep = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    fichiers_snt = [os.path.abspath(os.path.join(snt_rep, f)) for f in os.listdir(snt_rep)]
    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]
    doublons = ['DF-201-185522-35_2114.33', 'bc.cctv_0000.167', 'bc.cctv_0000.191', 'bolt12_6453_3271.7']
    nlp = spacy.load("en_core_web_sm")

    dico_snt = dict()
    for idSNT, dico in lister_phrases(fichiers_snt):
        if not idSNT in doublons:
            dico_snt[idSNT] = dico["snt"]
    with open(fichier_out, "w", encoding="utf-8") as FF:
        print("{", file=FF)
        for idSNT, dico in lister_phrases(fichiers_amr):
            if not idSNT in doublons:
                snt = dico_snt[idSNT]
                toks = dico["tok"]
                rspcy = nlp(snt)
                toks_spcy = [T.text for T in rspcy]
                lemmata = [T.lemma_ for T in rspcy]
                #pos = [(T.pos_, T.tag_) for T in rspcy]
                upos = [T.pos_ for T in rspcy]
                tagpos = [T.tag_ for T in rspcy]
                stops = [T.is_stop for T in rspcy]
                if len(toks) != len(toks_spcy) or any(t1 != t2 for t1, t2 in zip(toks, toks_spcy)):
                    rel_tg, rel_mg = aligneur_seq(toks, toks_spcy)
                    rel_amr_g = rel_mg.ren("amr", "groupe")
                    rel_spc_g = rel_tg.ren("spacy", "groupe")
                    d_spc_g = dict()
                    for s,g in rel_spc_g:
                        if not g in d_spc_g:
                            d_spc_g[g] = (s,)
                        else:
                            d_spc_g[g] = d_spc_g[g] + (s,)
                    d_amr_g = dict()
                    for a,g in rel_amr_g:
                        if not g in d_amr_g:
                            d_amr_g[g] = (a,)
                        else:
                            d_amr_g[g] = d_amr_g[g] + (a,)
                    l_spc_g = [(s[0],g) for g, s in d_spc_g.items()if len(s) == 1]
                    l_amr_g = [(a[0],g) for g, a in d_amr_g.items()if len(a) == 1]
                    rel_spc_g = RELATION("spacy", "groupe")
                    rel_spc_g.add(*l_spc_g)
                    rel_amr_g = RELATION("amr", "groupe")
                    rel_amr_g.add(*l_amr_g)
                    rel = (rel_amr_g * rel_spc_g).p("amr", "spacy")
                else:
                    rel = RELATION("amr", "spacy")
                    rel.add(*[(i,i) for i in range(len(toks))])
                rel2 = RELATION("spacy", "lemma", "upos", "tagpos", "stop")
                rel2.add(*[(i, lm, up, tg, st) for i, (lm, up, tg, st) in enumerate(zip(lemmata, upos, tagpos, stops))])
                rel3 = (rel * rel2).p("amr", "lemma", "upos", "tagpos", "stop")
                N = len(toks)
                l_lemma = [None] * N
                l_upos = [None] * N
                l_tagpos = [None] * N
                l_stop = [None] * N
                for T in rel3:
                    iii = T.amr
                    l_lemma[iii] = T.lemma
                    l_upos[iii] = T.upos
                    l_tagpos[iii] = T.tagpos
                    l_stop[iii] = T.stop

                jsn = {"amr_w":toks,
                       "lemma":l_lemma,
                       "upos":l_upos,
                       "tagpos":l_tagpos,
                       "stop":l_stop}
                
                print("   \"%s\": %s"%(idSNT, json.dumps(jsn)), file=FF)

        print("}", file=FF)


                
                



if __name__ == "__main__":
    main()