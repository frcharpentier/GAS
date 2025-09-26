import os
from alig.algebre_relationnelle import RELATION
import json_stream
import json
from tqdm.auto import tqdm
import itertools
from tacred.fichier_zero import rebuild_sentence
from alig.outils_alignement import ALIGNEUR


def transfo_to_filenames(transfo, QscalK):
    implemented = {"roberta": "minbert://roberta-base",
                   "robertaBase": "minbert://roberta-base",
                   "robertaLarge": "minbert://roberta-large",
                   "GPT2": "mingpt://gpt2",
                   "deberta": "huggingface://microsoft/deberta-v2-xxlarge",
                   "LLAMA32": "huggingface://meta-llama/Llama-3.2-3B",
                   "Llama3B": "huggingface://meta-llama/Llama-3.2-3B",
                   "Llama3Bi": "huggingface://meta-llama/Llama-3.2-3B-Instruct",
                   "Llama1B": "huggingface://meta-llama/Llama-3.2-1B",
                   "Llama1Bi": "huggingface://meta-llama/Llama-3.2-1B-Instruct",
                   "Llama8B": "huggingface://meta-llama/Llama-3.1-8B",
                   "Llama8Bi": "huggingface://meta-llama/Llama-3.1-8B-Instruct",
                   "mdrnBertBase" : "huggingface://answerdotai/ModernBERT-base",
                   "mdrnBertLarge" : "huggingface://answerdotai/ModernBERT-large",
                   "spring" : "huggingface://facebook/bart-large"}
    assert transfo in implemented
    label_QK = "QK" if QscalK else "att"
    rep_ds_grph = "./ds_graph_tacred_%s_%s_"%(transfo, label_QK)
    alig_file   = "./alig_tacred_%s"%(transfo)
    id_model = implemented[transfo]

    return alig_file, rep_ds_grph, id_model

def count_json_stream(file):
    N = 0
    with open(file, "r", encoding="utf-8") as F:
        data = json_stream.load(F)
        try:
            while(True):
                _ = data[N]
                N += 1
        except IndexError:
            pass
    return N

def iter_json(file):
    NN = count_json_stream(file)
    with open(file, "r", encoding="utf-8") as F:
        data = json_stream.load(F)
        try:
            for N in tqdm(range(NN)):
                resu = json_stream.to_standard_types(data[N])
                yield resu
        except IndexError:
            pass


def filtrer_iter(id, docid, rel_tg, rel_mg, groupe_source, groupe_cible, relation, toks_transfo):
    #rel_tg-grp est la relation num_token-groupe, rel_mg est la relation num_mot--groupe,
    # et toks_transfo est la liste in extenso des tokens du transformer
    
    rel_mP = RELATION("mot", "point")

    rel_mP.add(*[(m, "SOURCE") for m in range(groupe_source[0], 1+groupe_source[1])])
    rel_mP.add(*[(m, "CIBLE") for m in range(groupe_cible[0], 1+groupe_cible[1])])

    rel_tP = (rel_tg * rel_mg * rel_mP).p("token", "point")
    source = [t.token for t in rel_tP.select(lambda x: x.point == "SOURCE")]
    source.sort()
    cible = [t.token for t in rel_tP.select(lambda x: x.point == "CIBLE")]
    cible.sort()
    assert all(x in source for x in range(source[0], 1+source[-1]))
    assert all(x in cible for x in range(cible[0], 1+cible[-1]))
    assert not any(x in cible for x in source)
    assert not any(x in source for x in cible)
    sommets = source + cible
    aretes = [(i, relation, j) for i in range(len(source)) for j in range(len(source), len(sommets))]
    return {"id": id, "docid": docid, "tokens": toks_transfo, "sommets": sommets, "aretes": aretes}
    
    

#rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"

def make_aligs(rep, model_name="roberta_base", file_out="a_tej.json"):
    if file_out.endswith(".json"):
        file_out = file_out[:-5]
        filesOUT = {
                "train": file_out + "_train.json",
                "dev":   file_out + "_dev.json",
                "test":  file_out + "_test.json",
        }
    aligneur = ALIGNEUR(model_name)
    for split in ["train", "dev", "test"]:
        fileIN = os.path.join(rep, split + ".json")
        fileOUT = filesOUT[split]

        iter0 = iter_json(fileIN)
        iter1, iter2, iter_S, iter_C, iter_rel, iter_id, iter_docid = itertools.tee(iter0, 7)

        iter_id = (X["id"] for X in iter_id)
        iter_docid = (X["docid"] for X in iter_docid)
        iter_rel = (X["relation"] for X in iter_rel)
        iter_toks = (X["token"] for X in iter1)
        iter_phrases = (rebuild_sentence(X["token"]) for X in iter2)
        iter_source = ((X["subj_start"], X["subj_end"]) for X in iter_S)
        iter_cible = ((X["obj_start"], X["obj_end"]) for X in iter_C)

        iter_aligs = (aligneur.aligner_seq(T, S) for T, S in zip(iter_toks, iter_phrases))

        iter_argus = ((id,) + (docid,) + X + (S,) + (C,) + (R,) for id, docid, X, S, C, R in zip(iter_id, iter_docid, iter_aligs, iter_source, iter_cible, iter_rel))
        # Iterateur qui donne des 6-uplets avec (id, doc_id, rel_tg, rel_mg, toks_transfo, groupe_source, groupe_cible, relation)

        iterF = (
            filtrer_iter(
                id = X[0],
                docid = X[1],
                rel_tg=X[2],
                rel_mg=X[3],
                groupe_source=X[5],
                groupe_cible=X[6],
                relation=X[7],
                toks_transfo=X[4]
            ) for X in iter_argus
        )
        with open(fileOUT, "w", encoding="utf-8") as FOUT:
            print("# ::model_name %s"%model_name, file=FOUT)
            start = "["
            for dico in iterF:
                print("%s%s"%(start, json.dumps(dico)), file=FOUT, end="")
                start = ",\n"
            print("\n]", file=FOUT, end="")



def essai(rep):
    alig_file, rep_ds_grph, id_model = transfo_to_filenames("roberta_base")
    print("alig_file :",alig_file)
    print("rep_ds_grph: ", rep_ds_grph)
    print("id_model:", id_model)
    make_aligs("roberta_base", alig_file)
    print("TERMINÉ.")

    




if __name__ == "__main__":
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    file = os.path.join(rep, "test.json")
    N = count_json_stream(file)
    print("N= %d"%N)