try:
    from transformers import __version__ as transformers_version
except ModuleNotFoundError:
    transformers_version = None
assert transformers_version in ["4.47.1", "4.48.2", None]

#AMR_REP_TRAINING = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/split/training"
AMR_REP_TRAINING  = "./LDC_2020_T02_data/alignments/split/training"
#AMR_REP_DEV      = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/split/dev"
AMR_REP_DEV      = "./LDC_2020_T02_data/alignments/split/dev"
#AMR_REP_TEST     = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/split/test"
AMR_REP_TEST     = "./LDC_2020_T02_data/alignments/split/test"

#SNT_REP_TRAINING = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/split/training"
SNT_REP_TRAINING = "./LDC_2020_T02_data/amrs/split/training"
#SNT_REP_DEV      = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/split/dev"
SNT_REP_DEV      = "./LDC_2020_T02_data/amrs/split/dev"
#SNT_REP_TEST     = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/split/test"
SNT_REP_TEST     = "./LDC_2020_T02_data/amrs/split/test"

#AMR_REP = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
AMR_REP = "./LDC_2020_T02_data/alignments/unsplit"
#SNT_REP = "../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/amrs/unsplit"
SNT_REP = "./LDC_2020_T02_data/amrs/unsplit"

#PREFIXE_ALIGNEMENTS = "../alignement_AMR/leamr/data-release/alignments/ldc+little_prince."
PREFIXE_ALIGNEMENTS = "./LEAMR_data/alignments/ldc+little_prince."
AMR_UMR_91_ROLESETS_XML = "AMR-UMR-91-rolesets.xml"
#PROPBANK_DIRECTORY = "C:/Users/fcharpentier/Documents/Boulot/visuAMR/propbank-frames"
PROPBANK_DIRECTORY = "../../visuAMR/propbank-frames"

PROPBANK_TO_VERBATLAS = "../VerbAtlas-1.1.0/VerbAtlas-1.1.0/pb2va.tsv"
#PROPBANK_TO_VERBATLAS = "./pb2va.tsv"
UMR_91_ROLESETS = "./UMR_91_rolesets.tsv"


if False:
    def transfo_to_filenames(transfo, QscalK):
        assert transfo in ["roberta", "GPT2", "deberta", "LLAMA32"]
        label_QK = "QK" if QscalK else "att"
        if transfo == "roberta":
            rep_ds_grph = "./dataset_%s_"%label_QK
            rep_ds_edge = "./edges_f_%s_"%label_QK
            alig_file =    "./AMR_et_graphes_phrases_explct"
        elif transfo == "GPT2":
            rep_ds_grph = "./dataset_%s_GPT_"%label_QK
            rep_ds_edge = "./edges_f_%s_GPT_"%label_QK
            alig_file =    "./AMR_et_graphes_phrases_GPT_explct"
        elif transfo == "deberta":
            rep_ds_grph = "./deberta_%s_"%label_QK
            rep_ds_edge = "./edges_deberta_%s_"%label_QK
            alig_file =    "./AMR_grph_DebertaV2_xxlarge"
        elif transfo == "LLAMA32":
            rep_ds_grph = "LLAMA32_%s_"%label_QK
            rep_ds_edge = "edges_LLAMA32_%s_"%label_QK
            alig_file =    "./AMR_grph_LLAMA32"

        return alig_file, rep_ds_grph, rep_ds_edge
    
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
    rep_ds_grph = "./ds_graph_%s_%s_"%(transfo, label_QK)
    rep_ds_edge = "./ds_edges_%s_%s_"%(transfo, label_QK)
    alig_file   = "./alig_AMR_%s"%(transfo)
    id_model = implemented[transfo]

    return alig_file, rep_ds_grph, rep_ds_edge, id_model
