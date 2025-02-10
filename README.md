# Graph of Attention for Semantics

## Prerequisites

* Use python version >= 3.11
* Install pytorch with your version of cuda, following the instructions on pytorch.org
* run `pip install -r requirements.txt`, or install manually the following list :



```
fire==0.7.0
beautifulsoup4==4.12.3
jaraco.collections==5.1.0
lightning==2.4.0
lxml==5.3.0
penman==1.3.1
pickleshare==0.7.5
pyarrow==17.0.0
scikit-learn==1.5.2
seaborn==0.13.2
sentencepiece==0.2.0
tomli==2.0.1
torch-geometric==2.6.1
transformers==4.45.0
protobuf==5.28.2
```

* run the final following command : 

  `pip install git+https://github.com/frcharpentier/amr-utils.git@master`

## Instructions to build the dataset :

Build text files containing the AMRs, the tokenized sentences, and the alignments between the two. The file `fabrication_listes.py` serves this purpose : execute the function `construire_graphes`.

Type the following command : `python ./batch_calcul.py make_alig_file --transfo roberta`

* The parameter "split=True") will build three files : One for training, one for dev, and one for test.

* The parameter `--transfo` is a shorthand for the model. (used for the tokenizer). You can use `"gpt2"`,  `"roberta-base"` "`deberta`" or "`LLAMA32`".

The process assumes you have the LDC_2020_T02 dataset under `../../visuAMR/AMR_de_chez_LDC/LDC_2020_T02/data/alignments/split/training ... /dev, and .../test`.

Change the location in the file `dependencies.py`. You will also need a copy of the `pb2va.tsv` file of VerbAtlas under `../VerbAtlas-1.1.0/VerbAtlas-1.1.0/pb2va.tsv`. You can download it from https://verbatlas.org/downloads/VerbAtlas-1.1.0.zip

The function also assumes you have the leamr alignments by Blodgett et al. under the directory `../alignement_AMR/leamr/data-release/alignments/`. Get it at https://github.com/ablodge/leamr/tree/master/data-release/alignments.

Those locations can also be changed in the file `dependencies.py`.



Executing the command will yield three text files : `./alig_AMR_<transfo>_train.txt`, `./alig_AMR_<transfo>_dev.txt` and `./alig_AMR_<transfo>_test.txt.`

Those are sufficient to re-run experiments located in the directory `Experiment_results`. Everything uses the file `batch_calcul.py`. This file imports the python module `fire`, allowing to execute a python function from the command line. Each result file gives the command to launch to redo the experiment. In case of a problem, it also gives a md5 checksum of a git snapshot. Use `git checkout` to obtain exactly the same snapshot.



The first run will build torch geometric datasets. This make take a while and will create files on your filesystem of a few gigabytes. Subsequents runs will skip this step, as they will directly read from those files.



