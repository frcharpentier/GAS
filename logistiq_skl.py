
import traceback
import joblib
import torch

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import logging
from report_generator import HTML_REPORT

from make_dataset import AligDataset, EdgeDataset, EdgeDatasetMono, EdgeDatasetRdmDir


def dessin_matrice_conf(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(xticks_rotation="vertical", ax=ax)
    #pour calculer disp.figure_, qui est une figure matplotlib
    return disp.figure_

def pour_fusion(C, liste):
    if C.startswith(":") and C[1] != ">":
        CC = ":>" + C[1:].upper()
        if CC in liste:
            return CC
    return C

def faire_datasets_edges(train=True, dev=True, test=True):
    if train:
        DGRtr = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct", QscalK=True, split="train")
    if dev:
        DGRdv = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct", QscalK=True, split="dev")
    if test:
        DGRts = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct", QscalK=True, split="test")

    noms_classes = [k for k in DGRtr.filtre.alias]

    filtre = DGRtr.filtre.eliminer(":li", ":conj-as-if", ":op1", ":weekday", ":year", ":polarity", ":mode")
    filtre = filtre.eliminer(":>POLARITY")
    filtre = filtre.fusionner(lambda x: pour_fusion(x.al, noms_classes))
    filtre = filtre.eliminer(lambda x: x.al.startswith(":prep"))
    filtre = filtre.eliminer(lambda x: (x.ef < 1000) and (not x.al.startswith(":>")))
    filtre2 = filtre.eliminer(lambda x: x.al.startswith("{"))

    filtre2 = filtre.garder(":>AGENT", ":>BENEFICIARY", ":>CAUSE", ":>THEME",
                            ":>CONDITION", ":degree", ":>EXPERIENCER",
                            ":>LOCATION", ":>MANNER", ":>MOD", ":>PATIENT",
                            ":poss", ":>PURPOSE", ":>TIME", ":>TOPIC")

    if train:
        DGRtr_f2 = AligDataset("./dataset_QK_train", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="train")
    if dev:
        DGRdv_f2 = AligDataset("./dataset_QK_dev", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="dev")
    if test:
        DGRts_f2 = AligDataset("./dataset_QK_test", "./AMR_et_graphes_phrases_explct",
                            transform=filtre2, QscalK=True, split="test")
        
    datasets = ()
    if train:
        datasets += (EdgeDatasetMono(DGRtr_f2, "./edges_f_QK_train"),)
    if dev:
        datasets += (EdgeDatasetMono(DGRdv_f2, "./edges_f_QK_dev"),)
    if test:
        datasets += (EdgeDatasetMono(DGRts_f2, "./edges_f_QK_test"),)
    
    return datasets

def batch_Lgstq():
    
    DARtr, DARts = faire_datasets_edges(train=True, dev=False, test=True)

    nom_rapport = "./classif_Logistiq.html"
    
    #noms_dataset = ["./dataframe_QscalK.fth"] #, "./dataframe_QscalK_gpt2.fth", "./dataframe_QscalK_gpt2_CSSC.fth"]
    random_state = 152331
    #RS_MLP = 152331
    RS_MLP = 425
    RS_oversampling = 8236
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    dual = False
    tol=1.0e-4
    valC = 1.0
    fit_intercept = True
    cw = "balanced"
    solver = "saga" #"lbfgs"
    max_iter=150
    multi_class = "multinomial"
    

    try:
        with HTML_REPORT(nom_rapport) as R:
            for nom_dataset in ["converti de pytorch"]:
                logging.info("Chargement du dataset %s"%(nom_dataset))

                #datafr, relations = importer_dataset(nom_dataset)
                #dic_list, effectifs = calculer_effectifs(datafr["relation"])


                R.ligne()
                R.titre("Dataset : %s"%nom_dataset)
                R.texte("Effectifs avant filtrage :")
                #R.table(relations=dic_list, effectifs=effectifs)
                #R.flush()

                logging.info("Dataset : %s"%nom_dataset)
                

                X_train = DARtr.X.to(dtype=torch.float32).numpy()
                y_train = DARtr.roles.to(dtype=torch.long).numpy()
                    
                for varbls in ["sans_token_sep"]: #["sans_token_sep", "toutes"]:
                    if varbls == "toutes":
                        sl = slice(5,None) #slice du type [5:]
                    else:
                        sl = slice(5,5+2*144) #slice du type [5:5+2*144]
                    modele = R.skl(LogisticRegression)(
                        dual=dual,
                        tol=tol,
                        C=valC,
                        fit_intercept=fit_intercept,
                        class_weight = cw,
                        random_state=RS_MLP,
                        solver=solver,
                        max_iter=max_iter,
                        multi_class = multi_class,
                        verbose=3
                    )
                    
                    logging.info("Calcul de la régression (logistique)")
                    
                    
                    modele.fit(X_train, y_train)
                    
                    
                    #mlp.partial_fit(x_tr, y_tr, np.unique(y_tr))
                    logging.info("fin du calcul.")
                    logging.info("Calcul des perfs.")
                    X_tst = DARts.X.to(dtype=torch.float32).numpy()
                    y_tst = DARts.roles.to(dtype=torch.long).numpy()
                    predictions = modele.predict(X_tst)
                    accuracy = accuracy_score(y_tst, predictions)
                    bal_accuracy = balanced_accuracy_score(y_tst, predictions)
                    logging.info("Accuracy : %f, bal_accuracy : %f"%(accuracy, bal_accuracy))
                    dtype1 = predictions.dtype
                    dtype0 = y_tst.dtype
                    logging.info("dtype %s %s"%(repr(dtype0),repr(dtype1)))
                    logging.info("mlp.classes_ : %s"%repr(modele.classes_))
                    logging.info("predictions %s"%(repr(predictions[:20])))
                    
                    R.titre("Accuracy : %f, balanced accuracy : %f"%(accuracy, bal_accuracy), 2)
                    R.titre("Matrice de confusion", 2)
                    with R.new_img_with_format("svg") as IMG:
                        #confusion = plot_confusion_matrix(y_tst.to_numpy(), predictions, dic_list, IMG.fullname, numeric=False)
                        fig = dessin_matrice_conf(y_tst, predictions, DARtr.liste_roles)
                        fig.savefig(IMG.fullname, format="svg")
                    #R.titre("confusion au format python :", 3)
                    #R.texte(repr(confusion))
                    #R.ligne()
                    #R.flush()
                    logging.info("Sauvegarde du modèle")
                    with R.new_ressource() as RES:
                        joblib.dump(modele, RES.fullname)

    except Exception as e:
        chaine = traceback.format_exc()
        logging.error(chaine)
        with HTML_REPORT(nom_rapport) as R:
            R.ligne()
            R.titre("Une erreur est survenue !")
            R.texte(chaine)






if __name__=="__main__":
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        filename='batch_calcul.log',
        encoding='utf-8',
        level=logging.INFO
    )
    #essai_oversampling()
    batch_Lgstq()