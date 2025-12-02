import itertools
import time
from outils.recompile import reload_loop, reload_cond, reloading, break_loop, continue_loop
import inspect
import tqdm


def essai():
    optimiseur = {"parametres": "model", "λ": 0.1}
    scheduler  = {"optim": "optimiseur", "type": "exp"}
    
    print("optimiseur initial :", optimiseur)
    print("scheduler  initial :", scheduler)
    
    epoques = itertools.count(0)
    
    frame_info = inspect.stack()[0]
    print(frame_info[1])
    for _ in reload_loop(forever=True, label="loop1"):
        # faire du rien
        #scheduler  = {"optim": "optimiseur", "type": "dents_de_scie"}
        #break
        while True:
            epoch = next(epoques)
            print("epoch: %d"%epoch)
            print("optimiseur :", optimiseur)
            print("scheduler  :", scheduler)
            for _ in tqdm.tqdm(range(1000)):
                time.sleep(0.01)
            print("-----")
            if reload_cond(label = "cond1"):
                #if epoch == 2:
                #    continue_loop("loop1")
                print("Fin de l’époque")
    print("TERMINÉ.")


def essai2():
    optimiseur = {"parametres": "model", "λ": 0.1}
    scheduler  = {"optim": "optimiseur", "type": "exp"}
    
    print("optimiseur initial :", optimiseur)
    print("scheduler  initial :", scheduler)
    
    epoques = itertools.count(0)
    
    frame_info = inspect.stack()[0]
    print(frame_info[1])
    for _ in range(1):
        # faire du rien
        #scheduler  = {"optim": "optimiseur", "type": "dents de scie"}
        #break
        while True:
            epoch = next(epoques)
            print("epoch: %d"%epoch)
            print("optimiseur :", optimiseur)
            print("scheduler  :", scheduler)
            for _ in tqdm.tqdm(range(1000)):
                time.sleep(0.01)
            print("-----")
            if True:
                if epoch == 4:
                    break
    print("TERMINÉ.")


def essai3():
    for epoch in reload_loop(range(3), label="loop2"):
        print("epoch: %d"%epoch)
        print("  * étape 1")
        for _ in tqdm.tqdm(range(500)):
            time.sleep(0.01)
        continue_loop("loop2")
        print("  * étape 2")
        for _ in tqdm.tqdm(range(500)):
            time.sleep(0.01)
    print("TERMINÉ")



    
    
if __name__ == "__main__":
    essai()