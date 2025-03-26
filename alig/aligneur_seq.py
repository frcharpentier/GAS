import os
from algebre_relationnelle import RELATION



def aligneur_seq(motsH, toksV, zeroV = 0, zeroH = 0):
    # Chercher le parcours de coût minimal
    # pour traverser une grille du coin supérieur gauche
    # au coin inférieur droit, d’intersection en intersection.
    # les intervalles entre deux intersections sont indexés
    # verticalement et horizontalement
    # par les tokens toksV et motsH.
    # À chaque intersection, on a le droit de se déplacer
    # vers la droite ou vers le bas, pour un coût qui vaut 1.
    # Si les tokens vertical et horizontal de l’intervalle
    # à traverser sont identiques, on peut se déplacer en
    # diagonale (vers le bas et la droite) pour un coût nul.
    # Il s’agit de l’algo de Needleman-Wunsch, que je considère
    # comme un cas particulier de l’algo A*.
    #
    # motsH contient les tokens de l’AMR
    # phrase contient la phrase à faire tokeniser

    visites = dict()
    front = {(0,0): (0, (0,0))}
    # format : position(i,j)  : (cout_chemin_parcouru, mvt)
    nV = len(toksV)
    nH = len(motsH)
    
    estim = lambda x: abs(nV-x[0] - nH + x[1])
    #estimation du cout restant à partir d’une position
    
    clef = lambda x : x[1][0] + estim(x[0]) 
    #calcul du cout du chemin parcouru + estimation du cout à venir

    while True:
        choix = min(front.items(), key=clef)
        
        (posV, posH), (cout, mvt) = choix
        if (posV, posH) == (nV, nH):
            break
        # On va faire évoluer ce cheminement, et
        # considérer tous les cheminements possibles en 
        #ajoutant à chaque fois un déplacement élémentaire.
        del front[(posV, posH)]
        visites[(posV, posH)] = mvt
        if posV < nV and posH < nH and len(toksV[posV]) <= len(motsH[posH]):
            H = motsH[posH].lower()
            mvt0 = 0
            V = toksV[posV + mvt0].lower()
            vf = True
            #while H.startswith(V) and posV+mvt0+1 < nV:
            #    mvt0 += 1
            #    H = H[len(V):]
            #    V = toksV[posV + mvt0].lower()
            while H.startswith(V) and posV+mvt0+1 <= nV:
                mvt0 += 1
                H = H[len(V):]
                if posV+mvt0 < nV:
                    V = toksV[posV + mvt0].lower()
                else:
                    V = "????????"
            if len(H) == 0:
                #possibilité de déplacement en diagonale
                posV2, posH2 = posV+mvt0, posH+1
                if not (posV2, posH2) in visites:
                    if (posV2, posH2) in front:
                        cout0, _ = front[(posV2, posH2)]
                        if cout0 > cout+0:
                            front[(posV2, posH2)] = (cout, (1,mvt0))
                    else:
                        front[(posV2, posH2)] = (cout, (1,mvt0))
        if posV < nV:
            #possibilité de déplacement vertical
            posV2, posH2 = posV+1, posH
            if not (posV2, posH2) in visites:
                if (posV2, posH2) in front:
                    cout0, mvt0 = front[(posV2, posH2)]
                    if cout0 > cout+1:
                        front[(posV2, posH2)] = (cout+1, (0,1))
                else:
                    front[(posV2, posH2)] = (cout+1, (0,1))
        if posH < nH:
            #possibilité de déplacement horizontal
            posV2, posH2 = posV, posH+1
            if not (posV2, posH2) in visites:
                if (posV2, posH2) in front:
                    cout0, mvt0 = front[(posV2, posH2)]
                    if cout0 > cout+0:
                        front[(posV2, posH2)] = (cout+1, (1,0))
                else:
                    front[(posV2, posH2)] = (cout+1, (1,0))
    
    chem = []
    while (posV, posH) != (0,0):
        chem.append(mvt)
        posH, posV = posH - mvt[0], posV-mvt[1]
        mvt = visites[(posV, posH)]
    chem = chem[::-1]

    # On connaît le cheminement optimal dans la grille.
    # déduisons-en un alignement de la chaine motsH vers la chaine toksV
    # On représentera cet alignement sous forme de deux relations.
    # Une relation de schéma("token", "groupe") et une autre de schéma
    # ("mot", "groupe"). Les tokens et les mots seront représentés par
    # leur numéro d’ordre.
    rel_tg = RELATION("token", "groupe") #Token de transformer -- groupe
    rel_mg = RELATION("mot", "groupe")   #mot de phrase -- groupe
    
    i = zeroV #Numéro de token
    j = zeroH #Numéro de mot
    cumulH = 0
    cumulV = 0
    NG = 0 #Numéro de groupe pour l’alignement
    
    for H, V in chem:
        if H==0 or V==0:
            #Accumulation
            cumulH += H
            cumulV += V
        else:
            if cumulH > 0 or cumulV > 0:
                if cumulH == cumulV:
                    #distribuons un token pour un mot
                    rel_tg.add(*[(i+k, NG+k) for k in range(cumulV)])
                    rel_mg.add(*[(j+k, NG+k) for k in range(cumulH)])
                    NG += cumulV
                elif cumulH > 0:
                    #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
                    rel_tg.add(*[(i+k, NG) for k in range(cumulV)])
                    rel_mg.add(*[(j+k, NG) for k in range(cumulH)])
                    NG += 1
                i += cumulV
                j += cumulH
                cumulH = 0
                cumulV = 0
            assert H == 1
            rel_tg.add(*[(i+k, NG) for k in range(V)])
            rel_mg.add((j, NG))
            i += V
            j += 1
            NG += 1
    if cumulH > 0 or cumulV > 0:
        if cumulH == cumulV:
            #distribuons un token pour un mot
            rel_tg.add(*[(i+k, NG+k) for k in range(cumulV)])
            rel_mg.add(*[(j+k, NG+k) for k in range(cumulH)])
            NG += cumulV
        elif cumulH > 0:
            #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
            rel_tg.add(*[(i+k, NG) for k in range(cumulV)])
            rel_mg.add(*[(j+k, NG) for k in range(cumulH)])
            NG += 1
        i += cumulV
        j += cumulH
        cumulH = 0
        cumulV = 0

    return rel_tg, rel_mg

    


if __name__ == "__main__":
    motsH = [T for T in "1234"]
    toksV = [T for T in "1A4"]
    Vg, Hg = aligneur_seq(motsH, toksV)