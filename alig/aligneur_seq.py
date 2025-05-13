import os
#from alig.algebre_relationnelle import RELATION




def grouper_aligs(chem):
    sortie = []
    cumulH, cumulV = 0,0
    for H, V in chem:
        if H==0 or V==0:
            #Accumulation
            cumulH += H
            cumulV += V
        else:
            if cumulH > 0 or cumulV > 0:
                if cumulH == cumulV:
                    #distribuons un token pour un mot
                    sortie.extend([(1, 1) for _ in range(cumulV)])
                elif cumulH > 0:
                    #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
                    sortie.append((cumulH, cumulV))
                cumulH = 0
                cumulV = 0
            assert H == 1
            sortie.append((1, V))
    if cumulH > 0 or cumulV > 0:
        if cumulH == cumulV:
            #distribuons un token pour un mot
            sortie.extend([(1, 1) for _ in range(cumulV)])
        elif cumulH > 0:
            #alignons l’ensemble des tokens sautés sur l’ensemble des mots sautés
            sortie.append((cumulH, cumulV))
        cumulH = 0
        cumulV = 0
    return sortie
    



def aligs_to_RELs(chem, zeroV, zeroH):
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
    NG = 0 #Numéro de groupe pour l’alignement
    
    for H, V in chem:
        rel_tg.add(*[(i+k, NG) for k in range(V)])
        rel_mg.add(*[(j+k, NG) for k in range(H)])
        i += V
        j += H
        NG += 1

    return rel_tg, rel_mg


def afficher_alignement(motsH, toksV, chem):
    cumulH, cumulV = 0,0
    #truc = "↑↓"
    aligMots = []
    aligTokens = []
    for H, V in chem:
        mots, tokens = motsH[cumulH:cumulH+H], toksV[cumulV:cumulV+V]
        mots, tokens = " ".join(mots), " ".join(tokens)
        dif = len(mots) - len(tokens)
        if dif > 0:
            gauche, droite = " "*(dif//2), " "*(dif-(dif//2))
            tokens = gauche + tokens + droite
        elif dif < 0:
            dif = -dif
            gauche, droite = " "*(dif//2), " "*(dif-(dif//2))
            mots = gauche + mots + droite
        cumulH += H
        cumulV += V
        aligMots.append(mots)
        aligTokens.append(tokens)
    aligMots =   " ↓ ".join(aligMots)
    aligTokens = " ↑ ".join(aligTokens)
    print("%s\n%s"%(aligMots, aligTokens))

def afficher_aligs2(motsH, toksV, aligsH, aligsV):
    aligMots = []
    aligTokens = []
    for aligH, aligV in zip(aligsH, aligsV):
        mots = " ".join(motsH[j] for j in aligH)
        tokens = " ".join(toksV[i] for i in aligV)
        dif = len(mots) - len(tokens)
        if dif > 0:
            gauche, droite = " "*(dif//2), " "*(dif-(dif//2))
            tokens = gauche + tokens + droite
        elif dif < 0:
            dif = -dif
            gauche, droite = " "*(dif//2), " "*(dif-(dif//2))
            mots = gauche + mots + droite
        aligMots.append(mots)
        aligTokens.append(tokens)
    aligMots =   " ↓ ".join(aligMots)
    aligTokens = " ↑ ".join(aligTokens)
    print("%s\n%s"%(aligMots, aligTokens))




def aligneur_seq(motsH, toksV, zeroV = 0, zeroH = 0, return_REL=True):
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

    afficher_alignement(motsH, toksV, chem)

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
    


def better_seq_aligner(motsH, toksV, zeroV=0, zeroH=0, return_REL=True):
    nV = len(toksV)
    nH = len(motsH)

    couts = [[0]*(1+nH) for _ in range(1+nV)]
    mvts =  [[None]*(1+nH) for _ in range(1+nV)]
    couts[0][0] = 0
    cumul = 0
    for j in range(1, 1+nH):
        couts[0][j] = cumul = 1 + cumul
        mvts[0][j] = (1, 0)
    cumul = 0
    for i in range(1, 1+nV):
        couts[i][0] = cumul = 1 + cumul
        mvts[i][0] = (0, 1)
    
    for j in range(1, 1+nH):
        mot_   = motsH[j-1].lower()
        mot = mot_
        #cout_en_haut = couts[0][j]
        diago = 0
        for i in range(1, 1+nV):
            token = toksV[i-1].lower()
            mvt = (1, 0)
            cout = couts[i][j-1]+1
            C = couts[i-1][j]+1
            if C < cout:
                mvt = (0, 1) 
                cout = C
            if mot == token:
                C = couts[i-1-diago][j-1]
                if C <= cout:
                    cout = C
                    mvt = (1, diago+1)
                diago = 0
            elif mot.startswith(token):
                mot = mot[len(token):]
                diago += 1
            elif diago > 0:
                mot = mot_
                diago = 0
            cout_en_haut = couts[i][j] = cout
            mvts[i][j] = mvt
    
    chem = []
    posV,posH = nV, nH
    while True:
        mvt = mvts[posV][posH]
        if mvt is None:
            break
        chem.append(mvt)
        H, V = mvt
        posV, posH = posV-V, posH-H
    chem = chem[::-1]

    afficher_alignement(motsH, toksV, chem)

    chem = grouper_aligs(chem)
    print("\n##############\n")

    afficher_alignement(motsH, toksV, chem)

    if return_REL:
        return aligs_to_RELs(chem, zeroV, zeroH)
    else:
        return chem
    

def aligneur_par_caracteres(motsH, toksV):
    motsH_ = motsH
    toksV_ = toksV
    motsH = " ".join(motsH)
    toksV = " ".join(toksV)
    nV = len(toksV)
    nH = len(motsH)
    infini = float("inf")
    INITIAL = 1
    INSERT_H = 2
    INSERT_V = 3
    CORRES_VH = 4

    coutsCARH = [infini, 1, 1, infini, infini]
    coutsSPCH = [infini, 1, None, infini, 1]
    coutsCARV = [infini, 1, infini, 1, infini]
    coutsSPCV = [infini, 1, infini, None, 1]

    couts = [[0]*(1+nH) for _ in range(1+nV)]
    mvts =  [[None]*(1+nH) for _ in range(1+nV)]
    etats = [[0]*(1+nH) for _ in range(1+nV)]
    couts[0][0] = 0
    etats[0][0] = INITIAL
    cumul = 0
    for j in range(1, 1+nH):
        couts[0][j] = cumul = 1 + cumul
        mvts[0][j] = (1, 0)
        if motsH[j-1] == " " or j == nH or motsH[j] == " ":
            etats[0][j] = INITIAL
        else:
            etats[0][j] = INSERT_H
    cumul = 0
    for i in range(1, 1+nV):
        couts[i][0] = cumul = 1 + cumul
        mvts[i][0] = (0, 1)
        if toksV[i-1] == " " or i == nV or toksV[i] == " ":
            etats[i][0] = INITIAL
        else:
            etats[i][0] = INSERT_V

    for i in range(1, 1+nV):
        carV = toksV[i-1].lower()
        for j in range(1, 1+nH):
            carH = motsH[j-1].lower()
            etat = etats[i][j-1]
            if carH == " ":
                cout = coutsSPCH[etat]
                if cout == infini:
                    etatf = 0
                elif etat == CORRES_VH:
                    etatf = CORRES_VH
                else:
                    etatf = INITIAL
            else:
                cout = coutsCARH[etat]
                if cout == infini:
                    etatf = 0
                elif j == nH or motsH[j] == " ":
                    etatf = INITIAL
                else:
                    etatf = INSERT_H
            cout += couts[i][j-1] 
            if cout < infini:
                mvt = (1, 0)
            else:
                mvt = None
            
            etat = etats[i-1][j]
            if carV == " ":
                coutV = coutsSPCV[etat]
                if coutV == infini:
                    etatfV = 0
                elif etat == CORRES_VH:
                    etatfV = CORRES_VH
                else:
                    etatfV = INITIAL
            else:
                coutV = coutsCARV[etat]
                if coutV == infini:
                    etatfV = 0
                elif i == nV or toksV[i] == " ":
                    etatfV = INITIAL
                else:
                    etatfV = INSERT_V
            coutV += couts[i-1][j]
            if coutV < infini:
                mvtV = (0, 1)
            else:
                mvtV = None
            if coutV < cout:
                cout = coutV
                mvt = mvtV
                etatf = etatfV

            etat = etats[i-1][j-1]
            coutD = infini
            etatfD = 0
            if carH == carV:
                if carH == " ":
                    coutD = 0
                    etatfD = INITIAL
                elif etat in [INITIAL, CORRES_VH]:
                    coutD = 0
                    if (j == nH or motsH[j] == " ") and (i == nV or toksV[i] == " "):
                        etatfD = INITIAL
                    else:
                        etatfD = CORRES_VH
            coutD += couts[i-1][j-1]
            if coutD < infini:
                mvtD = (1,1)
            else:
                mvtD = None
            if coutD <= cout:
                cout = coutD
                mvt = mvtD
                etatf = etatfD

            couts[i][j] = cout
            mvts[i][j]  = mvt
            etats[i][j] = etatf

    chem = []
    posV,posH = nV, nH
    while True:
        mvt = mvts[posV][posH]
        etat = etats[posV][posH]
        if mvt is None:
            break
        chem.append(mvt)
        H, V = mvt
        posV, posH = posV-V, posH-H
    chem.append((0,0))
    chem = chem[::-1]
    aligsH = []
    aligsV = []
    aligH = []
    aligV = []
    posV = posH = 0
    numH = numV = 0
    motsH += " "
    toksV += " "
    for H, V in chem:  #[1:]:
        posH += H
        posV += V
        carH = motsH[posH]
        carV = toksV[posV]
        if carH == carV == " ":
            aligH.append(numH)
            numH += 1
            aligV.append(numV)
            numV += 1
            aligsH.append(aligH)
            aligsV.append(aligV)
            aligH = []
            aligV = []
        elif carH == " ":
            aligH.append(numH)
            numH += 1
        elif carV == " ":
            aligV.append(numV)
            numV += 1
    afficher_aligs2(motsH_, toksV_, aligsH, aligsV)



            

            

            
            







    


if __name__ == "__main__":
    motsH = [T for T in "1234"]
    toksV = [T for T in "1A4"]

    #motsH = "Suave , mari magno , ( (".split()
    #toksV = "Suave, mari mag no, ((".split()
    #aligneur_par_caracteres(motsH, toksV)

    motsH = "Suave , mari magno , ( ( turbantibus aequora ventis ) ) e terra magnum . . .".split()
    toksV = "Suave, mari mag no, (( turban tibus ae quora ventis )) e terra magnum ...".split()
    aligneur_par_caracteres(motsH, toksV)
    #aligs = better_seq_aligner(motsH, toksV, return_REL=False)
    #aligs = aligneur_seq(motsH, toksV, return_REL=False)