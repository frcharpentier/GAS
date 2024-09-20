strategies = {
        "and_op_N" : {"conj" : "{and}"},
        "or_op_N" : {"conj" : "{or}"},
        "slash_op_N" : {"conj" : "{and_or}"},
        "and-or_op_N" : {"conj" : "{and_or}"},
        "into_op_1" : "prep",
        "after_op_1": "prep",
        "name_op_1" : True,
        "name_op_N" : True,
        "multiple_op_1" : "adj",
    }



def definir_strategie(clef):
    if clef in strategies:
        strat = strategies[clef]
    else:
        strat = False

    elim_ascen = True
    elim_syntax = True
    elim_descen = True
    distr_parents = True
    distr_enfants = False
    modif_syntax = False
    conj = False

    if type(strat) is dict and "conj" in strat:
        elim_ascen = True
        elim_syntax = True
        elim_descen = True
        distr_parents = True
        distr_enfants = True
        modif_syntax = False
        conj = strat["conj"]
    elif strat == "prep":
        elim_ascen = True
        elim_syntax = True
        elim_descen = False
        distr_parents = True
        distr_enfants = False
        modif_syntax = "{syntax}"
        conj = False
    elif strat == "adj":
        elim_ascen = True
        elim_syntax = True
        elim_descen = False
        distr_parents = True
        distr_enfants = False
        modif_syntax = {"syn": ":mod", "reverse": True}
        conj = False
    elif strat == True:
        elim_ascen = False
        elim_syntax = False
        elim_descen = False
        distr_parents = False
        distr_enfants = False
        modif_syntax = False
        conj = False
    return (elim_ascen, elim_syntax, elim_descen,
            distr_parents, distr_enfants,
            modif_syntax, conj)