strategies = {
        "and_op_N" : {"conj" : "{and}"},
        "or_op_N" : {"conj" : "{or}"},
        "slash_op_N" : {"conj" : "{and_or}"},
        "and-or_op_N" : {"conj" : "{and_or}"},
        "into_op_1" : "prep",
        "after_op_1": "prep",
        "in-excess-of_op_1" : "prep",
        "name_op_1" : True,
        "name_op_N" : True,
        "multiple_op_1" : "adj",
        "any_op_1" : "adj",
        "either_op_N": {"conj" : "{or}"},
        "further_op_1": "adj",
        "half_op_1": "adj",
        "south_op_1": "adj",
        "probable_op_1": "adj",
        "underneath_op_1": "prep",
        "while_op_1": "prep",
        "too_op_1": "modifies",
        "no-more-than_op_1": "prep",
        "amongst_op_1": "prep",
        "top_op_1": "prep",
        "beside_op_1": "prep",
        "even_op_1": "prep",
        'for-all_op_1'	: "prep",
        'for_op_1'	: "prep",
        'now_op_1'	: "adj",
        'plus_op_1'	: "prep",
        'next_op_1'	: "adj",
        'so-far_op_1'	: "adj",
        'beneath_op_1'	: "prep",
        'onto_op_1'	: "prep",
        'till_op_1'	: "prep",
        'towards_op_1'	: "prep",
        'toward_op_1'	: "prep",
        'both_op_N'	: {"conj" : "{and}"},
        'only_op_1'	: "adj",
        'close-to_op_1'	: "prep",
        'as_op_1'	: "prep",
        'at-most_op_1'	: "adj",
        'near_op_1'	: "adj",
        'alongside_op_1'	: "prep",
        'as-much-as_op_1'	: "prep",
        'roughly_op_1'	: "adj",
        'even-as_op_1'	: "prep",
        'middle_op_1'	: "adj",
        'on_op_1'	: "prep",
        'sum-of_op_N'	: {"conj" : "{and}"},
        'once_op_1'	: "prep",
        'as-many-as_op_1'	: "prep",
        'value-interval_op_N'	: {"conj" : "{inter}"},
        'versus_op_N' :	{"conj" : "{inter}"},
        'as-far-as_op_1'	: "prep",
        'ahead_op_1'	: "adj",
        'past_op_1'	: "adj",
        'as-soon-as_op_1'	: "prep",
        'even-when_op_1'	: "prep",
        'out-of_op_1'	: "prep",
        'within_op_1'	: "prep",
        'among_op_1'	: "prep",
        'throughout_op_1'	: "prep",
        'away_op_1'	: "adj",
        'all-over_op_1'	: "prep",
        'down_op_1'	: "prep",
        'prior_op_1'	: "adj",
        'below_op_1'	: "prep",
        'next-to_op_1'	: "prep",
        'in-front-of_op_1'	: "prep",
        'between_op_1'	: "prep",
        'from_op_1'	: "prep",
        'many_op_1'	: "adj",
        'almost_op_1'	: "adj",
        'couple_op_1'	: "adj",
        'inside_op_1'	: "prep",
        'some_op_1'	: "adj",
        'off_op_1'	: "prep",
        'through_op_1'	: "prep",
        'along_op_1'	: "prep",
        'less-than_op_1'	: "prep",
        'as-long-as_op_1'	: "prep",
        'against_op_1'	: "prep",
        'nearly_op_1'	: "adj",
        'above_op_1'	: "prep",
        'early_op_1'	: "adj",
        'late_op_1'	: "adj",
        'several_op_1'	: "adj",
        'behind_op_1'	: "prep",
        'beyond_op_1'	: "prep",
        'by_op_1'	: "prep",
        'outside_op_1'	: "prep",
        'amr-choice_op_N'	: {"conj" : "{or}"},
        'approximately_op_1'	: "adj",
        'across_op_1'	: "prep",
        'even-if_op_1'	: "prep",
        'under_op_1'	: "prep",
        'up-to_op_1'	: "prep",
        'date-interval_op_N'	: {"conj" : "{inter}"},
        'few_op_1'	: "adj",
        'relative-position_op_1'	: "prep",
        'at-least_op_1'	: "adj",
        'until_op_1'	: "prep",
        'around_op_1'	: "prep",
        'over_op_1'	: "prep",
        'about_op_1'	: "prep",
        'between_op_N'	: {"conj" : "{inter}"},
        'more-than_op_1'	: "prep",
        'since_op_1'	: "prep",
        'multiple_op_1'	: "adj",
        'before_op_1'	: "prep",
        'or_op_N'	: {"conj" : "{or}"},
        'name_op_N'	: {"conj" : "{and}"},
        'name_op_1'	: "prep",
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
    elif strat == "modifies":
        elim_ascen = True
        elim_syntax = True
        elim_descen = False
        distr_parents = True
        distr_enfants = False
        modif_syntax = {"syn": ":mod", "reverse": False}
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