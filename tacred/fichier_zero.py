import json
import os
from collections import defaultdict

relations = [
    'no_relation', 'org:alternate_names', 'org:city_of_headquarters', 'org:country_of_headquarters',
    'org:dissolved', 'org:founded', 'org:founded_by', 'org:member_of', 'org:members',
    'org:number_of_employees/members', 'org:parents', 'org:political/religious_affiliation',
    'org:shareholders', 'org:stateorprovince_of_headquarters', 'org:subsidiaries',
    'org:top_members/employees', 'org:website', 'per:age', 'per:alternate_names', 'per:cause_of_death',
    'per:charges', 'per:children', 'per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death',
    'per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:date_of_birth',
    'per:date_of_death', 'per:employee_of', 'per:origin', 'per:other_family', 'per:parents',
    'per:religion', 'per:schools_attended', 'per:siblings', 'per:spouse', 'per:stateorprovince_of_birth',
    'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence', 'per:title'
]
# Cette liste de relations a été obtenue avec la fonction suivante. Elle contient 42 entrées.

effectifs = {
    'org:founded_by': 268, 'no_relation': 84491, 'per:employee_of': 2163, 'org:alternate_names': 1359,
    'per:cities_of_residence': 742, 'per:children': 347, 'per:title': 3862, 'per:siblings': 250,
    'per:religion': 153, 'per:age': 833, 'org:website': 223, 'per:stateorprovinces_of_residence': 484,
    'org:member_of': 171, 'org:top_members/employees': 2770, 'per:countries_of_residence': 819,
    'org:city_of_headquarters': 573, 'org:members': 286, 'org:country_of_headquarters': 753,
    'per:spouse': 483, 'org:stateorprovince_of_headquarters': 350, 'org:number_of_employees/members': 121,
    'org:parents': 444, 'org:subsidiaries': 453, 'per:origin': 667, 'org:political/religious_affiliation': 125,
    'per:other_family': 319, 'per:stateorprovince_of_birth': 72, 'org:dissolved': 33, 'per:date_of_death': 394,
    'org:shareholders': 144, 'per:alternate_names': 153, 'per:parents': 296, 'per:schools_attended': 229,
    'per:cause_of_death': 337, 'per:city_of_death': 227, 'per:stateorprovince_of_death': 104,
    'org:founded': 166, 'per:country_of_birth': 53, 'per:date_of_birth': 103, 'per:city_of_birth': 103,
    'per:charges': 280, 'per:country_of_death': 61
}

def make_rel_list(files):
    rels = set()
    for fil in files:
        with open(fil, "r", encoding="utf-8") as F:
            docu = json.load(F)
            rels = rels.union(set(X["relation"] for X in docu))
    rels=list(rels)
    rels.sort()
    return rels

def make_rel_stats(files):
    stats = defaultdict(lambda: 0)
    for fil in files:
        with open(fil, "r", encoding="utf-8") as F:
            docu = json.load(F)
            for X in docu:
                stats[X["relation"]] += 1
    #for rel in relations:
    #    print("%s\t\t\t:%d"%(rel, stats[rel]))
    return dict(stats)


def list_first_char(files):
    alphabet = "azertyuiopqsdfghjklmwxcvbn"
    resu = set()
    for fil in files:
        with open(fil, "r", encoding="utf-8") as F:
            docu = json.load(F)
            for X in docu:
                tokens = X["token"]
                inits = [X[0] for X in tokens]
                inits = ["alpha" if X in alphabet else X for X in inits]
                resu = resu.union(inits)
    resu = list(resu)
    resu.sort()
    return resu

def rebuild_sentence(tokens):
    no_space = "!',-./:;?_·’"
    inits = [tok[0] for tok in tokens]
    inits = ["" if X in no_space else " " for X in inits]
    inits[0] = ""
    tokens = [Z[0] + Z[1] for Z in zip(inits, tokens)]
    phrase = "".join(tokens)
    return phrase


if __name__ == "__main__":
    rep = "C:\\Users\\fcharpentier\\Documents\\Boulot\\visuAMR\\tacred_LDC2018T24\\tacred_LDC2018T24\\tacred\\data\\json"
    files = [os.path.join(rep, X) for X in ["train.json", "dev.json", "test.json"]]
    #rels = make_rel_list(files)
    #print(len(rels))
    #########
    #stats = make_rel_stats(files)
    #print(repr(stats))
    inits = list_first_char(files)
    print(repr(inits))
    