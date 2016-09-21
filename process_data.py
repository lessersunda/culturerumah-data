#!/usr/bin/env python

import argparse
import json
import re
import pandas

from collections import OrderedDict, defaultdict
from itertools import cycle
from nameparser import HumanName

try:
    from clld.db.meta import DBSession
    
    from clld.db.models.common import (
        Dataset, DomainElement, Contributor, ContributionContributor, ValueSet, Value)
    from grambank.models import Feature, GrambankContribution, GrambankLanguage

    from clld.web.icon import ORDERED_ICONS
    
    model_is_available=True
except ImportError:
    class DummyDBSession:
        def add(self, data): pass
    DBSession = DummyDBSession()

    class Ignore:
        def __init__(self, *args, **kwargs): pass

    class DomainElement:
        def __init__(self, description, **kwargs):
            self.description = description
    Dataset = Contributor = ContributionContributor = ValueSet = Value = Ignore
    Feature = GrambankContribution = GrambankLanguage = Ignore

    ORDERED_ICONS = defaultdict(lambda: None)
    
    model_is_available=False

def yield_domainelements(s):
    for m in re.split('\s*,|;\s*', re.sub('^multistate\s+', '', s.strip())):
        if m.strip():
            if m.startswith('As many'):
                for i in range(100):
                    yield '%s' % i, '%s' % i
            else:
                number, desc = m.split(':')
                yield number.strip(), desc.strip()
    yield '?', 'Not known'

features_path = "features_collaborative_sheet.tsv"
def import_features():
    features = pandas.io.parsers.read_csv(
        features_path,
        sep='\t',
        index_col="GramBank ID",
        encoding='utf-16')
    features["db_Object"] = [
        Feature( 
            id = i,
            name = d['Feature'],
            doc = d['Clarifying Comments'],
            patron = d['Feature patron'],
            std_comments = d['Suggested standardised comments'],
            name_french = d['Feature question in French'],
            jl_relevant_unit = d['Relevant unit(s)'],
            jl_function = d['Function'],
            jl_formal_means = d['Formal means'],
            hard_to_deny = d['Very hard to deny'],
            prone_misunderstanding = d['Prone to misunderstandings among researchers'],
            requires_extensive_data = d['Requires extensive data on the language'],
            last_edited = d['Last edited'],
            other_survey = d['Is there a typological survey that already covers this feature somehow?'])
        for i, d in features.iterrows()]
    features["db_Domain"] = [
        {deid: DomainElement(
            id='{:s}-{:s}'.format(i, deid),
            parameter=d['db_Object'],
            abbr=deid,
            name='{:s} - {:s}'.format(deid, desc),
            number=int(deid) if deid != '?' else 999,
            description=desc,
            jsondata={'icon': ORDERED_ICONS[int(deid)].name} if deid != '?' else {})
         for deid, desc in yield_domainelements(d['Possible Values'])}
        for i, d in features.iterrows()]
    return features

languages_path = '../lexirumah-data/languages.tsv'
def import_languages():
    # TODO: be independent of the location of lexirumah-data, but do compare with it!
    languages = pandas.io.parsers.read_csv(
        languages_path,
        sep='\t',
        index_col="Language ID",
        encoding='utf-16')
    # TODO: Produce language objects
    languages["db_Object"] = [
        GrambankLanguage(
            id=i,
            name=row['Language name (-dialect)'],
            latitude=row['Lat'],
            longitude=row['Lon'])
        for i, row in languages.iterrows()]
    return languages

def report(problem, data1, data2):
    print(problem)
    print(data1)
    print("<->")
    print(data2)
    print("     [ ]")
    print()

def import_contribution(path, icons, features, languages, contributors={}, trust=[]):
    # look for metadata
    # look for sources
    # then loop over values
    
    mdpath = path + '-metadata.json'
    with open(mdpath) as mdfile:
        md = json.load(mdfile)

    try:
        abstract = md["abstract"]
    except KeyError:
        md["abstract"] = "Typological features of {:s}. Coded by {:s} following {:}.".format(
            md["language"],
            md["creator"],
            md["source"]+md["references"])  
      
    contrib = GrambankContribution(
        id=md["id"],
        name=md["name"],
        #sources=sources(md["source"]) + references(md["references"]),
        ## GrambankContribution can't take sources arguments yet.
        ## We expect "source" to stand for primary linguistic data (audio files etc.),
        ## and "references" to point to bibliographic data.
        desc=md["abstract"])
    contributor_name = HumanName(md["creator"])
    contributor_id = (contributor_name.last + contributor_name.first)
    try:
        contributor = contributors[md["creator"]]
    except KeyError:
        contributor = Contributor(
            id=contributor_id,
            name=contributor_name)
    DBSession.add(ContributionContributor(contribution=contrib, contributor=contributor))

    if mdpath not in trust:
        with open(mdpath, "w") as mdfile:
            json.dump(md, mdfile, indent=2)

    data = pandas.io.parsers.read_csv(
            path,
            sep="," if path.endswith(".csv") else "\t",
            encoding='utf-16')

    if "Language_ID" not in data.columns:
        data["Language_ID"] = md["language"]
    elif mdpath in trust:
        if path in trust:
            assert (data["Language_ID"] == md["language"]).all() 
        else:
            data["Language_ID"] = md["language"]
    else:
        if (data["Language_ID"] != md["language"]).any():
            report(
                "Language mismatch:",
                md["language"],
                data[data["Language_ID"] != md["language"]].to_string())
    language = languages.loc[md["language"]]

    if "Source" not in data.columns:
        data["Source"] = ""
    if "Question" not in data.columns:
        data["Question"] = ""
    if "Answer" not in data.columns:
        data["Answer"] = ""

    data["Value"] = data["Value"].astype(int)
    data["Source"] = data["Source"].astype(str)
    data["Question"] = data["Question"].astype(str)
    data["Answer"] = data["Answer"].astype(str)

    for i, row in data.iterrows():
        value = row['Value']
        feature = row['Feature_ID']
        assert not pandas.isnull(value)
        assert not pandas.isnull(feature)

        try:
            parameter = features.loc[feature]
        except (TypeError, KeyError):
            if path in trust:
                if features_path in trust:
                    raise AssertionError("{:s} and {:s} don't match!".format(
                        path,
                        features_path))
                else:
                    parameter = features.loc[feature] = {}
            else:
                report(
                    "Feature mismatch:",
                    feature,
                    features.index)
                if features_path in trust:
                    del data.loc[i]
                    continue
                else:
                    parameter = {}

        question = row["Question"]
        if question != parameter["Feature"]:
            if path in trust:
                if features_path in trust:
                    raise AssertionError("Feature question mismatch!")
                else:
                    parameter["Feature"] = question
            else:
                if features_path in trust:
                    data.set_value(i, "Question", parameter["Feature"])
                else:
                    print(trust)
                    report(
                        "Feature question mismatch!",
                        question,
                        parameter["Feature"])

        vs = ValueSet(
            id="{:s}-{:s}".format(md["language"], feature),
            parameter=parameter["db_Object"],
            language=language["db_Object"],
            contribution=contrib,
            source=row['Source'])

        domain = parameter["db_Domain"]   
        name = str(row['Value'])
        if name not in domain:
            if path in trust:
                deid = max(domain)+1
                domainelement = domain[name] = DomainElement(
                    id='{:s}-{:s}'.format(i, deid),
                    parameter=parameter['db_Object'],
                    abbr=deid,
                    name='{:s} - {:s}'.format(deid, desc),
                    number=int(deid) if deid != '?' else 999,
                    description=desc,
                    jsondata={'icon': ORDERED_ICONS[int(deid)].name})
            else:
                report(
                    "Feature domain mismatch:",
                    list(domain.keys()),
                    name)
                continue
        else:
            domainelement = domain[name]

        answer = row["Answer"]
        if answer != domainelement.description:
            if path in trust:
                if features_path in trust:
                    raise AssertionError("Feature domain element mismatch!")
                else:
                    domainelement.desc = answer
            else:
                if features_path in trust:
                    data.set_value(i, "Answer", domainelement.description)
                else:
                    print(trust)
                    report(
                        "Feature domain element mismatch!",
                        answer,
                        domainelement.description)
        

        DBSession.add(Value(
                 id="{:s}-{:s}-{:s}".format(md["language"], feature, name),
                 valueset=vs,
                 name=name,
                 description=row['Comment'],
                 domainelement=domainelement))

        print(".", end="")
    if path not in trust:
        data.sort_values(by=["Feature_ID", "Value"], inplace=True)
        data = data[["Language_ID", "Feature_ID", "Question", "Value", "Answer", "Comment", "Source"]]
        data.to_csv(
            path,
            index=False,
            sep="," if path.endswith(".csv") else "\t",
            encoding='utf-16')
    return data
            

def import_cldf(srcdir, features, languages, trust=[]):
    # loop over values
    # check if language needs to be inserted
    # check if feature needs to be inserted
    # add value if in domain
    icons = cycle(ORDERED_ICONS)
    datasets = {}
    for dirpath, dnames, fnames in os.walk(srcdir):
        for fname in fnames:
            if os.path.splitext(fname)[1] in ['.tsv', '.csv']:
                datasets[os.path.join(dirpath, fname)] = import_contribution(
                    os.path.join(dirpath, fname),
                    icons,
                    features,
                    languages,
                    trust=trust)
                print("Imported {:s}.".format(os.path.join(dirpath, fname)))
    return datasets

def main(trust=[], sqlite=None):
    with open("metadata.json") as md:
        dataset_metadata = json.load(md)
    DBSession.add(
        Dataset(
            id=dataset_metadata["id"],
            name=dataset_metadata["name"],
            publisher_name=dataset_metadata["publisher_name"],
            publisher_place=dataset_metadata["publisher_place"],
            publisher_url=dataset_metadata["publisher_url"],
            license=dataset_metadata["license"],
            domain=dataset_metadata["domain"],
            contact=dataset_metadata["contact"],
            jsondata={
                'license_icon': dataset_metadata["license_icon"],
                'license_name': dataset_metadata["license_name"]}))

    features = import_features()
    languages = import_languages()
    import_cldf("datasets", features, languages, trust=trust)
    if languages_path not in trust:
        languages.to_csv(
            languages_path,
            sep='\t',
            encoding='utf-16')
    if features_path not in trust:
        features.to_csv(
            features_path,
            sep='\t',
            encoding='utf-16')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GramRumah data with consistency in mind")
    if model_is_available:
        parser.add_argument("--sqlite", default=None, const="gramrumah.sqlite", nargs="?",
                            help="Generate an sqlite database from the data")
    parser.add_argument("--trust", "-t", nargs="*", type=argparse.FileType("r"),
                        help="Data files to be trusted in case of mismatch")
    args = parser.parse_args()
    main(args.trust, args.sqlite)
