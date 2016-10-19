#!/usr/bin/env python

import os
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
    from culturebank.models import (
        Feature, CulturebankContribution, CulturebankLanguage, Family)

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
    Feature = CulturebankContribution = CulturebankLanguage = Ignore

    class Icon:
        name = None
    ORDERED_ICONS = defaultdict(lambda: Icon())

    model_is_available=False

def yield_domainelements(s):
    done = {"?"}
    for m in re.split('\s*,|;\s*', re.sub('^multistate\s+', '', s.strip())):
        if m.strip():
            if m.startswith('As many'):
                for i in range(100):
                    yield '%s' % i, '%s' % i
            else:
                number, desc = m.split(':')
                if number == '?':
                    continue
                if number in done:
                    raise ValueError("Value specified multiple times",
                                     s)
                else:
                    done.add(number)
                yield number.strip(), desc.strip()
    yield '?', 'Not known'

features_path = "features_collaborative_sheet.tsv"
def import_features():
    features = pandas.io.parsers.read_csv(
        features_path,
        sep='\t',
        index_col="CultureBank ID",
        encoding='utf-8')
    features["db_Object"] = [
        Feature( 
            id = i,
            name = d['Feature'],
            doc = d['Clarifying Comments'],
            patron = d['Feature patron'],
            std_comments = d['Suggested standardised comments'],
            name_french = d['Feature prompt in Indonesian'],
            #jl_relevant_unit = d['Relevant unit(s)'],
            #jl_function = d['Function'],
            #jl_formal_means = d['Formal means'],
            hard_to_deny = d['Very hard to deny'],
            prone_misunderstanding = d['Prone to misunderstandings among researchers'],
            #requires_extensive_data = d['Requires extensive data on the language'],
            last_edited = d['Last edited'],
            other_survey = d['ID/Reference according to other cultural databases'])
        for i, d in features.iterrows()]
    features["db_Domain"] = [
        {deid: DomainElement(
            id='{:s}-{:s}'.format(i, deid),
            parameter=d['db_Object'],
            abbr=deid,
            name='{:s} - {:s}'.format(deid, desc),
            number=int(deid) if deid != '?' else 999,
            description=desc,
            jsondata={'icon': ORDERED_ICONS[int(deid)].name} if deid != '?' else {'icon': None})
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
        encoding='utf-8')
    families = {
        family: Family(
            jsondata={
                "icon": icon.name},
            id=family,
            name=family)
        for family, icon in zip(set(languages["Family"]), ORDERED_ICONS)}
    languages["db_Object"] = [
        CulturebankLanguage(
            id=i,
            name=row['Language name (-dialect)'],
            family=families[row['Family']],
            macroarea=row['Region'],
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

def possibly_int_string(value):
    try:
        return str(int(value))
    except ValueError:
        try:
            v = float(value)
        except ValueError:
            return str(value)
        if int(v) == v:
            return str(int(v))
        else:
            return str(v)

copy_from_features = ["Feature", "Possible Values", "Suggested standardised comments"]
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
            md["creator"][0],
            md["source"]+md["references"])  

    contrib = CulturebankContribution(
        id=md["id"],
        name=md["name"],
        #sources=sources(md["source"]) + references(md["references"]),
        ## CulturebankContribution can't take sources arguments yet.
        ## We expect "source" to stand for primary linguistic data (audio files etc.),
        ## and "references" to point to bibliographic data.
        desc=md["abstract"])
    contributor_name = HumanName(md["creator"][0])
    contributor_id = (contributor_name.last + contributor_name.first)
    try:
        contributor = contributors[md["creator"][0]]
    except KeyError:
        contributors[md["creator"][0]] = contributor = Contributor(
            id=contributor_id,
            name=str(contributor_name))
    DBSession.add(ContributionContributor(contribution=contrib, contributor=contributor))

    if mdpath not in trust:
        with open(mdpath, "w") as mdfile:
            json.dump(md, mdfile, indent=2)

    data = pandas.io.parsers.read_csv(
            path,
            sep="," if path.endswith(".csv") else "\t",
            encoding='utf-8')

    check_features = features.index.tolist()

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
                data["Language_ID"][data["Language_ID"] != md["language"]].to_string())
    language = languages.loc[md["language"]]

    if "Source" not in data.columns:
        data["Source"] = ""
    if "Answer" not in data.columns:
        data["Answer"] = ""

    data["Value"] = data["Value"].astype(str)
    data["Source"] = data["Source"].astype(str)
    data["Answer"] = data["Answer"].astype(str)

    for column in copy_from_features:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].astype(str)

    features_seen = {}
    for i, row in data.iterrows():
        value = possibly_int_string(row['Value'])
        data.set_value(i, 'Value', value)
        feature = row['Feature_ID']

        if pandas.isnull(feature):
            if pandas.isnull(row['Feature']):
                if path in trust:
                    raise AssertionError("Row {:} without feature found".format(row))
                else:
                    report(
                        "Row without feature found, dropping.",
                        row.to_string(),
                        "")
                    del data.loc[i]
                    continue
            else:
                candidates = features["Feature"]==row["Feature"]
                if candidates.any():
                    feature = candidates.argmax()
                else:
                    report(
                        "Row without matching feature found, ignoring.",
                        row.to_string(),
                        "")
                    continue


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

        for column in copy_from_features:
            question = row[column]
            if (question != parameter[column]
                and not (pandas.isnull(question) or question != "")):
                if path in trust:
                    if features_path in trust:
                        raise AssertionError("{:s} mismatch!".format(column))
                    else:
                        parameter[column] = question
                else:
                    if features_path in trust:
                        data.set_value(i, column, parameter[column])
                    else:
                        report(
                            ("{:s} mismatch!".format(column)),
                            question,
                            parameter[column])
            else:
                data.set_value(i, column, parameter[column])


        if feature in features_seen:
            vs = features_seen[feature]
        else:
            vs = features_seen[feature] = ValueSet(
            id="{:s}-{:s}".format(md["language"], feature),
            parameter=parameter["db_Object"],
            language=language["db_Object"],
            contribution=contrib,
            source=row['Source'])

        domain = parameter["db_Domain"]
        if value not in domain:
            if path in trust:
                deid = max(domain)+1
                domainelement = domain[value] = DomainElement(
                    id='_{:s}-{:s}'.format(i, deid),
                    parameter=parameter['db_Object'],
                    abbr=deid,
                    name='{:s} - {:s}'.format(deid, desc),
                    number=int(deid) if deid != '?' else 999,
                    description=desc,
                    jsondata={'icon': ORDERED_ICONS[int(deid)].name})
            else:
                report(
                    "Feature domain mismatch for {:s}:".format(feature),
                    list(domain.keys()),
                    value)
                continue
        else:
            domainelement = domain[value]

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
                    report(
                        "Feature domain element mismatch!",
                        answer,
                        domainelement.description)


        DBSession.add(Value(
                 id="{:s}-{:s}-{:}{:d}".format(
                     md["language"],
                     feature,
                     value if value!='?' else 'unknown',
                     i),
                 valueset=vs,
                 name=str(value),
                 description=row['Comment'],
                 domainelement=domainelement))

        print(".", end="")

        if feature in check_features:
            check_features.remove(feature)

    if features_path in trust:
        i = data.index.max()
        for feature in check_features:
            i += 1
            for column in copy_from_features:
                data.set_value(i, column, features[column][feature])
            data.set_value(i, "Language_ID", md["language"])
            data.set_value(i, "Feature_ID", feature)
            data.set_value(i, "Value", "?")


    print()
    if path not in trust:
        data.sort_values(by=["Feature_ID", "Value"], inplace=True)
        columns = list(data.columns)
        first_columns = ["Feature_ID",
                         "Language_ID",
                         "Feature",
                         "Value",
                         "Answer",
                         "Comment",
                         "Source",
                         "Possible Values",
                         "Suggested standardised comments"]
        for column in columns:
            if column not in first_columns:
                first_columns.append(column)
        data = data[first_columns]
        data.to_csv(
            path,
            index=False,
            sep="," if path.endswith(".csv") else "\t",
            encoding='utf-8')
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
                print("Importing {:s}…".format(os.path.join(dirpath, fname)))
                datasets[os.path.join(dirpath, fname)] = import_contribution(
                    os.path.join(dirpath, fname),
                    icons,
                    features,
                    languages,
                    trust=trust)
                print("Import done.")
    return datasets


def main(config=None, trust=[languages_path, features_path]):
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
            encoding='utf-8')
    if features_path not in trust:
        features.to_csv(
            features_path,
            sep='\t',
            encoding='utf-8')

import sys
sys.argv=["i", "../culturebank/sqlite.ini"]

if model_is_available:
        from clld.scripts.util import initializedb
        from clld.db.util import compute_language_sources
        try:
            initializedb(create=main, prime_cache=lambda x: None)
        except SystemExit:
            print("done")
else:
        parser = argparse.ArgumentParser(description="Process CultureRumah data with consistency in mind")
        parser.add_argument("--sqlite", default=None, const="gramrumah.sqlite", nargs="?",
                            help="Generate an sqlite database from the data")
        parser.add_argument("--trust", "-t", nargs="*", type=argparse.FileType("r"), default=[],
                            help="Data files to be trusted in case of mismatch")
        args = parser.parse_args()
        main([x.name for x in args.trust])
