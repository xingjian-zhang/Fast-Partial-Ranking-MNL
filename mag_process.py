"""Process MAG data for analysis.

- 5.3 Experiments on Real-World Network Data

The raw MAG data is too large (~800G). You can download a processed version here:
https://drive.google.com/file/d/17bgLs1iR96JW3Rd0mex3IK8qyU-qRElB/view?usp=sharing

"""


import csv
import json
import os
import random
from collections import Counter
from math import log

import pandas as pd
from tqdm import tqdm

raw_data = pd.read_csv("../data/mag_cli.csv")

raw_data["authors"] = raw_data["authors"].astype(str)
raw_data[raw_data["year"] > 1950]

critical_year = 2010

old_data = raw_data[raw_data["year"] < critical_year]
authors = raw_data["authors"].values.tolist()
names = [x for xs in authors for x in str(xs).split(",") if x != ""]
author_publication = {author: [] for author in names}

years_span = sorted(list(raw_data["year"].unique()))
print(min(years_span), max(years_span))

papers = old_data["id"].values.tolist()

papers_citation = {k: 0 for k in raw_data["id"].values}
papers_max = {k: 0 for k in raw_data["id"].values}
author_publication = {author: 0 for author in names}

total_edges = 0
include_edges = 0

# collect information before 2010
bar = tqdm(old_data.iterrows(), total=len(old_data))
for idx, row in bar:
    # count number of publications
    co_authors = row["authors"].split(",")
    year = row["year"]
    max_ = []
    for a in co_authors:
        if a == '':
            continue
        author_publication[a] += 1
        max_.append(author_publication[a])
    papers_max[row["id"]] = max(max_)
    # count number of publications
    references = row["references"]
    if pd.notna(references) and year < critical_year:
        references = references.split(",")
        for ref in references:
            total_edges += 1
            try:
                papers_citation[ref] += 1
                include_edges += 1
            except KeyError:
                continue  # not in this domain

print("#Include/#total:{}/{}".format(include_edges, total_edges))


papers_citation = dict(sorted(papers_citation.items(),
                       key=lambda item: item[1], reverse=True))
dict(list(papers_citation.items())[:20])


new_data = raw_data[raw_data["year"] >= critical_year]


new_data = new_data.sort_values(["year"])
new_data.head(10)


def extract_features(source: str, target: str) -> tuple:
    # collect
    # 1. log target degrees
    # 2. target has degrees
    # 3. target has same author with source
    # 4. log age
    # 5. target author_max papers
    if target not in papers_citation:
        raise KeyError("target {} not in domain.".format(target))

    # degree
    degree = papers_citation[target]
    log_degree = log(degree) if degree != 0 else 0
    has_degree = 1 if degree != 0 else 0

    # share authors
    source_authors = set(new_data.loc[source]["authors"].split(","))
    target_authors = set(raw_data.loc[target]["authors"].split(","))
    is_share_authors = 1 if bool(source_authors & target_authors) else 0

    # age
    age = new_data.loc[source]["year"] - raw_data.loc[target]["year"]
    log_age = log(age) if age != 0 else 0

    # MAX papers
    max_papers = papers_max[target]

    return (log_degree, has_degree, is_share_authors, log_age, max_papers)


new_data.set_index("id", drop=False, inplace=True)
old_data.set_index("id", drop=False, inplace=True)
raw_data.set_index("id", drop=False, inplace=True)


num_samples = 12000
num_negatives = 5000
sample_rate = 0.2
samples = []

n_sampled = 0
row_iter = new_data.iterrows()
year = critical_year
papers_citation_ = papers_citation.copy()
papers_max_ = papers_max.copy()
bar = tqdm(total=num_samples)
while n_sampled < num_samples:
    _, row = next(row_iter)
    old_year = year
    year = row["year"]
    if year != old_year:  # update info
        papers_citation = papers_citation_.copy()
        papers_max = papers_max_.copy()
        print(year, "\n")

    source = row["id"]
    targets = row["references"]

    # update papers_max
    authors = row["authors"].split(",")
    max_ = []
    for a in authors:
        if a != '':
            author_publication[a] += 1
            max_.append(author_publication[a])
    papers_max_[source] = max(max_)
    # update papers_citations
    try:
        papers_citation_[ref] += 1
        include_edges += 1
    except KeyError:
        continue  # not in this domain

    if pd.notna(targets):
        if random.random() < sample_rate:
            targets = targets.split(",")
            pos_samples = []
            neg_samples = []
            # collect pos_samples
            for target in targets:
                if target in papers:
                    pos_samples.append(extract_features(source, target))
            if len(pos_samples) == 0:
                continue
            # collect neg_samples
            neg_targets = set(random.sample(papers, num_negatives + 10))
            neg_targets = list(neg_targets - set(targets))[:num_negatives]
            for target in neg_targets:
                neg_samples.append(extract_features(source, target))

            samples.append((pos_samples, neg_samples))
            n_sampled += 1
            bar.update(1)

with open(f"mag.json", "w+") as fp:
    json.dump(samples, fp, indent=2)
