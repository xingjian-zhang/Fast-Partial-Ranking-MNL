"""Process Flickr data for analysis.

- 5.3 Experiments on Real-World Network Data

We sample 50000 users in a period (20 days) that have following events, and
record all the new edges formed by them. Edges formed earlier are assumed to be more
preferable than later ones. We then sample 100 edges uniformly at random from
the dataset as the negative samples for each user. The testing set is sampled
in the same way, with the sampling period later for 20 days.

The data should be downloaded first with:
wget -4 http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz ../data/
"""

import os
import random
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
from tqdm import tqdm


def collect_edge_features(i, j, alt=1):
    subsubset = {}
    if j in G.nodes:
        deg = G.in_degree(j)
    else:
        deg = 0
    try:
        hops = nx.shortest_path_length(G, source=i, target=j)
    except Exception:
        hops = 'NA'
    recip = 1 if G.has_edge(j, i) else 0
    subsubset["deg"] = deg
    subsubset["hops"] = hops
    subsubset["recip"] = recip
    subsubset["alt"] = alt
    return subsubset


data_path = '../data'

# date to start sampling from
start_date = '2006-11-05'
end_date = '2006-11-25'

# file names
fn_in = data_path + '/flickr-growth.txt.gz'
fn_out = data_path + '/flickr-growth_choices_p_%s.csv' % \
    datetime.strptime(start_date, '%Y-%m-%d').strftime('%y%m%d')
url = 'http://socialnetworks.mpi-sws.mpg.de/data'


# check if the input data has been downloaded yet
if not os.path.exists(fn_in):
    print("[ERROR] Input data not found. Please download with:\n" +
          "        wget %s/flickr-growth.txt.gz %s/ " % (url, data_path))

# check if the output data exists already
if os.path.exists(fn_out):
    print("[ERROR] Output data already exists! Please remove it to run.")

print("Reading raw data and creating graph.")
# read the edge list data
DF = pd.read_csv(fn_in, compression='gzip', header=0, sep='\t',
                 names=['from', 'to', 'ds'])
el_pre = DF[DF.ds < start_date]
el_pre = list(zip(el_pre['from'], el_pre['to'], el_pre['ds']))
el_post = DF[(DF.ds >= start_date) & (DF.ds < end_date)]
el_test = DF[DF.ds == end_date]

# create starting graph
G = nx.DiGraph()
G.add_edges_from([(x[0], x[1]) for x in el_pre])
print("Starting graph has %d nodes." % len(G))

num_samples = 60000
su = set(random.sample(list(el_post["from"].unique()), num_samples))
su = list(su.intersection(G.nodes))[:50000]

print("Collect train data.")

n = 0
ds = start_date
start_time = datetime.strptime(start_date, '%Y-%m-%d')
data = {i: [] for i in su}
while ds != end_date:
    el_curr = el_post[el_post["ds"] == ds]
    for i in tqdm(su, desc=ds):
        for j in el_curr[el_curr["from"] == i]["to"].values:
            data[i].append(collect_edge_features(i, j, 1 + n))
    edges = el_curr[["from", "to"]].values.tolist()
    G.add_edges_from(edges)
    n += 1
    ds = (start_time + timedelta(n)).strftime('%Y-%m-%d')

n_neg = 100
for i in tqdm(su):
    neg_samples = random.sample(G.nodes, n_neg+10)
    succs = set(G.successors(i))
    neg_samples = list(set(neg_samples) - succs)[:n_neg]
    for j in neg_samples:
        data[i].append(collect_edge_features(i, j, 0))

entries = []
for idx, v in enumerate(data.values()):
    for entry in v:
        entry["index"] = idx
        entries.append(entry)

pd.DataFrame(entries).to_csv("flickr_train.csv")

print("Collect test data.")

n_samples = 8000
n_neg = 100
samples = random.sample(list(DF[DF["ds"] == end_date]["from"].unique()), n_samples)
samples = list(set(samples).intersection(G.nodes))[:5000]
n = 0
ds = end_date
end_time = datetime.strptime(end_date, '%Y-%m-%d')
test_data = {i:[] for i in samples}
while n < 20:
    el_curr = DF[DF["ds"] == ds]
    for i in tqdm(samples, desc=ds):
        for j in el_curr[el_curr["from"] == i]["to"].values:
            test_data[i].append(collect_edge_features(i, j, 1))
    edges = el_curr[["from", "to"]].values.tolist()
    G.add_edges_from(edges)
    n += 1
    ds = (end_time + timedelta(n)).strftime('%Y-%m-%d')
n_neg = 100

for i in tqdm(samples):
    neg_samples = random.sample(G.nodes, n_neg+10)
    succs = set(G.successors(i))
    neg_samples = list(set(neg_samples) - succs)[:n_neg]
    for j in neg_samples:
        test_data[i].append(collect_edge_features(i, j, 0))

entries = []
for idx, v in enumerate(test_data.values()):
    for entry in v:
        entry["index"] = idx
        entries.append(entry)
pd.DataFrame(entries).to_csv("flickr_test.csv")