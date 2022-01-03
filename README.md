# Fast-Partial-Ranking-MNL

This repo provides a PyTorch implementation for the CopulaGNN models as described in the following paper:

[Fast Learning of MNL Model From General Partial Rankings with Application to Network Formation Modeling.](https://arxiv.org/abs/2112.15575)

Jiaqi Ma*, Xingjian Zhang*, and Qiaozhu Mei. WSDM 2022.


## Requirements
The code requires the following packages.

```
more_itertools==8.10.0
networkx==2.5.1
numpy==1.19.5
pandas==1.1.5
pyclustering==0.10.1.2
torch==1.9.0
tqdm==4.62.3
```


## Example Commands to Run the Experiments
1. Learning single MNL from partial rankings on synthetic data
```bash
python3 dag_synthetic.py --num_classes 100 --num_samples 5000  # single MNL
```
2. Learning mixture of MNL from partial rankings on synthetic data
```bash
python3 dag_synthetic.py --num_classes 60 --num_samples 5000 --alphas [1,1,1]  --init_by_cluster # 3 MNLs with clustering based init
```
3. Network formation modeling of synthetic network data
```bash
python3 network_synthetic.py -r 0.5 -p 0.5 --fof --ua --pa --loss topk  # run full model with 4 components on a mixed (r,p)-graph
```
4. Network formation modeling of Flickr & Microsoft Academic Graph
```bash
cd source
wget -4 http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz ../data/
python3 flickr_process.py # process flickr-growth.txt.gz, which is downloaded from http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz
python3 flickr_train.py
```
```bash
# download mag_cli.csv by google drive
python3 mag_process.py  # process mag_cli.csv, which is downloaded from https://drive.google.com/file/d/17bgLs1iR96JW3Rd0mex3IK8qyU-qRElB/view?usp=sharing
python3 mag_train.py
```

## Cite
```
@article{ma2022fast,
  title={Fast Learning of MNL Model From General Partial Rankings with Application to Network Formation Modeling},
  author={Ma, Jiaqi and Zhang, Xingjian and Mei, Qiaozhu},
  journal={Proceedings of the 15th ACM International Conference on Web Search and Data Mining},
  year={2022}
}
```
