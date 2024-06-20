import os
import json
import jsonlines
from Bio import SeqIO
from io import StringIO
from nio.uniprot import UniprotClient
from nio.niofold import AlphaFold
from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings("ignore")


def find(fpath="/Volumes/Extreme SSD/data/deaminase"):
    data = []
    for root, dirs, files in os.walk(fpath):
        for f in files:
            if not f.startswith(".") and not f.startswith("index"):
                fname = os.path.join(root, f)
                print(fname)
                lst = json.load(open(fname))
                ret = []
                for v in lst:
                    meta, entries = v["metadata"], v["entries"]
                    entry = [e for e in entries if e["entry_type"] == "family"][0]
                    obj = {
                        "accession": meta["accession"],
                        "name": meta["name"],
                        "length": meta["length"],
                        "gene": meta["gene"],
                        "in_alphafold": meta["in_alphafold"],
                        "interpro": entry["accession"]}
                    ret.append(obj)
                ret = [v for v in ret if v["length"] >= 100 and v["in_alphafold"]][:25]
                data.extend(ret)
    with jsonlines.open(os.path.join(fpath, "index.json"), "w") as fp:
        for v in data:
            fp.write(v)


def get_fasta(fpath="/Volumes/Extreme SSD/data/deaminase"):
    from tqdm import tqdm
    client = UniprotClient()
    records = []
    with jsonlines.open(os.path.join(fpath, "index.json")) as fp:
        for v in tqdm(fp):
            ret = client.get(v["accession"], format="fasta")
            if ret:
                rcds = SeqIO.parse(StringIO(ret), "fasta")
                for r in rcds:
                    records.append(r)
    SeqIO.write(records, os.path.join(fpath, "index.fasta"),"fasta")


def get_alphafold(fpath="/Volumes/Extreme SSD/data/deaminase"):

    client = AlphaFold()
    records = []
    with jsonlines.open(os.path.join(fpath, "index.json")) as fp:
        for v in tqdm(fp):
            ret = client.get(v["accession"])
            if ret:
                records.append(ret)
    with jsonlines.open(os.path.join(fpath, "index_af.json"), "w") as fp:
        for r in records:
            fp.write(r)


def get_pdb(fpath="/Volumes/Extreme SSD/data/deaminase"):
    fpdbs = os.path.join(fpath,"pdbs")
    if not os.path.exists(fpdbs):
        os.makedirs(fpdbs, exist_ok=True)
    with jsonlines.open(os.path.join(fpath, "index_af.json")) as fp:
        for v in tqdm(fp):
            fname = os.path.join(fpdbs,v["uniprotAccession"]+".pdb")
            if not os.path.exists(fname):
                response = requests.get(v["pdbUrl"])
                with open(fname, "wb") as wp:
                    wp.write(response.content)


if __name__ == '__main__':
    # find()
    # get_fasta()
    # get_alphafold()
    get_pdb()
