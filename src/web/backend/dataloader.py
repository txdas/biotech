from backend import constant, text
import csv
import jsonlines


def load_prompts():
    with open(constant.PROMPTS) as fp:
        reader = csv.DictReader(fp)
        for i, v in enumerate(reader):
            yield v["type"].strip(), v["content"].strip()


def load_human(limit=10):
    with open(constant.HUMAN) as fp:
        reader = csv.DictReader(fp)
        for i, v in enumerate(reader):
            v["target"] = bool(int(v["target"]))
            if i>=limit:
                break
            yield v


def load_uniprot(limit=20):
    lst, success, fail = [], 0, 0
    with open(constant.UNIPROT) as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for v in reader:
            organism = text.rnormalize(v["Organism"])
            protein = text.rnormalize(v["Protein names"])
            entry = v["Entry Name"]
            gos = v["Gene Ontology (GO)"]
            if "0046718" in gos:
                v = {"organism": organism, "protein":protein, "entry": entry, "target": True}
                success += 1
                if success <= limit:
                    lst.append(v)
            if "host cell" not in gos:
                v = {"organism": organism, "protein": protein, "entry":entry, "target": False}
                fail += 1
                if fail <= limit:
                    lst.append(v)
    return lst


def load_meta(fn, limit=500):
    with jsonlines.open(fn) as fp:
        for i, v in enumerate(fp):
            if i>limit:
                break
            else:
                yield v


def load(limit=10):
    total = 0
    with jsonlines.open(constant.PROTEINS) as fp:
        for v in fp:
            virus = text.normalize_virus(v["virus"])
            proteins = [text.normalize_protein(v) for v in v["proteins"]]
            for protein in proteins:
                if protein and "dna" not in protein.lower() and "rna" not in protein.lower():
                    total += 1
                    if total>limit:
                        break
                    yield {"organism": virus, "protein": protein, "target": True}


if __name__ == '__main__':
    # for v in load():
    # for v in load_meta(constant.RAW_INFO):
    #     print(v)
    load_prompts()