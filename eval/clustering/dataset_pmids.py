"""

Creates the metadata to generate a dataset for clustering evaluation.
Each entry of the dataset is composed of a series of papers obtained from the same query.
The goal is to cluster each entry and find more fine-grained clusters.
This script only extracts the PMIDs of the articles.

"""

import argparse
import requests
import json
import time


def getPmids(query:str, min_date:str, max_date:str, amount=5000):
    def _esearch(query, retmax, retstart=0, min_date=None, max_date=None):
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": retmax,
            "retstart": retstart,
            "term": query
        }
        if min_date is not None: params["mindate"] = min_date
        if max_date is not None: params["maxdate"] = max_date

        for i in range(3):
            try:
                res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
                return res.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError):
                time.sleep(2**(i+1))
                continue
        raise RuntimeError("Cannot retrieve PMIDs")
    
    res: dict = _esearch(query, retstart=0, retmax=amount, min_date=min_date, max_date=max_date)
    doc_pmids: list[str] = res["esearchresult"]["idlist"]

    return doc_pmids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Creates dataset metadata")
    parser.add_argument("--output", type=str, default="./pmids.json", required=False, help="Path for the output file")
    args = parser.parse_args()

    # Diseases selected from 
    # https://www.nhsinform.scot/illnesses-and-conditions/a-to-z and
    # https://rarediseases.org/rare-diseases/
    dataset_queries = [
        "asthma", 
        "attention deficit hyperactivity disorder", 
        "autistic spectrum disorder", 
        "brain tumours", 
        "bronchitis", 
        "common cold", 
        "covid-19", 
        "dementia", 
        "depression", 
        "gilbert syndrome", 
        "heart failure", 
        "hepatitis B ", 
        "hiv", 
        "kidney cancer", 
        "meningitis", 
        "migraine", 
        "multiple sclerosis", 
        "radiation sickness", 
        "type 2 diabetes", 
        "yellow fever"
    ]


    dataset_pmids: list[dict] = []

    # Retrieves PMIDs
    for query in dataset_queries:
        pmids = getPmids(query, "2022", "2022")
        dataset_pmids.append({
            "query": query,
            "pmids": pmids
        })
        print(f"{query} -- {len(pmids)} pmids")
        time.sleep(2)


    with open(args.output, "w", encoding="utf-8") as fout:
        json.dump(dataset_pmids, fout, ensure_ascii=False, indent=None)