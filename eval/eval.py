import sys
from os import path
sys.path.append(path.abspath("../../framework"))
from datasets import load_from_disk, load_dataset
import torch
import numpy as np
import argparse
import random
from metrics.rouge import evalROUGE
from metrics.bertscore import evalBERTScore
from metrics.logger import MetricsLogger
from modules.Document import Document
from modules.Cluster import Cluster
from modules.summarizer.ExtractiveSummarizer import ExtractiveSummarizer
from umap import UMAP
from sklearn.cluster import OPTICS
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import gensim.downloader
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import requests
import xml.etree.ElementTree as ET
import time

import warnings
warnings.simplefilter("ignore")



def _efetch(pmids):
    for i in range(3):
        try:
            res = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", data={
                "db": "pubmed",
                "retmode": "xml",
                "rettype": "abstract",
                "id": ",".join(pmids)
            })
            return res.content
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError):
            time.sleep(3**(i+1))
            continue
    raise RuntimeError("Cannot retrieve articles")

def _getKeywords(pmids):
    out_keywords = []

    query_xml = _efetch(pmids)
    query_tree = ET.fromstring(query_xml)
    for article_tree in query_tree:
        keywords = []

        # MeSH headings
        mesh_headings_xml = article_tree.find("MedlineCitation/MeshHeadingList")
        if mesh_headings_xml is not None: 
            for mesh_xml in mesh_headings_xml.findall("MeshHeading"):
                keywords.append(mesh_xml.find("DescriptorName").text) # type: ignore
        # Keywords
        keywords_xml = article_tree.find("MedlineCitation/KeywordList")
        if keywords_xml is not None: 
            for keyword_xml in keywords_xml.findall("Keyword"):
                keywords.append(keyword_xml.text)
                
        out_keywords.append(keywords)

    return out_keywords



class Framework():
    def __init__(self, embedding, clustering_criteria, umap_comps=10, optics_min_samples=2):
        self.embedding = embedding
        self.dim_reduction = UMAP(n_components=umap_comps, random_state=42)
        self.clustering = OPTICS(min_samples=optics_min_samples)
        self.summarizer = ExtractiveSummarizer(
            summ_pipeline=pipeline("summarization",
                model = "NotXia/longformer-bio-ext-summ",
                tokenizer = AutoTokenizer.from_pretrained("NotXia/longformer-bio-ext-summ"),
                trust_remote_code = True,
                device = 0 if torch.cuda.is_available() else -1
            ) # type: ignore
        )
        self.clustering_criteria = clustering_criteria

    def _cluster(self, clusters: list[Cluster]):
        embedded_docs = []
        out_clusters: list[Cluster] = []

        for cluster in clusters:
            abstracts = [d.abstract for d in cluster.docs]
            keywords = [" ".join(d.keywords) for d in cluster.docs] # type: ignore

            if self.clustering_criteria == "abstract":
                embedded_docs = self.embedding(abstracts)
            elif self.clustering_criteria == "keywords":
                embedded_docs = self.embedding(keywords)
            embedded_docs = self.dim_reduction.fit_transform(embedded_docs)

            labels = self.clustering.fit_predict(embedded_docs) # type: ignore

            for l in set(labels):
                # if l == -1: continue

                out_clusters.append(Cluster(
                    docs = [Document(title="", abstract=abstract) for i, abstract in enumerate(abstracts) if labels[i] == l]
                ))

        return out_clusters

    def __call__(self, abstracts: list[str], keywords: list[list[str]], summary_size: int):
        assert len(abstracts) == len(keywords)
        clusters = [
            Cluster(docs=[
                Document(title="", abstract=abstracts[i], keywords=keywords[i]) for i in range(len(abstracts))
            ])
        ]
        clusters = self._cluster(clusters)
        clusters, final_summary_sents = self.summarizer(
            clusters, 
            overall_summary = True, 
            cluster_summary_len = summary_size, 
            overall_summary_len = summary_size
        )
        return final_summary_sents
    


def gensimEmbed(embedding_model, documents: list[str]):
    embed_shape = embedding_model.get_vector(list(embedding_model.index_to_key)[0]).shape[0] # type: ignore
    zero_vector = np.zeros(embed_shape)

    # Applies embedding and mean pooling
    embeddings = []
    for doc in documents:
        embedding = [embedding_model.get_vector(word) for word in doc.split() if word in embedding_model.key_to_index] # type: ignore
        embeddings.append(np.mean(embedding, axis=0) if len(embedding) > 0 else zero_vector)

    embeddings = np.array(embeddings)
    return embeddings


"""
    Framework evaluation.

    Parameters
    ----------
        model : Framework|str

        original_dataset : DatasetDict

        extractive_dataset : DatasetDict

        splits : str[]
            Splits of the dataset to use

        threshold : int
            Entries with less than this number of articles are skipped.

        summary_size : int

        clustering_criteria : str
            Corpus to use for clustering
"""
def evaluate(model, original_dataset, extractive_dataset, splits, threshold, summary_size, clustering_criteria):
    metrics = MetricsLogger()

    if type(model) == str and model == "plain":
        summarizer = pipeline("summarization",
            model = "NotXia/longformer-bio-ext-summ",
            tokenizer = AutoTokenizer.from_pretrained("NotXia/longformer-bio-ext-summ"),
            trust_remote_code = True,
            device = 0 if torch.cuda.is_available() else -1
        ) 

    for split in splits:
        # Indexes to display current status
        curr_processed = 0
        threshold_entries = 0
        for entry in original_dataset[split]: 
            if len(entry["abstract"]) >= threshold: threshold_entries += 1

        for i in range(len(original_dataset[split])):
            docs = original_dataset[split][i]["abstract"]
            pmids = original_dataset[split][i]["pmid"]
            ref_summary = extractive_dataset[split][i]["ref_summary"]
            summary = ""

            if len(docs) < threshold: continue

            if type(model) == Framework:
                if clustering_criteria == "keywords":
                    keywords = _getKeywords(pmids)
                else:
                    keywords = [[]] * len(docs)
                selected_sents = model(docs, keywords, summary_size=summary_size)
            elif model == "plain":
                selected_sents, _ = summarizer({ "sentences": extractive_dataset[split][i]["sentences"] }, strategy="count", strategy_args=summary_size) # type: ignore
            elif model == "oracle":
                selected_sents = [sent for j, sent in enumerate(extractive_dataset[split][i]["sentences"]) if extractive_dataset[split][i]["labels"][j]]
            summary = "\n".join(selected_sents) # type: ignore

            metrics.add("rouge", evalROUGE( [ref_summary], [summary] ))
            metrics.add("bertscore", evalBERTScore( [ref_summary], [summary] ))
            sys.stdout.write(f"\r{curr_processed+1}/{threshold_entries} ({split}) --- {metrics.format(['rouge', 'bertscore'])}\033[K")
            sys.stdout.flush()
            curr_processed += 1

    sys.stdout.write("\r\033[K")
    print(f"{metrics.format(['rouge', 'bertscore'])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Framework evaluation")
    parser.add_argument("--dataset", type=str, choices=["ms2", "cochrane"], required=True, help="Evaluation dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path the preprocessed extractive dataset")
    parser.add_argument("--splits", type=str, default="test", required=True, help="Splits of the dataset to use for the evaluation (e.g. test,validation)")
    parser.add_argument("--docs-threshold", type=int, default=15, help="Exclude entries with less than this number of articles")
    model_args = parser.add_mutually_exclusive_group(required=True)
    model_args.add_argument("--embedding", type=str, choices=[
        "bow", "word2vec", "glove", "fasttext", "biowordvec", "minilm", "biobert", "pubmedbert"
    ], help="Embedding to use for clustering")
    model_args.add_argument("--plain", action="store_true", help="Evaluate using document chunking without clustering")
    model_args.add_argument("--oracle", action="store_true", help="Evaluate oracle")
    parser.add_argument("--embedding-path", type=str, required=False, help="Path to the local embedding model (for BioWordVec)")
    parser.add_argument("--clustering-criteria", choices=["abstract", "keywords"], default="abstract", help="Corpus to use for clustering")
    parser.add_argument("--umap-components", type=int, default=10, help="UMAP number of components")
    parser.add_argument("--optics-min-samples", type=int, default=2, help="OPTICS minimum number of neighbors")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.dataset == "cochrane":
        original_dataset = load_dataset("allenai/mslr2022", "cochrane")
        extractive_dataset = load_from_disk(args.dataset_path)
        summary_size = 4
    elif args.dataset == "ms2":
        original_dataset = load_dataset("allenai/mslr2022", "ms2")
        extractive_dataset = load_from_disk(args.dataset_path)
        summary_size = 4
    else:
        raise ValueError("Unknown dataset")
    
    if args.embedding is not None:
        if args.embedding == "bow":
            embedding_model = TfidfVectorizer(max_df=0.5, min_df=0.05, stop_words="english", ngram_range=(1, 3))
            embedding_fn = lambda docs: embedding_model.fit_transform(docs).toarray() # type: ignore
        elif args.embedding == "word2vec":
            embedding_model = gensim.downloader.load("word2vec-google-news-300")
            embedding_fn = lambda docs: gensimEmbed(embedding_model, docs)
        elif args.embedding == "glove":
            embedding_model = gensim.downloader.load("glove-wiki-gigaword-300")
            embedding_fn = lambda docs: gensimEmbed(embedding_model, docs)
        elif args.embedding == "fasttext":
            embedding_model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
            embedding_fn = lambda docs: gensimEmbed(embedding_model, docs)
        elif args.embedding == "biowordvec":
            if args.embedding_path is None: raise ValueError("Missing model path")
            embedding_model = KeyedVectors.load_word2vec_format(args.embedding_path, binary=True, limit=8000000)
            embedding_fn = lambda docs: gensimEmbed(embedding_model, docs)
        elif args.embedding == "minilm":
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_fn = lambda docs: embedding_model.encode(docs)
        elif args.embedding == "biobert":
            embedding_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
            embedding_fn = lambda docs: embedding_model.encode(docs)
        elif args.embedding == "pubmedbert":
            embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
            embedding_fn = lambda docs: embedding_model.encode(docs)
        else:
            raise ValueError("Unknown embedding")
        model = Framework(
            embedding = embedding_fn, 
            clustering_criteria = args.clustering_criteria,
            umap_comps = args.umap_components,
            optics_min_samples = args.optics_min_samples
        ) # type: ignore
    elif args.plain:
        model = "plain"
    elif args.oracle:
        model = "oracle"
    else:
        raise ValueError("Unknown model")
    

    print("-- Starting evaluation --")
    evaluate(
        model = model,
        original_dataset = original_dataset,
        extractive_dataset = extractive_dataset,
        splits = args.splits.split(","),
        threshold = args.docs_threshold,
        summary_size = summary_size,
        clustering_criteria = args.clustering_criteria
    )
