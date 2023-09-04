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
import itertools
from sentence_transformers import SentenceTransformer
import gensim.downloader
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.simplefilter("ignore")


class Framework():
    def __init__(self, embedding):
        self.embedding = embedding
        self.dim_reduction = UMAP(n_components=8, random_state=42)
        self.clustering = OPTICS(min_samples=2)
        self.summarizer = ExtractiveSummarizer(
            summ_pipeline=pipeline("summarization",
                model = "NotXia/longformer-bio-ext-summ",
                tokenizer = AutoTokenizer.from_pretrained("NotXia/longformer-bio-ext-summ"),
                trust_remote_code = True,
                device = 0 if torch.cuda.is_available() else -1
            ) # type: ignore
        )


    def _cluster(self, clusters: list[Cluster]):
        embedded_docs = []
        out_clusters: list[Cluster] = []

        for cluster in clusters:
            abstracts = [d.abstract for d in cluster.docs]
            embedded_docs = self.embedding(abstracts)
            embedded_docs = self.dim_reduction.fit_transform(embedded_docs)

            labels = self.clustering.fit_predict(embedded_docs) # type: ignore

            for l in set(labels):
                # if l == -1: continue

                out_clusters.append(Cluster(
                    docs = [Document(title="", abstract=abstract) for i, abstract in enumerate(abstracts) if labels[i] == l]
                ))

        return out_clusters

    def __call__(self, abstracts: list[str], summary_size: int):
        clusters = [
            Cluster(docs=[Document(title="", abstract=abstract) for abstract in abstracts])
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
"""
def evaluate(model, original_dataset, extractive_dataset, splits, threshold, summary_size):
    metrics = MetricsLogger()

    if type(model) == str and model == "plain":
        summarizer = pipeline("summarization",
            model = "NotXia/longformer-bio-ext-summ",
            tokenizer = AutoTokenizer.from_pretrained("NotXia/longformer-bio-ext-summ"),
            trust_remote_code = True,
            device = 0 if torch.cuda.is_available() else -1
        ) 

    for split in splits:
        for i in range(len(original_dataset[split])):
            docs = original_dataset[split][i]["abstract"]
            ref_summary = extractive_dataset[split][i]["ref_summary"]
            summary = ""

            if len(docs) < threshold: continue

            if type(model) == Framework:
                selected_sents = model(docs, summary_size=summary_size)
            elif model == "plain":
                selected_sents, _ = summarizer({ "sentences": extractive_dataset[split][i]["sentences"] }, strategy="count", strategy_args=summary_size) # type: ignore
            elif model == "oracle":
                selected_sents = [sent for j, sent in enumerate(extractive_dataset[split][i]["sentences"]) if extractive_dataset[split][i]["labels"][j]]
            summary = "\n".join(selected_sents) # type: ignore

            metrics.add("rouge", evalROUGE( [ref_summary], [summary] ))
            metrics.add("bertscore", evalBERTScore( [ref_summary], [summary] ))
            sys.stdout.write(f"\r{i+1}/{len(original_dataset[split])} ({split}) --- {metrics.format(['rouge', 'bertscore'])}\033[K")
            sys.stdout.flush()

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
        model = Framework(embedding=embedding_fn) # type: ignore
    elif args.plain:
        print("Plain")
        model = "plain"
    elif args.oracle:
        print("Oracle")
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
        summary_size = summary_size
    )
