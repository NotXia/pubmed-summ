import argparse
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from umap import UMAP
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, DBSCAN, OPTICS
from hdbscan import HDBSCAN
from sklearn import metrics


# 
# Embedding
#

def tfidfEmbedding(ngram_range):
    def embed(documents: list[str]):
        model = TfidfVectorizer(max_df=0.5, min_df=0.05, stop_words="english", ngram_range=ngram_range)
        embedding = model.fit_transform(documents).toarray() # type: ignore
        del model
        return embedding
    return embed

def neuralEmbedding(model_type, path=None):
    if model_type == "word2vec":
        model = gensim.downloader.load("word2vec-google-news-300")
    elif model_type == "fasttext":
        model = gensim.downloader.load("fasttext-wiki-news-subwords-300")
    elif model_type == "biowordvec":
        if path is None: raise ValueError("Missing model path")
        model = KeyedVectors.load_word2vec_format(path, binary=True, limit=8000000)

    # Based on the Gensim module of BERTopic
    # Source: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/backend/_gensim.py
    def embed(documents: list[str]):
        embed_shape = model.get_vector(list(model.index_to_key)[0]).shape[0] # type: ignore
        zero_vector = np.zeros(embed_shape)

        # Applies embedding and mean pooling
        embeddings = []
        for doc in documents:
            embedding = [model.get_vector(word) for word in doc.split() if word in model.key_to_index] # type: ignore
            embeddings.append(np.mean(embedding, axis=0) if len(embedding) > 0 else zero_vector)

        embeddings = np.array(embeddings)
        return embeddings
    return embed

def contextualEmbedding(model_name):
    model = SentenceTransformer(model_name)
    def embed(documents: list[str]):
        embedding = model.encode(documents) # type: ignore
        return embedding
    return embed


# 
# Dimensionality reduction
#

def noReduction():
    def reduce(embedded_docs):
        return embedded_docs
    return reduce

def pcaReduction(n_components):
    def reduce(embedded_docs):
        model = PCA(n_components=n_components)
        embedding = model.fit_transform(embedded_docs)
        del model
        return embedding
    return reduce

def lsaReduction(n_components):
    def reduce(embedded_docs):
        model = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
        embedding = model.fit_transform(embedded_docs)
        del model
        return embedding
    return reduce

def nmfReduction(n_components):
    def reduce(embedded_docs):
        model = NMF(n_components=n_components, init='random', random_state=0)
        embedding = model.fit_transform(embedded_docs)
        del model
        return embedding
    return reduce

def umapReduction(n_components):
    def reduce(embedded_docs):
        model = UMAP(n_components=n_components, random_state=42)
        embedding = model.fit_transform(embedded_docs)
        del model
        return embedding
    return reduce


# 
# Clustering
#

def kmeansClustering(n_clusters):
    def cluster(embedded_docs):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster

def bisectingKmeansClustering(n_clusters):
    def cluster(embedded_docs):
        model = BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster

def agglomerativeClustering(n_clusters=None, distance_threshold=None):
    def cluster(embedded_docs):
        if distance_threshold is not None:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None)
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster

def dbscanClustering(eps, min_samples):
    def cluster(embedded_docs):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster

def opticsClustering(min_samples):
    def cluster(embedded_docs):
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster

def hdbscanClustering(min_cluster_size):
    def cluster(embedded_docs):
        model = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(embedded_docs)
        del model
        return labels
    return cluster



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model evaluation")
    parser.add_argument("--dataset", type=str, default="./dataset.json", required=False, help="Path for the dataset")
    parser.add_argument("--embedding", type=str, choices=[
        "bow", "word2vec", "fasttext", "biowordvec", "minilm", "biobert", "pubmedbert"
        ], required=True, help="Embedding to evaluate")
    parser.add_argument("--model-path", type=str, required=False, help="Path to the model to load (neeeded for biowordvec)")
    parser.add_argument("--save-raw", type=str, required=False, help="Path to the location to save the raw scores")
    args = parser.parse_args()


    to_try_embeddings = []
    if args.embedding == "bow":
        to_try_embeddings.append( ("tdidf[1,2,3-gram]", tfidfEmbedding(ngram_range=(1,3))) )
    elif args.embedding == "word2vec":
        to_try_embeddings.append( ("word2vec", neuralEmbedding("word2vec")) )
    elif args.embedding == "fasttext":
        to_try_embeddings.append( ("fasttext", neuralEmbedding("fasttext")) )
    elif args.embedding == "biowordvec":
        to_try_embeddings.append( ("biowordvec", neuralEmbedding("biowordvec", path=args.model_path)) )
    elif args.embedding == "minilm":
        to_try_embeddings.append( ("minilm", contextualEmbedding("all-MiniLM-L12-v2")) )
    elif args.embedding == "biobert":
        to_try_embeddings.append( ("biobert", contextualEmbedding("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")) )
    elif args.embedding == "pubmedbert":
        to_try_embeddings.append( ("pubmedbert", contextualEmbedding("pritamdeka/S-PubMedBert-MS-MARCO")) )
    else:
        raise RuntimeError("Invalid embedding")

    to_try_reductions = [
        ("no-reduction", noReduction()),
        ("pca[n_comp=25]", pcaReduction(n_components=25)),
        ("lsa[n_comp=25]", lsaReduction(n_components=25)),
        ("nmf[n_comp=25]", nmfReduction(n_components=25)),
        ("umap[n_comp=25]", umapReduction(n_components=25))
    ]

    to_try_clustering = [
        ("kmeans[n_cl=5]", kmeansClustering(n_clusters=5)),
        ("kmeans[n_cl=10]", kmeansClustering(n_clusters=10)),
        ("kmeans[n_cl=20]", kmeansClustering(n_clusters=20)),

        ("biskmeans[n_cl=5]", bisectingKmeansClustering(n_clusters=5)),
        ("biskmeans[n_cl=10]", bisectingKmeansClustering(n_clusters=10)),
        ("biskmeans[n_cl=20]", bisectingKmeansClustering(n_clusters=20)),

        ("agglo[thresh=5]", agglomerativeClustering(distance_threshold=1)),
        ("agglo[thresh=5]", agglomerativeClustering(distance_threshold=5)),
        ("agglo[thresh=10]", agglomerativeClustering(distance_threshold=15)),

        ("dbscan[eps=0.1,min_neigh=5]", dbscanClustering(eps=0.1, min_samples=5)),
        ("dbscan[eps=0.5,min_neigh=5]", dbscanClustering(eps=0.5, min_samples=5)),
        ("dbscan[eps=1,min_neigh=5]", dbscanClustering(eps=1, min_samples=5)),
        ("dbscan[eps=0.1,min_neigh=10]", dbscanClustering(eps=0.1, min_samples=10)),
        ("dbscan[eps=0.5,min_neigh=10]", dbscanClustering(eps=0.5, min_samples=10)),
        ("dbscan[eps=1,min_neigh=10]", dbscanClustering(eps=1, min_samples=10)),
        ("dbscan[eps=0.1,min_neigh=20]", dbscanClustering(eps=0.1, min_samples=20)),
        ("dbscan[eps=0.5,min_neigh=20]", dbscanClustering(eps=0.5, min_samples=20)),
        ("dbscan[eps=1,min_neigh=20]", dbscanClustering(eps=1, min_samples=20)),

        ("optics[min_neigh=5]", opticsClustering(min_samples=5)),
        ("optics[min_neigh=10]", opticsClustering(min_samples=10)),
        ("optics[min_neigh=20]", opticsClustering(min_samples=20)),

        ("hdbscan[min=5]", hdbscanClustering(min_cluster_size=5)),
        ("hdbscan[min=10]", hdbscanClustering(min_cluster_size=10)),
        ("hdbscan[min=20]", hdbscanClustering(min_cluster_size=20)),
    ]


    with open("./dataset.json", "r") as f:
        dataset = json.load(f)

    scores = {}
    failed_configs = {}

    num_runs = len(dataset) * len(to_try_embeddings) * len(to_try_reductions) * len(to_try_clustering)
    run_i = 1

    for entry_i, entry in enumerate(dataset):
        documents = entry["abstracts"]

        for embedder_label, embedder in to_try_embeddings:
            try:
                embedding = embedder(documents)
                # eval_embedding = tfidfEmbedding(ngram_range=(1,3))(documents)
                # eval_embedding = embedding
            except Exception as e:
                failed_configs.setdefault(f"{embedder_label}", set()).add(f"{e}")
                continue

            for reducer_label, reducer in to_try_reductions:
                try:
                    embed_reduced = reducer(embedding)
                    eval_embedding = embed_reduced
                except Exception as e:
                    failed_configs.setdefault(f"{embedder_label} {reducer_label}", set()).add(f"{e}")
                    continue
                
                for clusterer_label, clusterer in to_try_clustering:
                    conf_id = f"{embedder_label} {reducer_label} {clusterer_label}"

                    # Skip this configuration as it failed before
                    if conf_id in failed_configs: continue 

                    sys.stdout.write(f"\r{run_i}/{num_runs} {conf_id} \033[K")
                    sys.stdout.flush()
                    run_i += 1
                    
                    try:
                        labels = clusterer(embed_reduced)
                        # Removes outliers
                        filtered_embeds = [emb for i, emb in enumerate(eval_embedding) if labels[i] != -1]
                        labels = [i for i in labels if i != -1]
                        if len(labels) == 0: 
                            raise ValueError("All outliers")

                        if conf_id not in scores: scores[conf_id] = { "sc": [], "ch": [], "db": [] }
                        scores[conf_id]["sc"].append(metrics.silhouette_score(filtered_embeds, labels)) # type: ignore
                        scores[conf_id]["ch"].append(metrics.calinski_harabasz_score(filtered_embeds, labels)) # type: ignore
                        scores[conf_id]["db"].append(metrics.davies_bouldin_score(filtered_embeds, labels)) # type: ignore
                    except Exception as e:
                        failed_configs.setdefault(conf_id, set()).add(f"{e}")

                        # Removes score entry
                        if conf_id in scores: del scores[conf_id]
                        
                # End clustering loop
            # End reduction loop
        # End embedding loop


    sys.stdout.write("\r\033[K")
    print()

    print("\n--- Failed runs ---")
    for conf in failed_configs:
        print(conf, failed_configs[conf])
    print()

    print("\n--- Results ---")
    label_padding = max([len(conf) for conf in scores]) + 1 # For a nicer print
    for conf in scores:
        sc_mean = np.mean(scores[conf]["sc"])
        sc_mean_dev = np.std(scores[conf]["sc"])
        ch_mean = np.mean(scores[conf]["ch"])
        ch_mean_dev = np.std(scores[conf]["ch"])
        db_mean = np.mean(scores[conf]["db"])
        db_mean_dev = np.std(scores[conf]["db"])
        print(
            f"{conf.ljust(label_padding)} | "
            f"SC: {format(round(sc_mean, 4), '.4f').rjust(10)} +/- {format(round(sc_mean_dev, 4), '.4f').ljust(8)} ({format(round(abs(sc_mean_dev/sc_mean), 4), '.4f').ljust(7)}) | "
            f"CH: {format(round(ch_mean, 4), '.4f').rjust(10)} +/- {format(round(ch_mean_dev, 4), '.4f').ljust(8)} ({format(round(abs(ch_mean_dev/ch_mean), 4), '.4f').ljust(7)}) | "
            f"DB: {format(round(db_mean, 4), '.4f').rjust(10)} +/- {format(round(db_mean_dev, 4), '.4f').ljust(8)} ({format(round(abs(db_mean_dev/db_mean), 4), '.4f').ljust(7)})"
        )


    if args.save_raw is not None:
        with open(args.save_raw, "w", encoding="utf-8") as fout:
            json.dump(scores, fout, indent=4)