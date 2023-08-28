import { Article } from "~/types/Article"
import { Cluster } from "~/types/Cluster"


export function castToArticle(article_data: any) : Article {
    return ({
        title: article_data.title,
        abstract: article_data.abstract,
        date: article_data.date !== null ? new Date(article_data.date*1000) : null,
        authors: article_data.authors,
        doi: article_data.doi, 
        pmid: article_data.pmid,
        summary: article_data.summary
    }) as Article
}

export function castToCluster(cluster_data: any) : Cluster {
    return ({
        topics: cluster_data.topics,
        docs: cluster_data.docs.map((article: any) => castToArticle(article)),
        is_outlier: cluster_data.is_outlier,
        summary: cluster_data.summary,
        metadata: cluster_data.metadata,
    }) as Cluster
}