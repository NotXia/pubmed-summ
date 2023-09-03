<template>
    <div class="flex justify-center items-center">
        <SearchBar :onquery="querySubmitted" class="w-96" />
    </div>

    <div class="mx-36 mt-10">
        <ul class="list-disc list-inside" v-if="overall_summary.length > 0">
            <li v-for="sent in overall_summary">{{ sent }}</li>
        </ul>
    </div>

    <div v-if="request_id != ''" class="mt-4 mx-20" :key="request_id">
        <div v-if="clusters.length > 0" v-for="(cluster, i) in clusters" :key="cluster.topics.join('-')">
            <ClusterContainer class="w-3/4 mt-12 mx-auto" :cluster="cluster" />
        </div>
        
        <Loading v-else class="flex justify-center"/>
    </div>
</template>

<script setup lang="ts">
    import { io } from "socket.io-client"
    import { Cluster } from "~/types/Cluster"
    const config = useRuntimeConfig()

    const clusters = ref<Cluster[]>([])
    const request_id = ref("")
    const overall_summary = ref<string[]>([])

    
    function querySubmitted(query: String) {
        const socket = io(config.public.BACKEND_SOCKET, { path: config.public.BACKEND_SOCKETIO_PATH })

        socket.on("connect", () => {

            socket.emit("query", query)

            socket.on("on_query_received", (data) => {
                request_id.value = data.id
                clusters.value = []
            })

            socket.on("on_clusters_created", (created_clusters) => {
                clusters.value = created_clusters.map((cluster: any) => castToCluster(cluster))
            })

            socket.on("on_document_ready", (ready_article: any) => {
                const curr_clusters = clusters.value.slice();

                // Seaches and updates the article
                for (let i=0; i<curr_clusters.length; i++) {
                    for (let j=0; j<curr_clusters[i].docs.length; j++) {
                        if (curr_clusters[i].docs[j].pmid === ready_article.pmid) {
                            curr_clusters[i].docs[j] = castToArticle(ready_article)
                            clusters.value = curr_clusters
                            return
                        }
                    }
                }
            })

            socket.on("on_cluster_ready", (ready_cluster: Cluster) => {
                const curr_clusters = clusters.value.slice()

                function isSameCluster(c1: Cluster, c2: Cluster) : boolean {
                    if (c1.topics.length !== c2.topics.length) { return false }

                    for (let j=0; j<c1.topics.length; j++) {
                        if (c1.topics[j] !== c2.topics[j]) {
                            return false
                        }
                    }
                    return true
                }

                // Seaches and updates the cluster
                for (let i=0; i<curr_clusters.length; i++) {
                    if (isSameCluster(curr_clusters[i], ready_cluster)) {
                        curr_clusters[i] = castToCluster(ready_cluster)
                        break
                    }
                }

                clusters.value = curr_clusters
            })

            socket.on("on_overall_summary", (sentences: string[]) => {
                overall_summary.value = sentences
            })

        })
    }
</script>