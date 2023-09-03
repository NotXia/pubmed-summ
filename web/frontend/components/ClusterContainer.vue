<template>

<div>

    <h2 class="text-xl font-bold" >
        <div v-if="!cluster.is_outlier">
            <span class="text-base">Topics</span><br/>
            {{ cluster.topics.join(" &nbsp;&#183;&nbsp; ").toUpperCase() }}
        </div>
        <div v-else>
            Outliers
        </div>
    </h2>

    <div>
        <h3 class="text-base font-bold mt-3">Summary</h3>
        <ul class="list-disc list-inside" v-if="cluster.summary?.length > 0">
            <li v-for="sent in cluster.summary">{{ sent }}</li>
        </ul>
        <Loading v-else class="inline-block"/>
    </div>

    <div data-accordion="open" class="mt-3">
        <div class="border-t-2 last:border-b-2 border-slate-300 dark:border-slate-600">
            <button type="button" :id="`accordion-cluster-${cluster_id}-heading`"
            class="flex items-center w-full px-5 py-3 font-medium text-left
                hover:bg-gray-100 dark:hover:bg-gray-800" 
            :data-accordion-target="`#accordion-cluster-${cluster_id}-body`" 
            aria-expanded="false" :aria-controls="`accordion-cluster-${cluster_id}-body`">
                <svg data-accordion-icon class="w-3 h-3 shrink-0 mr-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
                    <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5 5 1 1 5"/>
                </svg>
                <span>Articles</span>
            </button>
            
            <div :id="`accordion-cluster-${cluster_id}-body`" :aria-labelledby="`accordion-cluster-${cluster_id}-heading`" class="hidden">
                <ArticleContainer v-for="article in cluster.docs" :key="article.pmid" :article="article" 
                    class="first:mt-8 mb-8 mx-10 border rounded-sm p-5" />
            </div>
        </div>
    </div>

</div>

</template>


<script setup lang="ts">
    import { initAccordions } from "flowbite";
    import { Cluster } from "~/types/Cluster";
    
    const props = defineProps({
        cluster: {
            type: Object as PropType<Cluster>,
            required: true,
        },
    });
    const cluster_id = ref(props.cluster.topics.join('-').replace(/\s/g, ''))

    onMounted(() => {
        initAccordions();
    })

    watch(() => props.cluster, (clusters, old_clusters) => {
        nextTick(() => {
            initAccordions()
        })
    })
</script>