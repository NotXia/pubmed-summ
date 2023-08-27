<template>

<div id="accordion-collapse" data-accordion="open">

    <div class="border-t-2 last:border-b-2 border-slate-300 dark:border-slate-600" v-for="(cluster, i) in props.clusters" :key="`cluster${i}`">
        <button type="button" :id="`accordion-cluster${i}-heading`"
        class="flex items-center w-full p-5 font-medium text-left
            hover:bg-gray-100 dark:hover:bg-gray-800" 
        :data-accordion-target="`#accordion-cluster${i}-body`" aria-expanded="false" :aria-controls="`accordion-cluster${i}-body`">
            <svg data-accordion-icon class="w-3 h-3 shrink-0 mr-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
                <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5 5 1 1 5"/>
            </svg>
            <h2>{{ cluster.topics.join(" &#183; ") }}</h2>
        </button>
        
        <div :id="`accordion-cluster${i}-body`" class="hidden" :aria-labelledby="`accordion-cluster${i}-heading`">
            <Article v-for="article in cluster.docs" :article="article" class="first:mt-8 mb-8" />
        </div>
    </div>

</div>

</template>


<script setup lang="ts">
    import { initFlowbite } from "flowbite";
    import { Cluster } from "~/types/Cluster";
    
    const props = defineProps({
        clusters: {
            type: Object as PropType<Cluster[]>,
            required: true,
        },
    });

    onMounted(() => {
        initFlowbite();
    })
</script>