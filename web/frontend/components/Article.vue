<template>
    <div>
        <h3 class="font-bold text-xl mb-1">{{ props.article.title }}</h3>
        <p class="mb-1">{{ props.article.authors?.join(" &#183; ") }}</p>
        <p class="mb-2">
            <client-only>
                {{ props.article.date ? props.article.date.toLocaleDateString() : "Date not available" }}
            </client-only>
            <span v-if="props.article.pmid != null">
                &#183; PMID: <a :href="`https://pubmed.ncbi.nlm.nih.gov/${props.article.pmid}`" class="underline">{{ props.article.pmid }}</a>
            </span>
            <span v-if="props.article.doi != null">
                &#183; DOI: <a :href="`https://doi.org/${props.article.doi}`" class="underline">{{ props.article.doi }}</a>
            </span>
        </p>
        
        <div class="whitespace-pre-wrap">
            <span v-for="chunk in abstract_chunks">
                <span v-if="chunk.is_highlighted" class="bg-amber-200 dark:bg-amber-900">{{ chunk.content }}</span>
                <span v-else>{{ chunk.content }}</span>
            </span>
        </div>
    </div>
</template>


<script setup lang="ts">
    import { Article } from "~/types/Article";

    const props = defineProps({
        article: {
            type: Object as PropType<Article>,
            required: true,
        },
    });

    interface AbstractChunk {
        content: string,
        is_highlighted: boolean
    }

    const abstract_chunks = ref<AbstractChunk[]>([])

    let abstract = props.article.abstract
    // Since the order of the sentences in the summary is not guaranteed, find the starting index first and order them
    let to_highlight: [number, string][] = [] 
    for (const sentence of props.article.summary) {
        const index = abstract.indexOf(sentence)
        if (index > -1) {
            to_highlight.push([index, sentence])
        }
    }
    to_highlight.sort((a,b) => a[0] - b[0])


    let curr_offset = 0 // Index to start from at the next iteration
    for (const [index, sentence] of to_highlight) {
        // Offset needed since the indexes for the selected sentences are relative to the full abstract
        const real_highlight_index = index - curr_offset
        
        // Part before the highlighed sentence 
        const non_highlight = abstract.slice(0, real_highlight_index)
        if (non_highlight.length > 0) {
            abstract_chunks.value.push({
                content: non_highlight,
                is_highlighted: false
            })
        }

        // Highlighted sentence
        abstract_chunks.value.push({
            content: abstract.slice(real_highlight_index, real_highlight_index + sentence.length),
            is_highlighted: true
        })
        
        // Remove the processed part of the abstract
        abstract = abstract.slice(real_highlight_index + sentence.length)
        curr_offset = real_highlight_index + sentence.length
    }

    // Remaining part
    abstract_chunks.value.push({
        content: abstract,
        is_highlighted: false
    })
</script>