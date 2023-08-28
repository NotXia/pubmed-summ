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
            <span v-for="(chunk, i) in abstract_chunks">
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


    onMounted(() => highlight())

    watch(() => props.article, (new_article, old_article) => {
        highlight()
    })


    function highlight() {
        if (props.article.summary == null) {
            abstract_chunks.value = [{ 
                content: props.article.abstract,
                is_highlighted: false
            }]
            return
        }

        let abstract = props.article.abstract
        let new_chunks: AbstractChunk[] = []

        // Since the order of the sentences in the summary may not be ordered, find the starting index first and order them
        let to_highlight: [number, string][] = [] 
        for (const sentence of props.article.summary) {
            const index = abstract.indexOf(sentence)
            if (index > -1) {
                to_highlight.push([index, sentence])
            }
        }
        to_highlight.sort((a,b) => a[0] - b[0])

        let curr_offset = 0
        for (const [index, sentence] of to_highlight) {
            // Offset needed since the indexes for the selected sentences are relative to the full abstract
            const highlight_start = index - curr_offset
            const highlight_end = highlight_start + sentence.length
    
            // Part before the highlighed sentence 
            const non_highlighted = abstract.slice(0, highlight_start)
            if (non_highlighted.length > 0) {
                new_chunks.push({
                    content: non_highlighted,
                    is_highlighted: false
                })
            }

            // Highlighted sentence
            new_chunks.push({
                content: abstract.slice(highlight_start, highlight_end),
                is_highlighted: true
            })

            // Remove the processed part of the abstract
            abstract = abstract.slice(highlight_end)
            curr_offset += highlight_end
        }

        // Remaining part
        new_chunks.push({
            content: abstract,
            is_highlighted: false
        })

        abstract_chunks.value = new_chunks
    }

</script>