<template>

    <form @submit.prevent="formSubmitted">
        <div class="relative w-full">
            <label for="input-query" class="sr-only">Query</label>
            <input type="search" id="input-query" placeholder="Search" v-model="query" required
            class="border rounded-lg block p-2.5 w-full z-20 text-sm
                bg-gray-50 border-gray-300 focus:border-blue-500 
                dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400  dark:focus:border-blue-500">
            <button type="submit"
            class="absolute top-0 right-0 p-2.5 h-full border rounded-r-lg border-blue-700
                bg-blue-700 hover:bg-blue-800
                dark:bg-blue-600 dark:hover:bg-blue-700">
                <img src="~/assets/images/icons/search.svg" alt="Search" class="invert w-4 h-4" aria-hidden="true" />
            </button>
        </div>
    </form>

</template>

<script setup lang="ts">
    const props = defineProps({
        onquery: {
            type: Function as PropType<(query: String) => void>,
            required: true,
        },
    });

    const query = ref("")

    function formSubmitted(e: Event) {
        props.onquery(query.value)
    }
</script>