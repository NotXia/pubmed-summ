<template>
    <button class="rounded-full p-1 hover:bg-slate-200 dark:hover:bg-slate-700" @click="changeTheme">
        <div class="w-5 h-5 flex items-center justify-center">
            <div v-if="current_theme === 'light'">
                <img src="~/assets/images/icons/moon.svg" alt="Dark theme" class="h-full w-full" />
            </div>

            <div v-if="current_theme === 'dark'">
                <img src="~/assets/images/icons/sun.svg" alt="Light theme" class="invert h-full w-full" />
            </div>
        </div>
    </button>
</template>

<script setup lang="ts">
    const current_theme = ref("");

    onMounted(() => {
        current_theme.value = getTheme();
    })

    function changeTheme() {
        flipTheme();
        current_theme.value = getTheme();
        applyTheme(current_theme.value);
    }
</script>