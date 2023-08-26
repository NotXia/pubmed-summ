// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    devtools: { enabled: true },
    css: ["~/assets/css/main.css"],
    postcss: {
        plugins: {
        tailwindcss: {},
        autoprefixer: {},
        },
    },
    runtimeConfig: {
        public: {
            BACKEND_URL: process.env.BACKEND_URL,
            BACKEND_SOCKET: process.env.BACKEND_SOCKET,
            BACKEND_SOCKETIO_PATH: process.env.BACKEND_SOCKETIO_PATH
        }
    }
})
