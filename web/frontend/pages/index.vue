<template>
    <div class="flex justify-center items-center">
        AAAA
    </div>
</template>

<script setup lang="ts">
    import { io } from "socket.io-client";
    const config = useRuntimeConfig()

    const socket = io(config.public.BACKEND_SOCKET, { path: config.public.BACKEND_SOCKETIO_PATH })

    socket.on("connect", () => {
        console.log(socket.id);
    });
</script>