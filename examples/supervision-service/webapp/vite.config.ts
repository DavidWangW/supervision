import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), vueDevTools()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      // ``ws: true`` lets the live stream WebSocket (/api/v1/streams/{id}/ws)
      // be proxied to the backend during development.
      '/api': { target: 'http://127.0.0.1:8005', ws: true },
      '/health': 'http://127.0.0.1:8005',
      '/docs': 'http://127.0.0.1:8005',
      '/openapi.json': 'http://127.0.0.1:8005',
    },
  },
})
