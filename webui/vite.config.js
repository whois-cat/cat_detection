import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// In dev, API, live WebSockets and recordings come from a running streamhub.
const streamhub = process.env.STREAMHUB || 'http://127.0.0.1:8096';

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': {
        target: streamhub,
        ws: true,
        changeOrigin: true,
        // streamhub only accepts same-origin WebSockets.
        configure: proxy => proxy.on('proxyReqWs', req => req.setHeader('origin', streamhub)),
      },
      '/recordings': { target: streamhub, changeOrigin: true },
    },
  },
});
