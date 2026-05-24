import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// In dev mode (`vite dev`) we proxy backend paths to host services so the
// browser sees the same URLs as in production. host.docker.internal works
// inside the dev container via extra_hosts; for direct local dev it also
// resolves on recent Docker Desktop / docker-ce on Linux.
const upstream = process.env.UPSTREAM_HOST || 'host.docker.internal';

export default defineConfig({
  plugins: [svelte()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    watch: { usePolling: true, interval: 300 },
    proxy: {
      '/detector':         { target: `http://${upstream}:8091`, ws: true, changeOrigin: true,
                             rewrite: p => p.replace(/^\/detector/, '') },
      '/detector-control': { target: `http://${upstream}:8092`,            changeOrigin: true,
                             rewrite: p => p.replace(/^\/detector-control/, '') },
      // mediamtx WebRTC (WHEP) and recording playback
      '/whep':             { target: `http://${upstream}:8889`,            changeOrigin: true,
                             rewrite: p => p.replace(/^\/whep/, '') },
      '/recordings':       { target: `http://${upstream}:9996`,            changeOrigin: true,
                             rewrite: p => p.replace(/^\/recordings/, '') },
    },
  },
});
