import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173 },
  optimizeDeps: {
    // tfjs backends ship large ESM bundles; let Vite pre-bundle them.
    include: [
      "@tensorflow/tfjs-core",
      "@tensorflow/tfjs-backend-webgpu",
      "@tensorflow/tfjs-backend-cpu",
    ],
  },
});
