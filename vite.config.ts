import { defineConfig } from 'vitest/config';

export default defineConfig({
  // relative asset paths so the build works on GitHub Pages and any sub-path
  base: './',
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1000,
  },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
});
