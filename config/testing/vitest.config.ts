import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const configDirectory = path.dirname(fileURLToPath(import.meta.url));
const rootDirectory = path.resolve(configDirectory, '../..');

export default defineConfig({
  resolve: {
    alias: {
      'three/examples/jsm/renderers/CSS2DRenderer.js': path.join(rootDirectory, 'tests/__mocks__/three-examples.cjs'),
      three: path.join(rootDirectory, '__mocks__/three.cjs'),
      '@': path.join(rootDirectory, 'src'),
      '@/components': path.join(rootDirectory, 'src/components'),
      '@/utils': path.join(rootDirectory, 'src/utils'),
      '@/types': path.join(rootDirectory, 'src/types'),
      '@/services': path.join(rootDirectory, 'src/services'),
      '@/hooks': path.join(rootDirectory, 'src/hooks'),
      '@/store': path.join(rootDirectory, 'src/store'),
      '@/api': path.join(rootDirectory, 'src/api'),
      '@/config': path.join(rootDirectory, 'src/config'),
      '@/security': path.join(rootDirectory, 'src/security'),
    },
  },
  test: {
    root: rootDirectory,
    globals: true,
    environment: 'node',
    include: ['src/**/__tests__/**/*.{ts,tsx,js,jsx}', 'src/**/*.{test,spec}.{ts,tsx,js,jsx}'],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/.next/**',
      '**/.nuxt/**',
      '**/wordpress-theme/**/tests/e2e/**',
      '**/frontend/**',
      '**/skyyrose-suite/**',
      '**/design-system/skyyrose-storefront/test/**',
    ],
    setupFiles: [path.join(rootDirectory, 'tests/setup-three-mocks.ts')],
    clearMocks: true,
    restoreMocks: true,
    testTimeout: 10_000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov', 'html', 'json'],
      reportsDirectory: path.join(rootDirectory, 'coverage'),
      include: ['src/**/*.{ts,tsx,js,jsx}'],
      exclude: [
        'src/**/*.d.ts',
        'src/**/*.config.{ts,js}',
        'src/**/index.{ts,js}',
        'src/**/*.stories.{ts,tsx,js,jsx}',
        'src/collections/ARTryOnViewer.ts',
        'src/collections/BaseCollectionExperience.ts',
        'src/collections/EnvironmentTransition.ts',
        'src/collections/WebXRARViewer.ts',
        'src/collections/ShowroomExperience.ts',
        'src/collections/RunwayExperience.ts',
        'src/lib/ModelAssetLoader.ts',
        'src/collections/ProductionHandlers.ts',
        'src/app/layout.tsx',
      ],
      thresholds: {
        global: {
          branches: 77,
          functions: 80,
          lines: 80,
          statements: 80,
        },
      },
    },
  },
});
