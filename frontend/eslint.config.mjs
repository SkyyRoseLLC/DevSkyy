import js from "@eslint/js";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import nextPlugin from "@next/eslint-plugin-next";
import reactHooksPlugin from "eslint-plugin-react-hooks";

export default [
  // Global ignores (must be first, standalone object)
  {
    ignores: [".next/", "node_modules/", "out/", "coverage/", ".claude/"],
  },
  // Base JS recommended rules
  js.configs.recommended,
  // TypeScript + Next.js config
  {
    files: ["**/*.{ts,tsx,js,jsx,mjs}"],
    plugins: {
      "@typescript-eslint": tsPlugin,
      "@next/next": nextPlugin,
      "react-hooks": reactHooksPlugin,
    },
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true },
      },
      globals: {
        // Browser globals
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        console: "readonly",
        fetch: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        setTimeout: "readonly",
        setInterval: "readonly",
        clearTimeout: "readonly",
        clearInterval: "readonly",
        requestAnimationFrame: "readonly",
        cancelAnimationFrame: "readonly",
        localStorage: "readonly",
        sessionStorage: "readonly",
        HTMLElement: "readonly",
        HTMLDivElement: "readonly",
        HTMLCanvasElement: "readonly",
        HTMLImageElement: "readonly",
        HTMLInputElement: "readonly",
        HTMLFormElement: "readonly",
        HTMLButtonElement: "readonly",
        HTMLAnchorElement: "readonly",
        HTMLVideoElement: "readonly",
        Element: "readonly",
        Event: "readonly",
        MouseEvent: "readonly",
        KeyboardEvent: "readonly",
        IntersectionObserver: "readonly",
        ResizeObserver: "readonly",
        MutationObserver: "readonly",
        AbortController: "readonly",
        FormData: "readonly",
        Headers: "readonly",
        Request: "readonly",
        Response: "readonly",
        ReadableStream: "readonly",
        TextEncoder: "readonly",
        TextDecoder: "readonly",
        Blob: "readonly",
        File: "readonly",
        FileReader: "readonly",
        Image: "readonly",
        performance: "readonly",
        MediaQueryList: "readonly",
        CustomEvent: "readonly",
        WebSocket: "readonly",
        crypto: "readonly",
        atob: "readonly",
        btoa: "readonly",
        structuredClone: "readonly",
        // Node globals (for config files, scripts)
        process: "readonly",
        require: "readonly",
        module: "readonly",
        __dirname: "readonly",
        __filename: "readonly",
        Buffer: "readonly",
        global: "readonly",
        // React/JSX globals
        React: "readonly",
        JSX: "readonly",
      },
    },
    rules: {
      // TypeScript rules
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_" },
      ],
      "@typescript-eslint/no-explicit-any": "warn",
      // Disable base rules handled by TypeScript
      "no-unused-vars": "off",
      "no-undef": "off", // TypeScript handles undefined variable checks

      // React hooks rules
      "react-hooks/exhaustive-deps": "warn",
      "react-hooks/rules-of-hooks": "error",

      // Next.js rules
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,

      // General rules
      "no-console": ["warn", { allow: ["warn", "error"] }],
    },
  },
  // Command-line, E2E, queue, and autonomous workers intentionally emit
  // operator-visible progress. Browser application modules remain protected
  // by the global no-console rule above.
  {
    files: [
      "scripts/**/*.{ts,js,mjs}",
      "e2e/**/*.{ts,js,mjs}",
      "lib/autonomous/**/*.{ts,js}",
      "lib/queue/**/*.{ts,js}",
    ],
    rules: {
      "no-console": "off",
    },
  },
  // These API clients mirror schemaless third-party payloads. Their public
  // typed methods are validated by TypeScript; keeping `any` at the raw JSON
  // boundary avoids falsely asserting a stable vendor response schema.
  {
    files: [
      "lib/wordpress/operations-manager.ts",
      "lib/vercel/deployment-manager.ts",
      "scripts/deploy.ts",
    ],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  // Asset/QA views must display the exact source URL (including signed and
  // local object URLs) without image transformation; storefront images still
  // require next/image through the global rule.
  {
    files: [
      "app/admin/assets/page.tsx",
      "app/admin/imagery/page.tsx",
      "app/admin/wordpress/page.tsx",
      "components/admin/assets/*.tsx",
      "components/admin/qa/*.tsx",
    ],
    rules: {
      "@next/next/no-img-element": "off",
    },
  },
];
