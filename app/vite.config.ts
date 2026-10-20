/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
// @ts-expect-error type error without @types/node package
import process from "node:process";
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],

  // Vite options for `tauri dev` / `tauri build`:
  // 1. keep Rust errors visible
  clearScreen: false,
  // 2. Tauri expects a fixed port; fail if it is taken
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. don't watch the Rust side
      ignored: ["**/src-tauri/**"],
    },
  },
  test: {
    environment: "jsdom",
  },
}));
