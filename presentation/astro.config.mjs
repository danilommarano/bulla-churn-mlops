import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import expressiveCode from "astro-expressive-code";

// Deploy-ready: em GitHub Pages/Vercel defina SITE_BASE (ex.: "/churn/"); local fica "/".
const base = process.env.SITE_BASE ?? "/";

export default defineConfig({
  base,
  integrations: [
    expressiveCode({
      themes: ["dracula"],
      styleOverrides: { borderRadius: "0.5rem" },
    }),
  ],
  vite: { plugins: [tailwindcss()] },
});
