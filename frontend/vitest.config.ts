/// <reference types="vitest/config" />
import path from "node:path";
import { fileURLToPath } from "node:url";
import { getViteConfig } from "astro/config";

// The guard's data lives at the REPO root, one level up. Computed from this file's location,
// never hardcoded: a machine-specific absolute path is what once broke CI for the Python half.
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export default getViteConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    reporters: ["default", ["tdd-guard-vitest", { projectRoot: repoRoot }]],
  },
});
