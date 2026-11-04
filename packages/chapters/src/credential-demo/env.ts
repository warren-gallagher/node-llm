import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const REPO_ROOT = new URL("../../../../", import.meta.url);

/**
 * The platform settings the demo needs. They are read from `.env` at the repository root
 * (git-ignored) with Node's built-in `process.loadEnvFile`:
 *
 * - `AQ_API_URL`: the platform's public URL, the same value `realtime` has as `AQIO_API_URL`
 *   (locally, its ngrok URL). Request tokens are bound to it through the `htu` claim;
 * - `AQ_TENANT_ID`: the tenant the API key belongs to;
 * - `AQ_API_KEY`: a v2 API key (hex) with Admin access.
 */
export interface DemoEnv {
  apiUrl: string;
  tenantId: string;
  apiKey: string;
}

export function loadDemoEnv(): DemoEnv {
  const envFile = fileURLToPath(new URL(".env", REPO_ROOT));
  if (existsSync(envFile)) {
    process.loadEnvFile(envFile);
  }

  const names = ["AQ_API_URL", "AQ_TENANT_ID", "AQ_API_KEY"] as const;
  const missing = names.filter((name) => !process.env[name]);
  if (missing.length > 0) {
    throw new Error(`missing ${missing.join(", ")}: set them in ${envFile}`);
  }

  const apiUrl = (process.env.AQ_API_URL as string).replace(/\/+$/, "");
  if (!URL.canParse(apiUrl)) {
    throw new Error("AQ_API_URL is not a URL");
  }
  const apiKey = process.env.AQ_API_KEY as string;
  if (!/^(?:[0-9a-f]{2})+$/i.test(apiKey)) {
    throw new Error("AQ_API_KEY must be a v2 API key (a hex string)");
  }
  return { apiUrl, tenantId: process.env.AQ_TENANT_ID as string, apiKey };
}
