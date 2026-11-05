import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { type ChainReading, listModels, pickModel, readChain } from "./buyer";
import { claimChecks, fileCheck } from "./checks";
import { AqClient } from "./client";
import { loadDemoEnv } from "./env";
import type { FileDigest } from "./hashing";
import { loadManifest, type Manifest } from "./state";
import { credentialGraph } from "./viewer-graph";

/**
 * Credential demo, the viewer: a local page that draws a model's credential graph from the
 * platform's verification of its model passport, links each credential to its did:web page, and
 * checks files dropped on it against the passports. The browser hashes the files and sends only
 * their digests. The page is served on 127.0.0.1 only; it forwards verification to `realtime`.
 *
 * Usage: pnpm demo:viewer [port]   (default 4400)
 */

const PORT = Number(process.argv[2] ?? 4400);
const HOST = "127.0.0.1";
const VIEWER_DIR = new URL("./viewer/", import.meta.url);
const STATIC_FILES: Record<string, { file: string; type: string }> = {
  "/": { file: "index.html", type: "text/html; charset=utf-8" },
  "/viewer.js": { file: "viewer.js", type: "text/javascript; charset=utf-8" },
};
const MAX_BODY_BYTES = 64 * 1024;

const env = loadDemoEnv();
const client = new AqClient(env);
/** The latest reading of each run, so dropped files are checked against what the page shows. */
const readings = new Map<string, ChainReading>();

createServer((request, response) => {
  handle(request, response).catch((error) => {
    send(response, 500, { error: error instanceof Error ? error.message : String(error) });
  });
}).listen(PORT, HOST, () => {
  console.log(`Credential viewer: http://${HOST}:${PORT}/`);
});

async function handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const { pathname } = new URL(request.url ?? "/", `http://${HOST}`);
  const staticFile = STATIC_FILES[pathname];
  if (request.method === "GET" && staticFile) {
    response.writeHead(200, { "Content-Type": staticFile.type, "Cache-Control": "no-store" });
    response.end(readFileSync(new URL(staticFile.file, VIEWER_DIR)));
    return;
  }
  if (request.method === "GET" && pathname === "/api/runs") {
    send(response, 200, runs(manifest()));
    return;
  }
  if (request.method === "POST" && pathname === "/api/chain") {
    const { runId } = JSON.parse(await readBody(request));
    send(response, 200, await chain(runId));
    return;
  }
  if (request.method === "POST" && pathname === "/api/files") {
    const { runId, files } = JSON.parse(await readBody(request));
    send(response, 200, checkFiles(runId, files));
    return;
  }
  send(response, 404, { error: "not found" });
}

function manifest(): Manifest {
  return loadManifest(env.apiUrl, env.tenantId);
}

/** The models in the manifest, runs and derivations, newest first. */
function runs(current: Manifest) {
  return listModels(current).map((model) => ({
    runId: model.id,
    modelPassport: model.modelPassport,
  }));
}

/** Verifies a model's passport through the platform, and lays out the result. */
async function chain(runId: string) {
  const model = pickModel(manifest(), runId);
  const reading = await readChain(client, model.passport.credentialId);
  readings.set(runId, reading);
  return {
    runId,
    modelPassport: model.passport.credentialId,
    verified: reading.platform.verified,
    graph: credentialGraph(reading.platform),
    checks: [reading.credentialCheck, ...claimChecks(reading.claims)],
  };
}

/** Checks each dropped file, by its digest, against the run's credentials as last read. */
function checkFiles(runId: string, files: FileDigest[]) {
  const reading = readings.get(runId);
  if (!reading) throw new Error(`verify run ${runId} first`);
  if (!Array.isArray(files)) throw new Error("expected a list of files");
  return files.map((file) => fileCheck(reading.claims, file));
}

function readBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk: string) => {
      body += chunk;
      if (body.length > MAX_BODY_BYTES) {
        reject(new Error("request too large"));
        request.destroy();
      }
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });
}

function send(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}
