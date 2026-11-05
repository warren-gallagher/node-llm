import { describe, expect, it } from "vitest";
import { fileCheck, readChainClaims } from "../../src/credential-demo/checks";
import { datasetDpp, evaluationDcc, modelDpp } from "../../src/credential-demo/claims";
import type { CredentialType, VerificationResult } from "../../src/credential-demo/client";
import { credentialGraph, NODE_WIDTH } from "../../src/credential-demo/viewer-graph";
import { CTX, DATASET, EVALUATION, issued, MODEL } from "./fixtures";

const PASS = { status: "pass" as const, code: "OK", message: "fine" };

function result(
  id: string,
  type: CredentialType,
  children: VerificationResult[] = [],
  verified = true,
): VerificationResult {
  return {
    verified,
    credentialId: `https://x.example.com/credentials/${type}/2026/${id}`,
    credentialType: type,
    issuer: "did:web:x.example.com",
    name: `Credential ${id}`,
    checks: { proof: PASS },
    upstreamResults: children.map((child) => ({ ...child, link: { url: child.credentialId } })),
    errors: verified ? [] : [{ code: "FAILED", message: "failed" }],
    warnings: [],
  };
}

/** The demo's chain: the model passport links the DTE, the dataset passport and both DCCs. */
function demoChain(evaluationVerified = true): VerificationResult {
  const accreditationB = result("accreditation-b", "dcc");
  const accreditationM = result("accreditation-m", "dcc");
  const rights = result("rights", "dcc", [accreditationB]);
  const dataset = result("dataset", "dpp", [rights]);
  const training = result("training", "dte", [dataset, rights]);
  const evaluation = result("evaluation", "dcc", [accreditationM], evaluationVerified);
  return result("model", "dpp", [training, dataset, rights, evaluation], evaluationVerified);
}

function column(graph: ReturnType<typeof credentialGraph>, id: string): number | undefined {
  return graph.nodes.find((node) => node.id.endsWith(`/${id}`))?.column;
}

describe("credential viewer graph", () => {
  const graph = credentialGraph(demoChain());

  it("shows each credential once, and each link once", () => {
    expect(graph.nodes).toHaveLength(7);
    expect(graph.edges).toHaveLength(9);
  });

  it("puts each credential after every credential that links it, so links point right", () => {
    expect(column(graph, "model")).toBe(0);
    expect(column(graph, "training")).toBe(1);
    expect(column(graph, "evaluation")).toBe(1);
    expect(column(graph, "dataset")).toBe(2);
    expect(column(graph, "rights")).toBe(3);
    expect(column(graph, "accreditation-b")).toBe(4);
    const byId = new Map(graph.nodes.map((node) => [node.id, node]));
    for (const edge of graph.edges) {
      expect(byId.get(edge.to)?.x).toBeGreaterThan(byId.get(edge.from)?.x ?? Infinity);
    }
  });

  it("routes links that skip a column in lanes above every node", () => {
    const columns = new Map(graph.nodes.map((node) => [node.id, node.column]));
    const top = Math.min(...graph.nodes.map((node) => node.y));
    const lanes = new Set<number>();
    for (const edge of graph.edges) {
      const span = (columns.get(edge.to) ?? 0) - (columns.get(edge.from) ?? 0);
      const across = /L[\d.]+,([\d.]+)/.exec(edge.path);
      if (span === 1) {
        expect(across).toBeNull();
      } else {
        expect(Number(across?.[1])).toBeLessThan(top);
        lanes.add(Number(across?.[1]));
      }
    }
    // model -> dataset passport, model -> rights and training -> rights each get a lane
    expect(lanes.size).toBe(3);
  });

  it("gives the drawing room for every node", () => {
    for (const node of graph.nodes) {
      expect(node.x + NODE_WIDTH).toBeLessThanOrEqual(graph.width);
      expect(node.y).toBeLessThan(graph.height);
    }
  });

  it("carries each credential's status, checks and errors", () => {
    const failing = credentialGraph(demoChain(false));
    const evaluation = failing.nodes.find((node) => node.id.endsWith("/evaluation"));
    expect(evaluation).toMatchObject({
      verified: false,
      name: "Credential evaluation",
      checks: [{ name: "proof", status: "pass", message: "fine" }],
      errors: ["FAILED: failed"],
    });
  });

  it("survives a link back to a credential on the path", () => {
    const loop = result("a", "dpp");
    const b = result("b", "dte", [loop]);
    loop.upstreamResults = [{ ...b, link: { url: b.credentialId } }];
    expect(credentialGraph(loop).nodes).toHaveLength(2);
  });
});

describe("checking one file", () => {
  const rights = issued("dcc", "rights");
  const claims = readChainClaims(
    {
      credentialSubject: modelDpp(CTX, DATASET, MODEL, EVALUATION, {
        datasetPassport: issued("dpp", "dataset"),
        rights,
        trainingEvent: issued("dte", "training"),
        evaluation: issued("dcc", "evaluation"),
      }).credentialSubject,
    },
    { credentialSubject: datasetDpp(CTX, DATASET, rights).credentialSubject },
    {
      credentialSubject: evaluationDcc(CTX, DATASET, MODEL, EVALUATION, issued("dcc", "acc"))
        .credentialSubject,
    },
  );

  it("recognises the weights, a listed file and the dataset by hash", () => {
    expect(fileCheck(claims, MODEL.weights)).toMatchObject({ passed: true });
    expect(fileCheck(claims, MODEL.config).detail).toContain("config.json, as the model passport");
    expect(fileCheck(claims, DATASET.file).detail).toContain("the dataset its passport identifies");
  });

  it("names the file a changed copy should have been", () => {
    const changed = { ...MODEL.weights, sha256: "9".repeat(64) };
    expect(fileCheck(claims, changed)).toMatchObject({ passed: false });
    expect(fileCheck(claims, changed).detail).toContain(
      "not the weights.bin the model passport lists",
    );
    expect(fileCheck(claims, { ...changed, name: "other.bin" }).detail).toContain(
      "matches nothing",
    );
  });
});
