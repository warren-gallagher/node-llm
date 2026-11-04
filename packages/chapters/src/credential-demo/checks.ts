import { createHash } from "node:crypto";
import { digestMultibase, type FileDigest, niUri } from "./hashing";

/**
 * The buyer's own checks, which the platform can't do: it verifies that credentials are genuine
 * and agree with each other, but not that the files in the buyer's hands are the ones they
 * describe, or that the evaluation they report can be reproduced.
 *
 * Everything here is pure: the caller fetches the credentials, hashes the files and reruns the
 * evaluation.
 */

export type CheckGroup = "credentials" | "claims" | "artifact" | "evaluation";

export interface DemoCheck {
  group: CheckGroup;
  name: string;
  passed: boolean;
  detail: string;
}

/** A UNTP 0.7.0 link between credentials. */
export interface Link {
  linkURL: string;
  linkName?: string;
  linkType: string;
  digestMultibase?: string;
}

/** A file as a passport lists it (the AI model extension's `files`). */
export interface ClaimedFile {
  name: string;
  byteLength: number;
  digestMultibase: string;
}

/** What the buyer reads from the model's credentials. */
export interface ChainClaims {
  modelId: string;
  modelFiles: ClaimedFile[];
  /** The loss the model passport claims, citing the evaluation. */
  claimedLoss: number;
  datasetId: string;
  evaluation: {
    assessedProductId: string;
    loss: number;
    datasetId: string;
    harness: { name: string; version: string; repository: string; commit: string };
    split: string;
    metric: string;
  };
}

/** The links from a model passport that the buyer follows. */
export interface ModelLinks {
  datasetPassport: Link;
  evaluation: Link;
}

/** How far a rerun loss may be from the claimed one; covers float differences across machines. */
export const LOSS_TOLERANCE = 1e-4;

const LINK_TYPES = "https://test.uncefact.org/vocabulary/linkTypes/";

// biome-ignore lint/suspicious/noExplicitAny: credentials are read loosely, like any JSON
type Json = any;

/** The payload of a JWT, decoded without checking its signature (the platform did that). */
export function decodeJwtPayload(jwt: string): Json {
  const payload = jwt.split(".")[1];
  if (!payload) throw new Error("not a JWT");
  return JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
}

/** `digestMultibase` of some bytes: `u` + base64url of the SHA-256 multihash. */
export function digestOf(data: string | Uint8Array): string {
  return digestMultibase(createHash("sha256").update(data).digest("hex"));
}

/**
 * The model passport's links to the dataset passport (its one `dpp` link) and to the evaluation
 * (the evidence for the claim that carries a measured value).
 */
export function modelLinks(modelPassport: Json): ModelLinks {
  const subject = modelPassport.credentialSubject;
  const datasetLinks = (subject.relatedDocument ?? []).filter(
    (link: Link) => link.linkType === `${LINK_TYPES}dpp`,
  );
  if (datasetLinks.length !== 1) {
    throw new Error(`expected one dataset passport link, found ${datasetLinks.length}`);
  }
  const evaluation = lossClaim(modelPassport).evidence?.find(
    (link: Link) => link.linkType === `${LINK_TYPES}dcc`,
  );
  if (!evaluation) throw new Error("the model passport's loss claim cites no DCC");
  return { datasetPassport: datasetLinks[0], evaluation };
}

/** Reads the facts the checks compare from the three credentials. */
export function readChainClaims(
  modelPassport: Json,
  datasetPassport: Json,
  evaluation: Json,
): ChainClaims {
  const assessment = evaluation.credentialSubject.conformityAssessment[0];
  const run = assessment.evaluationRun;
  return {
    modelId: modelPassport.credentialSubject.id,
    modelFiles: modelPassport.credentialSubject.characteristics?.files ?? [],
    claimedLoss: lossClaim(modelPassport).claimedPerformance[0].measure.value,
    datasetId: datasetPassport.credentialSubject.id,
    evaluation: {
      assessedProductId: assessment.assessedProduct[0].product.id,
      loss: assessment.assessedPerformance[0].measure.value,
      datasetId: run.datasetId,
      harness: run.harness,
      split: run.split,
      metric: run.metric,
    },
  };
}

function lossClaim(modelPassport: Json): Json {
  const claim = (modelPassport.credentialSubject.performanceClaim ?? []).find(
    (c: Json) => c.claimedPerformance?.[0]?.measure !== undefined,
  );
  if (!claim) throw new Error("the model passport makes no measured performance claim");
  return claim;
}

/** Each credential the buyer read is the one its link names, byte for byte. */
export function credentialDigestCheck(read: { link: Link; jwt: string }[]): DemoCheck {
  const mismatched = read.filter(({ link, jwt }) => link.digestMultibase !== digestOf(jwt));
  return {
    group: "credentials",
    name: "Credentials read match the digests in their links",
    passed: mismatched.length === 0,
    detail:
      mismatched.length === 0
        ? `${read.length} credentials`
        : `different: ${mismatched.map(({ link }) => link.linkName ?? link.linkURL).join(", ")}`,
  };
}

/** The credentials tell one story: the evaluation covers this model, on this dataset. */
export function claimChecks(claims: ChainClaims): DemoCheck[] {
  const { evaluation } = claims;
  return [
    {
      group: "claims",
      name: "The evaluation assessed the model in the passport",
      passed: evaluation.assessedProductId === claims.modelId,
      detail: evaluation.assessedProductId,
    },
    {
      group: "claims",
      name: "The evaluation used the dataset in the dataset passport",
      passed: evaluation.datasetId === claims.datasetId,
      detail: evaluation.datasetId,
    },
    {
      group: "claims",
      name: "The passport claims the loss the evaluation measured",
      passed: claims.claimedLoss === evaluation.loss,
      detail: `passport ${claims.claimedLoss}, evaluation ${evaluation.loss}`,
    },
  ];
}

/** The files in the buyer's hands are the ones the passports identify. */
export function artifactChecks(
  claims: ChainClaims,
  files: { weights: FileDigest; config: FileDigest; dataset: FileDigest },
): DemoCheck[] {
  return [
    {
      group: "artifact",
      name: "weights.bin is the model in the passport",
      passed: niUri(files.weights.sha256) === claims.modelId && listed(claims, files.weights),
      detail: `sha256 ${files.weights.sha256}`,
    },
    {
      group: "artifact",
      name: "config.json is the one the passport lists",
      passed: listed(claims, files.config),
      detail: `sha256 ${files.config.sha256}`,
    },
    {
      group: "artifact",
      name: "The dataset file is the dataset in its passport",
      passed: niUri(files.dataset.sha256) === claims.datasetId,
      detail: `sha256 ${files.dataset.sha256}`,
    },
  ];
}

function listed(claims: ChainClaims, file: FileDigest): boolean {
  return claims.modelFiles.some(
    (claimed) =>
      claimed.name === file.name &&
      claimed.byteLength === file.byteLength &&
      claimed.digestMultibase === digestMultibase(file.sha256),
  );
}

/** Rerunning the named evaluation on the buyer's files gives the claimed loss. */
export function evaluationRerunCheck(
  claims: ChainClaims,
  measured: number,
  harnessVersion: string,
  tolerance = LOSS_TOLERANCE,
): DemoCheck {
  const { evaluation } = claims;
  const difference = Math.abs(measured - evaluation.loss);
  const versionNote =
    harnessVersion === evaluation.harness.version
      ? ""
      : `; harness ${harnessVersion} here, ${evaluation.harness.version} in the evaluation`;
  return {
    group: "evaluation",
    name: "Rerunning the evaluation gives the claimed loss",
    passed: difference <= tolerance,
    detail: `measured ${measured}, claimed ${evaluation.loss}, difference ${difference.toExponential(1)} (tolerance ${tolerance})${versionNote}`,
  };
}

// ============================================================================
// Tampering, for demo:tamper
// ============================================================================

/** A copy of the weights with the first float32 value changed by 0.001. */
export function changeOneWeight(weights: Uint8Array): Buffer {
  const copy = Buffer.from(weights);
  copy.writeFloatLE(copy.readFloatLE(0) + 0.001, 0);
  return copy;
}

/** A copy of The Verdict with one name changed in the opening paragraph. */
export function changeOneWord(text: string): string {
  const changed = text.replace("Gisburn", "Gisborne");
  if (changed === text) throw new Error("the text has no 'Gisburn' to change");
  return changed;
}
