import { createHash } from "node:crypto";
import type { WeightChangeProfile } from "./forensics";
import { digestMultibase, type FileDigest, niUri } from "./hashing";

/**
 * The buyer's own checks, which the platform can't do: it verifies that credentials are genuine
 * and agree with each other, but not that the files in the buyer's hands are the ones they
 * describe, or that the evaluation they report can be reproduced.
 *
 * Everything here is pure: the caller fetches the credentials, hashes the files and reruns the
 * evaluation.
 */

export type CheckGroup = "credentials" | "claims" | "artifact" | "evaluation" | "derivation";

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
  /** Present when the model was derived from another model rather than trained. */
  derivation?: DerivationClaim;
}

/** What a derivation event says it did: the method, and the parent and derived models' ids. */
export interface DerivationClaim {
  processType: string;
  hyperparameters: unknown;
  parentId: string;
  derivedId: string;
}

/** The links from a model passport that the buyer follows. */
export interface ModelLinks {
  /** The evaluation the passport's loss claim cites. */
  evaluation: Link;
  /** The event that produced the model: its training run, or its derivation. */
  event?: Link;
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
 * The model passport's links to the evaluation (the evidence for the claim that carries a measured
 * value) and to the event that produced the model (its one `dte` link). The dataset passport isn't
 * linked from a derived model's passport, so the buyer finds it through the evaluation.
 */
export function modelLinks(modelPassport: Json): ModelLinks {
  const subject = modelPassport.credentialSubject;
  const evaluation = lossClaim(modelPassport).evidence?.find(
    (link: Link) => link.linkType === `${LINK_TYPES}dcc`,
  );
  if (!evaluation) throw new Error("the model passport's loss claim cites no DCC");
  const event = (subject.relatedDocument ?? []).find(
    (link: Link) => link.linkType === `${LINK_TYPES}dte`,
  );
  return { evaluation, event };
}

/**
 * Reads the facts the checks compare from the credentials. `event` is the event that produced the
 * model; when it made the model from another model, not from the dataset, it is a derivation.
 */
export function readChainClaims(
  modelPassport: Json,
  datasetPassport: Json,
  evaluation: Json,
  event?: Json,
): ChainClaims {
  const assessment = evaluation.credentialSubject.conformityAssessment[0];
  const run = assessment.evaluationRun;
  return {
    derivation: derivationClaim(event, datasetPassport.credentialSubject.id),
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

function derivationClaim(event: Json, datasetId: string): DerivationClaim | undefined {
  const made = Array.isArray(event?.credentialSubject) ? event.credentialSubject[0] : undefined;
  const inputs: string[] = (made?.inputProduct ?? []).map((input: Json) => input.product?.id);
  if (!made || inputs.length === 0 || inputs.includes(datasetId)) return undefined;
  return {
    processType: made.trainingRun?.processType ?? "unknown",
    hyperparameters: made.trainingRun?.hyperparameters,
    parentId: inputs[0],
    derivedId: made.outputProduct?.[0]?.product?.id,
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
      name: `${files.weights.name} is the model in the passport`,
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

/**
 * What one file is, judged by its hash alone: the model, a file the model passport lists, or the
 * dataset. For files checked one at a time, as in `demo:viewer`.
 */
export function fileCheck(claims: ChainClaims, file: FileDigest): DemoCheck {
  const id = niUri(file.sha256);
  const digest = digestMultibase(file.sha256);
  const listedAs = claims.modelFiles.find((claimed) => claimed.digestMultibase === digest);
  const sameName = claims.modelFiles.find((claimed) => claimed.name === file.name);
  let detail: string;
  let passed = true;
  if (id === claims.modelId) {
    detail = "the model the passport identifies";
  } else if (listedAs) {
    detail = `${listedAs.name}, as the model passport lists it`;
  } else if (id === claims.datasetId) {
    detail = "the dataset its passport identifies";
  } else {
    passed = false;
    detail = sameName
      ? `not the ${sameName.name} the model passport lists`
      : "matches nothing the credentials identify";
  }
  return {
    group: "artifact",
    name: file.name,
    passed,
    detail: `${detail} (sha256 ${file.sha256})`,
  };
}

function listed(claims: ChainClaims, file: FileDigest): boolean {
  return claims.modelFiles.some(
    (claimed) =>
      claimed.name === file.name &&
      claimed.byteLength === file.byteLength &&
      claimed.digestMultibase === digestMultibase(file.sha256),
  );
}

/**
 * Rerunning a quantization from the parent's weights gives the derived model. `rerunSha256` is
 * undefined when the derivation names a method the buyer can't rerun.
 */
export function quantizationRerunChecks(
  derivation: DerivationClaim,
  modelId: string,
  parentSha256: string,
  rerunSha256: string | undefined,
): DemoCheck[] {
  const rerunId = rerunSha256 === undefined ? undefined : niUri(rerunSha256);
  return [
    {
      group: "derivation",
      name: "The derivation starts from the parent weights the buyer holds",
      passed: niUri(parentSha256) === derivation.parentId,
      detail: `parent sha256 ${parentSha256}`,
    },
    {
      group: "derivation",
      name: "Rerunning the quantization gives the model in the passport",
      passed: rerunId !== undefined && rerunId === modelId && rerunId === derivation.derivedId,
      detail:
        rerunSha256 === undefined
          ? `can't rerun "${derivation.processType}" with these settings`
          : `rerun sha256 ${rerunSha256}`,
    },
  ];
}

/**
 * The difference between the parent's weights and the derived model's fits the claimed
 * derivation: a quantization moves no value beyond half a rounding step, and a fine-tune moves
 * values further in most tensors. Undefined for derivations it has no expectation for.
 */
export function weightComparisonCheck(
  derivation: DerivationClaim,
  profile: WeightChangeProfile,
): DemoCheck | undefined {
  const detail =
    `${profile.beyondRounding} of ${profile.tensors} tensors moved beyond half a rounding step, ` +
    `${profile.withinRounding} within it, ${profile.unchanged} unchanged; largest relative ` +
    `change ${profile.largestRelativeChange.toExponential(1)}`;
  if (derivation.processType.startsWith("Quantization")) {
    return {
      group: "derivation",
      name: "The weight changes fit a quantization",
      passed: profile.beyondRounding === 0 && profile.unchanged < profile.tensors,
      detail,
    };
  }
  if (derivation.processType.startsWith("Fine-tuning")) {
    return {
      group: "derivation",
      name: "The weight changes fit a fine-tune",
      passed: profile.beyondRounding > profile.tensors / 2,
      detail,
    };
  }
  return undefined;
}

// ============================================================================
// Which claims a derived model inherits
// ============================================================================

/**
 * Where a derived passport's claim comes from, judged against the parent's passport:
 * - inherited: the parent makes a claim citing the same evidence;
 * - overrides: otherwise, the parent makes a claim against the same criterion that the derived
 *   passport doesn't inherit, so this claim replaces it;
 * - new: neither.
 */
export type ClaimOrigin = "inherited" | "overrides" | "new";

export interface ClassifiedClaim {
  name: string;
  origin: ClaimOrigin;
  /** The URLs of the credentials the claim cites. */
  evidence: string[];
  /** The parent's claim it inherits or overrides. */
  parentClaim?: string;
}

/** Classifies each claim in a derived model's passport against its parent's passport. */
export function classifyClaims(modelPassport: Json, parentPassport: Json): ClassifiedClaim[] {
  const claims: Json[] = modelPassport.credentialSubject.performanceClaim ?? [];
  const parentClaims: Json[] = parentPassport.credentialSubject.performanceClaim ?? [];
  const inheritedFrom = new Map<Json, Json>();
  for (const claim of claims) {
    const source = parentClaims.find((parent) => overlaps(evidenceOf(parent), evidenceOf(claim)));
    if (source) inheritedFrom.set(claim, source);
  }
  const inheritedParents = new Set(inheritedFrom.values());

  return claims.map((claim) => {
    const evidence = evidenceOf(claim);
    const source = inheritedFrom.get(claim);
    if (source)
      return { name: claim.name, origin: "inherited", evidence, parentClaim: source.name };
    const replaced = parentClaims.find(
      (parent) => !inheritedParents.has(parent) && overlaps(criteriaOf(parent), criteriaOf(claim)),
    );
    return replaced
      ? { name: claim.name, origin: "overrides", evidence, parentClaim: replaced.name }
      : { name: claim.name, origin: "new", evidence };
  });
}

/**
 * A derived model may inherit claims about its inputs, such as data rights, but not claims about
 * its parent's behaviour: any change to the weights can change that. So no inherited claim may
 * rest on a DCC that assessed the parent model. `assessed` maps each cited DCC to the products it
 * assessed.
 */
export function inheritanceCheck(
  classified: ClassifiedClaim[],
  assessed: Record<string, string[]>,
  parentId: string,
): DemoCheck {
  const inherited = classified.filter((claim) => claim.origin === "inherited");
  const aboutParent = inherited.filter((claim) =>
    claim.evidence.some((url) => assessed[url]?.includes(parentId)),
  );
  return {
    group: "claims",
    name: "No inherited claim rests on an assessment of the parent model",
    passed: aboutParent.length === 0,
    detail:
      aboutParent.length === 0
        ? `${inherited.length} inherited claim(s), none about the parent's behaviour`
        : `inherited from the parent's own assessment: ${aboutParent.map((c) => c.name).join(", ")}`,
  };
}

function evidenceOf(claim: Json): string[] {
  return (claim.evidence ?? []).map((link: Link) => link.linkURL);
}

function criteriaOf(claim: Json): string[] {
  return (claim.referenceCriteria ?? []).map((criterion: Json) => criterion.id);
}

function overlaps(a: string[], b: string[]): boolean {
  return a.some((item) => b.includes(item));
}

/** The products a DCC assessed. */
export function assessedProducts(dcc: Json): string[] {
  return (dcc.credentialSubject?.conformityAssessment ?? []).flatMap((assessment: Json) =>
    (assessment.assessedProduct ?? []).map((item: Json) => item.product?.id),
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
