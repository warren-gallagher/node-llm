import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Identity } from "./claims";
import type { CredentialType, SignedCredential } from "./client";
import { REPO_ROOT } from "./env";
import { digestMultibase, sha256Hex } from "./hashing";
import { type AccreditedKey, PARTIES, PARTY_KEYS, type PartyKey } from "./parties";

/**
 * The manifest, `checkpoints/verdict-demo/credentials.json` (git-ignored): the authorities, the
 * credentials issued so far, and each training run. Setup and later runs reuse what it holds.
 */

export const MANIFEST_PATH = fileURLToPath(
  new URL("checkpoints/verdict-demo/credentials.json", REPO_ROOT),
);

export interface AuthorityRecord {
  id: string;
  name: string;
  did: string;
}

/** A credential as the demo keeps it: where it is published, and the digest of its JWT. */
export interface IssuedCredential {
  credentialId: string;
  type: CredentialType;
  issuer: string;
  /** Multibase SHA-256 of the JWT, as links to this credential carry it. */
  digestMultibase: string;
  validFrom: string;
  validUntil: string;
}

export interface DatasetRecord {
  /** SHA-256 (hex) of the dataset file. The credentials are reused while it is unchanged. */
  sha256: string;
  rights: IssuedCredential;
  passport: IssuedCredential;
}

export interface RunRecord {
  runId: string;
  /** Relative to the repository root. */
  checkpoint: string;
  weightsSha256: string;
  configSha256: string;
  validationLoss: number;
  trainingEvent: IssuedCredential;
  evaluation: IssuedCredential;
  passport: IssuedCredential;
}

/** A model derived from a run's model, such as its quantized copy. */
export interface DerivationRecord {
  derivationId: string;
  kind: "quantization" | "fine-tuning";
  /** The run whose model this was derived from. */
  parentRunId: string;
  /** Relative to the repository root. */
  checkpoint: string;
  /** The file that holds the weights, in the checkpoint. */
  weightsFile: string;
  weightsSha256: string;
  configSha256: string;
  validationLoss: number;
  event: IssuedCredential;
  evaluation: IssuedCredential;
  passport: IssuedCredential;
}

export interface Manifest {
  /** The platform and tenant the records belong to. */
  apiUrl: string;
  tenantId: string;
  authorities: Partial<Record<PartyKey, AuthorityRecord>>;
  accreditations: Partial<Record<AccreditedKey, IssuedCredential>>;
  dataset?: DatasetRecord;
  /** The data the fine-tuned models were trained on (phase 2). */
  fineTuningDataset?: DatasetRecord;
  runs: RunRecord[];
  /** Models derived from the runs' models (phase 2). */
  derivations?: DerivationRecord[];
}

/**
 * The saved manifest, or a new one if there is none or it belongs to another platform or tenant.
 */
export function loadManifest(
  apiUrl: string,
  tenantId: string,
  path: string = MANIFEST_PATH,
): Manifest {
  const fresh: Manifest = { apiUrl, tenantId, authorities: {}, accreditations: {}, runs: [] };
  if (!existsSync(path)) return fresh;

  const saved = JSON.parse(readFileSync(path, "utf8")) as Manifest;
  if (saved.apiUrl !== apiUrl || saved.tenantId !== tenantId) {
    console.warn(`${path} is for another platform or tenant; starting a new manifest`);
    return fresh;
  }
  return saved;
}

export function saveManifest(manifest: Manifest, path: string = MANIFEST_PATH): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
}

/** Each party's DID and name, for the claim builders. Throws if setup hasn't run. */
export function partyIdentities(manifest: Manifest): Record<PartyKey, Identity> {
  const identities = {} as Record<PartyKey, Identity>;
  for (const key of PARTY_KEYS) {
    const record = manifest.authorities[key];
    if (!record) throw new Error(`no authority for ${PARTIES[key].name}: run pnpm demo:setup`);
    identities[key] = { did: record.did, name: record.name };
  }
  return identities;
}

export function toIssued(signed: SignedCredential): IssuedCredential {
  return {
    credentialId: signed.credentialId,
    type: signed.type,
    issuer: signed.issuer,
    digestMultibase: digestMultibase(sha256Hex(signed.jwt)),
    validFrom: signed.validFrom,
    validUntil: signed.validUntil,
  };
}
