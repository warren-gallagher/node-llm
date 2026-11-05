/**
 * The demo's parties. Each issuing party is an authority on the platform, with its own did:web and
 * signing key. Names carry a prefix so that a lookup by name in a shared tenant can't match an
 * unrelated authority. Model Deployment Inc. is the buyer in phases 1 and 2, where it only verifies;
 * in phase 3 it runs the retrieval system, and issues its index's credentials and answer records.
 */

export const NAME_PREFIX = "Verdict Demo: ";

export type PartyKey =
  | "bookPublisher"
  | "bookAuditor"
  | "modelPublisher"
  | "modelAuditor"
  | "evaluatorsAssociation"
  | "modelDeployer";

/** The two evaluators, each accredited by Evaluators Association. */
export type AccreditedKey = "bookAuditor" | "modelAuditor";

export interface Country {
  countryCode: string;
  countryName: string;
}

export interface Party {
  role: string;
  name: string;
  description: string;
  /** Where the party produces things; only the producers need one. */
  facility?: { slug: string; name: string };
  country?: Country;
}

const US: Country = { countryCode: "US", countryName: "United States of America" };
const CA: Country = { countryCode: "CA", countryName: "Canada" };

export const PARTIES: Record<PartyKey, Party> = {
  bookPublisher: {
    role: "Data curator",
    name: `${NAME_PREFIX}Book Publisher`,
    description: "Prepares and publishes the training corpus. (Credential demo)",
    facility: { slug: "corpus-preparation", name: "Book Publisher corpus preparation" },
    country: US,
  },
  bookAuditor: {
    role: "Rights evaluator",
    name: `${NAME_PREFIX}Book Auditor`,
    description: "Verifies that a work may be used for training. (Credential demo)",
  },
  modelPublisher: {
    role: "Model builder",
    name: `${NAME_PREFIX}Model Publisher`,
    description: "Trains and publishes the model. (Credential demo)",
    facility: { slug: "training-workstation", name: "Model Publisher training workstation" },
    country: CA,
  },
  modelAuditor: {
    role: "Model evaluator",
    name: `${NAME_PREFIX}Model Auditor`,
    description: "Evaluates models independently of their builders. (Credential demo)",
  },
  evaluatorsAssociation: {
    role: "Accreditor",
    name: `${NAME_PREFIX}Evaluators Association`,
    description: "Accredits the evaluators and runs the evaluation schemes. (Credential demo)",
  },
  modelDeployer: {
    role: "Model deployer",
    name: `${NAME_PREFIX}Model Deployment Inc.`,
    description:
      "Deploys models, runs the retrieval system and answers questions. (Credential demo)",
    facility: { slug: "inference-service", name: "Model Deployment Inc. inference service" },
    country: CA,
  },
};

export const PARTY_KEYS = Object.keys(PARTIES) as PartyKey[];

/**
 * The host part of a did:web, which is also the host to tunnel to the did:web server.
 * `did:web:example.com%3A8443:user:alice` -> `example.com:8443`.
 */
export function didWebHost(did: string): string {
  const [scheme, method, host] = did.split(":");
  if (scheme !== "did" || method !== "web" || !host) {
    throw new Error(`not a did:web: ${did}`);
  }
  return decodeURIComponent(host);
}

/** Where a did:web's DID document is served, per the did:web method specification. */
export function didDocumentUrl(did: string): string {
  const path = did.split(":").slice(3).map(decodeURIComponent);
  const base = `https://${didWebHost(did)}`;
  return path.length === 0 ? `${base}/.well-known/did.json` : `${base}/${path.join("/")}/did.json`;
}
