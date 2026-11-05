import type {
  DatasetFacts,
  DemoContext,
  DerivationFacts,
  EvaluationFacts,
  ModelFacts,
  TrainingFacts,
} from "../../src/credential-demo/claims";
import type { IssuedCredential } from "../../src/credential-demo/state";

/** Sample inputs for the claim builders, shaped like a real run's. */

export const API_URL = "https://realtime.example.com";

export const CTX: DemoContext = {
  apiUrl: API_URL,
  date: "2026-11-04",
  parties: {
    bookPublisher: { did: "did:web:bp.dids.example.com", name: "Verdict Demo: Book Publisher" },
    bookAuditor: { did: "did:web:ba.dids.example.com", name: "Verdict Demo: Book Auditor" },
    modelPublisher: { did: "did:web:mp.dids.example.com", name: "Verdict Demo: Model Publisher" },
    modelAuditor: { did: "did:web:ma.dids.example.com", name: "Verdict Demo: Model Auditor" },
    evaluatorsAssociation: {
      did: "did:web:ea.dids.example.com",
      name: "Verdict Demo: Evaluators Association",
    },
  },
};

const TOKENIZER = {
  name: "r50k_base",
  package: "gpt-tokenizer",
  version: "3.4.0",
  digestMultibase: "uEiAwbNJ_A8GnFOynEIDj1mt9wEKr6sJYy0TBnHqe2YON2TA",
};

export const DATASET: DatasetFacts = {
  title: "The Verdict (Edith Wharton, 1908)",
  description: "The training corpus.",
  file: {
    name: "the-verdict.txt",
    mediaType: "text/plain",
    byteLength: 20479,
    sha256: "b41e41a68f0398a3154ae69e2e4c0e2694e17fe0d66730536837f1b01935b31f",
  },
  sourceUrl: "https://example.com/the-verdict.txt",
  tokenCount: 5145,
  tokenizer: TOKENIZER,
};

export const MODEL: ModelFacts = {
  name: "Verdict GPT 2026-11-04T12-00-00Z",
  description: "A small model.",
  architecture: "GPT-2",
  parameterCount: 13_313_152,
  modelConfig: { vocabSize: 50257, contextLength: 64, embDim: 128, numHeads: 4, numLayers: 2 },
  weights: {
    name: "weights.bin",
    mediaType: "application/octet-stream",
    byteLength: 53_252_608,
    sha256: "de24e3f3".padEnd(64, "0"),
  },
  config: {
    name: "config.json",
    mediaType: "application/json",
    byteLength: 1234,
    sha256: "c0ffee".padEnd(64, "1"),
  },
};

/** The model above, quantized to int8. */
export const DERIVED: ModelFacts = {
  ...MODEL,
  name: `${MODEL.name} (int8)`,
  architecture: "GPT-2 (int8 weights)",
  weights: {
    name: "weights.q8.bin",
    mediaType: "application/octet-stream",
    byteLength: 13_313_276,
    sha256: "2116a219".padEnd(64, "0"),
  },
};

export const DERIVATION: DerivationFacts = {
  derivationId: "2026-11-05T12-00-00Z-int8",
  eventDate: "2026-11-05T12:00:00.000Z",
  processType: "Quantization (int8)",
  hyperparameters: { method: "symmetric int8, one scale per tensor", bits: 8 },
  code: { repository: "https://github.com/example/node-llm", commit: "e".repeat(40) },
  durationSeconds: 0.6,
};

export const TRAINING: TrainingFacts = {
  runId: "2026-11-04T12-00-00Z",
  eventDate: "2026-11-04T12:00:00.000Z",
  durationSeconds: 23.4,
  epochs: 10,
  hyperparameters: { optimizer: "AdamW", learningRate: 0.001, weightDecay: 0.1, batchSize: 8 },
  split: { train: 0.9, validation: 0.1 },
  finalLoss: { train: 5.1, validation: 6.37 },
  code: { repository: "https://github.com/example/node-llm", commit: "d0fc0cb".padEnd(40, "0") },
  reproduction: {
    seed: 42,
    prng: "mulberry32 replacing Math.random",
    threads: 8,
    runtime: { node: "v22.23.4", "tfjs-node": "4.22.0", libtensorflow: "2.7.0" },
    cpu: "Apple M2",
  },
};

export const EVALUATION: EvaluationFacts = {
  validationLoss: 6.37,
  harness: {
    name: "@node-llm/core evaluateLoss",
    version: "0.1.0",
    repository: TRAINING.code.repository,
    commit: TRAINING.code.commit,
  },
  split: "validation: the last 10% of the text's characters",
  metric: "cross-entropy loss (nats per token)",
};

export function issued(type: IssuedCredential["type"], name: string): IssuedCredential {
  return {
    credentialId: `https://dids.example.com/credentials/${type}/2026/${name}`,
    type,
    issuer: "did:web:dids.example.com",
    digestMultibase: `uEiA${name}`,
    validFrom: "2026-11-04T00:00:00Z",
    validUntil: "2027-11-04T00:00:00Z",
  };
}
