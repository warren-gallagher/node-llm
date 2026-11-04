import type { SignRequest } from "./client";
import { digestMultibase, type FileDigest, niUri } from "./hashing";
import { type AccreditedKey, didWebHost, PARTIES, type PartyKey } from "./parties";
import type { IssuedCredential } from "./state";

/**
 * Builders for the demo's seven sign requests, UNTP 0.7.0 with the draft AI model extension.
 * They are pure: every value comes from the arguments. The mapping from pipeline facts to UNTP
 * fields is described in docs/credential-demo/credential-demo.md ("Mapping onto the UNTP schemas").
 *
 * Products are identified by the hash of their content, so a credential can name a model before
 * the model's own passport exists. Links between credentials carry the digest of the linked JWT.
 */

const LINK_TYPES = "https://test.uncefact.org/vocabulary/linkTypes/";
const EXTENSION_PATH = "/untp/ai-model/0.1.0/";

const NI_SCHEME = {
  id: "https://www.rfc-editor.org/rfc/rfc6920",
  name: "Named Information (ni) URI, RFC 6920",
};
const COMMISSIONING = {
  code: "commissioning",
  name: "Commissioning",
  schemeId: "https://ref.gs1.org/cbv/BizStep",
  schemeName: "GS1 CBV Business Step",
};
const PUBLIC_DOMAIN = {
  id: "https://creativecommons.org/publicdomain/mark/1.0/",
  name: "Public Domain Mark 1.0",
};

export interface Identity {
  did: string;
  name: string;
}

export interface DemoContext {
  /** The platform's public URL; the extension files are served from it. */
  apiUrl: string;
  parties: Record<PartyKey, Identity>;
  /** The issue date, YYYY-MM-DD. */
  date: string;
}

export interface Tokenizer {
  name: string;
  package: string;
  version: string;
  /** Digest of the encoding's vocabulary file. */
  digestMultibase: string;
}

export interface DatasetFacts {
  title: string;
  description: string;
  file: FileDigest;
  sourceUrl: string;
  tokenCount: number;
  tokenizer: Tokenizer;
}

export interface ModelFacts {
  name: string;
  description: string;
  architecture: string;
  parameterCount: number;
  modelConfig: object;
  weights: FileDigest;
  config: FileDigest;
}

export interface TrainingFacts {
  runId: string;
  /** ISO date-time the run started. */
  eventDate: string;
  durationSeconds: number;
  epochs: number;
  hyperparameters: object;
  split: { train: number; validation: number };
  finalLoss: { train: number; validation: number };
  code: { repository: string; commit: string };
  reproduction: { seed: number; prng: string; threads: number; runtime: object; cpu: string };
}

export interface EvaluationFacts {
  validationLoss: number;
  harness: { name: string; version: string; repository: string; commit: string };
  split: string;
  metric: string;
}

export function extensionContext(apiUrl: string): string {
  return `${apiUrl}${EXTENSION_PATH}context.jsonld`;
}

/** A UNTP link to an issued credential, with the digest of its JWT. */
export function link(credential: IssuedCredential, linkName: string) {
  return {
    linkURL: credential.credentialId,
    linkName,
    linkType: `${LINK_TYPES}${credential.type}`,
    digestMultibase: credential.digestMultibase,
  };
}

// ============================================================================
// Accreditation and rights (setup, and once per dataset)
// ============================================================================

const ACCREDITATION_SCOPE: Record<AccreditedKey, { slug: string; name: string }> = {
  bookAuditor: { slug: "rights-verification", name: "Verification of training-use rights" },
  modelAuditor: { slug: "model-evaluation", name: "Independent model evaluation" },
};

/** Evaluators Association accredits one of the evaluators. */
export function accreditationDcc(ctx: DemoContext, auditor: AccreditedKey): SignRequest {
  const association = ctx.parties.evaluatorsAssociation;
  const evaluator = partyRef(ctx.parties[auditor]);
  const scope = ACCREDITATION_SCOPE[auditor];
  const scheme = schemes(ctx).accreditation;
  const id = home(association, `accreditations/${scope.slug}`);

  return {
    name: `Accreditation: ${evaluator.name}`,
    credentialSubject: {
      id,
      name: `Accreditation of ${evaluator.name} for ${scope.name.toLowerCase()}`,
      // The accrediting body assesses against its own scheme.
      assessorLevel: "3rdParty",
      assessmentLevel: "scheme-self",
      attestationType: "certification",
      issuedToParty: evaluator,
      referenceScheme: scheme,
      conformityAssessment: [
        {
          id: `${id}#assessment`,
          name: scope.name,
          assessmentCriteria: [{ id: `${scheme.id}#${scope.slug}`, name: scope.name }],
          assessmentDate: ctx.date,
          assessedPerformance: [{ score: { code: "accredited" } }],
          assessedOrganisation: evaluator,
          conformityTopic: [topics(ctx).evaluatorCompetence],
          conformance: true,
        },
      ],
    },
  };
}

/** Book Auditor verifies that the dataset may be used for training. */
export function rightsDcc(
  ctx: DemoContext,
  dataset: DatasetFacts,
  accreditation: IssuedCredential,
): SignRequest {
  const auditor = ctx.parties.bookAuditor;
  const scheme = schemes(ctx).rights;
  const id = home(auditor, `attestations/training-rights/${dataset.file.sha256}`);

  return {
    name: `Training rights: ${dataset.title}`,
    credentialSubject: {
      id,
      name: `Training-use rights for ${dataset.title}`,
      assessorLevel: "3rdParty",
      assessmentLevel: "scheme-cab",
      attestationType: "verification",
      issuedToParty: partyRef(ctx.parties.bookPublisher),
      authorisation: [endorsement(ctx, accreditation)],
      referenceScheme: scheme,
      conformityAssessment: [
        {
          id: `${id}#assessment`,
          name: "Training-use rights",
          description: "The work is in the public domain, so it may be used to train models.",
          assessmentCriteria: [rightsCriterion(ctx)],
          assessmentDate: ctx.date,
          assessedPerformance: [
            { score: { code: "permitted", definition: `Public domain (${PUBLIC_DOMAIN.name})` } },
          ],
          assessedProduct: [
            { product: productRef(dataset.title, dataset.file), idVerifiedByCAB: true },
          ],
          conformityTopic: [topics(ctx).trainingRights],
          evidence: [
            { linkURL: dataset.sourceUrl, linkName: "Source text", mediaType: "text/plain" },
          ],
          conformance: true,
        },
      ],
    },
  };
}

// ============================================================================
// Pipeline credentials
// ============================================================================

/** Book Publisher's passport for the dataset, backed by the rights DCC. */
export function datasetDpp(
  ctx: DemoContext,
  dataset: DatasetFacts,
  rights: IssuedCredential,
): SignRequest {
  const publisher = ctx.parties.bookPublisher;
  const party = PARTIES.bookPublisher;

  return {
    name: `Dataset passport: ${dataset.title}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(dataset.title, dataset.file),
      description: dataset.description,
      productCategory: [category(ctx, "dataset")],
      producedAtFacility: facility(ctx, "bookPublisher"),
      productionDate: ctx.date,
      countryOfProduction: party.country,
      materialProvenance: [
        {
          name: dataset.title,
          originCountry: { countryCode: "US", countryName: "United States of America" },
          materialType: category(ctx, "dataset"),
          massFraction: 1,
        },
      ],
      characteristics: {
        type: ["DatasetCharacteristics"],
        byteLength: dataset.file.byteLength,
        mediaType: "text/plain",
        encoding: "utf-8",
        sourceURL: dataset.sourceUrl,
        licence: PUBLIC_DOMAIN,
        tokenCounts: [{ tokenizer: dataset.tokenizer, count: dataset.tokenCount }],
        files: [extensionFile(dataset.file)],
      },
      performanceClaim: [trainingUseClaim(ctx, publisher, rights)],
    },
  };
}

/** Model Publisher's record of the training run: the dataset in, the model's weights out. */
export function trainingDte(
  ctx: DemoContext,
  dataset: DatasetFacts,
  model: ModelFacts,
  run: TrainingFacts,
  refs: { datasetPassport: IssuedCredential; rights: IssuedCredential },
): SignRequest {
  const builder = ctx.parties.modelPublisher;
  const modelId = niUri(model.weights.sha256);

  return {
    name: `Training run ${run.runId}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: [
      {
        id: home(builder, `events/training/${run.runId}`),
        name: `Pretraining of ${model.name}`,
        description: `Trains ${model.name} on ${dataset.title}`,
        eventDate: run.eventDate,
        activityType: COMMISSIONING,
        inputProduct: [
          {
            product: productRef(dataset.title, dataset.file),
            quantity: { value: dataset.file.byteLength, unit: "AD" },
            // Training reads the dataset; it isn't used up.
            disposition: "active",
          },
        ],
        outputProduct: [
          {
            product: productRef(model.name, model.weights),
            quantity: { value: model.parameterCount, unit: "C62" },
            disposition: "new",
          },
        ],
        madeAtFacility: facility(ctx, "modelPublisher"),
        relatedDocument: [
          link(refs.datasetPassport, `Dataset passport: ${dataset.title}`),
          link(refs.rights, `Training rights: ${dataset.title}`),
        ],
        trainingRun: {
          type: ["TrainingRun"],
          processType: "Pretraining (next-token prediction)",
          modelConfig: model.modelConfig,
          hyperparameters: run.hyperparameters,
          epochs: run.epochs,
          split: run.split,
          finalLoss: run.finalLoss,
          code: run.code,
          durationSeconds: run.durationSeconds,
          // C62 ("one") has no room to say what it counts, so the extension says it here.
          quantityBasis: [{ productId: modelId, counts: "parameter" }],
          reproduction: run.reproduction,
        },
      },
    ],
  };
}

/** Model Auditor's evaluation of the exact weights it hashed. */
export function evaluationDcc(
  ctx: DemoContext,
  dataset: DatasetFacts,
  model: ModelFacts,
  evaluation: EvaluationFacts,
  accreditation: IssuedCredential,
): SignRequest {
  const auditor = ctx.parties.modelAuditor;
  const scheme = schemes(ctx).evaluation;
  const id = home(auditor, `attestations/evaluation/${model.weights.sha256}`);

  return {
    name: `Evaluation: ${model.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      id,
      name: `Evaluation of ${model.name}`,
      assessorLevel: "3rdParty",
      assessmentLevel: "scheme-cab",
      attestationType: "testing",
      issuedToParty: partyRef(ctx.parties.modelPublisher),
      authorisation: [endorsement(ctx, accreditation)],
      referenceScheme: scheme,
      conformityAssessment: [
        {
          id: `${id}#validation-loss`,
          name: "Validation loss",
          assessmentCriteria: [lossCriterion(ctx)],
          assessmentDate: ctx.date,
          assessedPerformance: [lossPerformance(ctx, evaluation.validationLoss)],
          // The auditor hashed the weights it evaluated, rather than taking the builder's word.
          assessedProduct: [
            { product: productRef(model.name, model.weights), idVerifiedByCAB: true },
          ],
          conformityTopic: [topics(ctx).modelPerformance],
          conformance: true,
          evaluationRun: {
            type: ["EvaluationRun"],
            harness: evaluation.harness,
            datasetId: niUri(dataset.file.sha256),
            split: evaluation.split,
            metric: evaluation.metric,
          },
        },
      ],
    },
  };
}

/** Model Publisher's passport for the model, linking the whole chain. */
export function modelDpp(
  ctx: DemoContext,
  dataset: DatasetFacts,
  model: ModelFacts,
  evaluation: EvaluationFacts,
  refs: {
    datasetPassport: IssuedCredential;
    rights: IssuedCredential;
    trainingEvent: IssuedCredential;
    evaluation: IssuedCredential;
  },
): SignRequest {
  const builder = ctx.parties.modelPublisher;
  const party = PARTIES.modelPublisher;

  return {
    name: `Model passport: ${model.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(model.name, model.weights),
      description: model.description,
      productCategory: [category(ctx, "model")],
      producedAtFacility: facility(ctx, "modelPublisher"),
      productionDate: ctx.date,
      countryOfProduction: party.country,
      materialProvenance: [
        {
          name: dataset.title,
          originCountry: { countryCode: "US", countryName: "United States of America" },
          materialType: category(ctx, "dataset"),
          massFraction: 1,
        },
      ],
      characteristics: {
        type: ["ModelCharacteristics"],
        architecture: model.architecture,
        parameterCount: model.parameterCount,
        config: model.modelConfig,
        tokenizer: dataset.tokenizer,
        files: [extensionFile(model.weights), extensionFile(model.config)],
      },
      relatedDocument: [
        link(refs.trainingEvent, `Training run of ${model.name}`),
        link(refs.datasetPassport, `Dataset passport: ${dataset.title}`),
      ],
      performanceClaim: [
        {
          ...trainingUseClaim(ctx, builder, refs.rights),
          description: `Carried over from the passport of ${dataset.title}`,
        },
        {
          id: home(builder, `claims/validation-loss/${model.weights.sha256}`),
          name: "Validation loss",
          referenceCriteria: [lossCriterion(ctx)],
          claimDate: ctx.date,
          claimedPerformance: [lossPerformance(ctx, evaluation.validationLoss)],
          conformityTopic: [topics(ctx).modelPerformance],
          evidence: [link(refs.evaluation, `Evaluation: ${model.name}`)],
        },
      ],
    },
  };
}

// ============================================================================
// Shared pieces
// ============================================================================

/** An https URL on the party's own did:web host, for things the party names. */
function home(identity: Identity, path: string): string {
  return `https://${didWebHost(identity.did)}/${path}`;
}

function partyRef(identity: Identity) {
  return { id: identity.did, name: identity.name };
}

function facility(ctx: DemoContext, key: "bookPublisher" | "modelPublisher") {
  const site = PARTIES[key].facility;
  if (!site) throw new Error(`${key} has no facility`);
  return { id: home(ctx.parties[key], `facilities/${site.slug}`), name: site.name };
}

function category(ctx: DemoContext, code: "dataset" | "model") {
  return {
    code,
    name: code === "dataset" ? "Dataset" : "Machine learning model",
    schemeId: `${ctx.apiUrl}${EXTENSION_PATH}categories.json`,
    schemeName: "AI model extension product categories",
  };
}

/** The identifying fields of a product passport for one exact file. */
function productIdentity(name: string, file: FileDigest) {
  return { id: niUri(file.sha256), name, idScheme: NI_SCHEME, idGranularity: "item" };
}

/** A product named inside an event or an assessment. */
function productRef(name: string, file: FileDigest) {
  return { id: niUri(file.sha256), name, idGranularity: "item" };
}

function extensionFile(file: FileDigest) {
  return {
    name: file.name,
    mediaType: file.mediaType,
    byteLength: file.byteLength,
    digestMultibase: digestMultibase(file.sha256),
  };
}

/** The schemes Evaluators Association runs. */
function schemes(ctx: DemoContext) {
  const association = ctx.parties.evaluatorsAssociation;
  return {
    accreditation: {
      id: home(association, "schemes/evaluator-accreditation"),
      name: "Evaluators Association evaluator accreditation",
    },
    rights: {
      id: home(association, "schemes/training-rights"),
      name: "Evaluators Association training-rights verification",
    },
    evaluation: {
      id: home(association, "schemes/model-evaluation"),
      name: "Evaluators Association model evaluation",
    },
  };
}

function topics(ctx: DemoContext) {
  const association = ctx.parties.evaluatorsAssociation;
  return {
    evaluatorCompetence: {
      id: home(association, "topics/evaluator-competence"),
      name: "Evaluator competence",
    },
    trainingRights: {
      id: home(association, "topics/training-rights"),
      name: "Rights to use data for training",
    },
    modelPerformance: {
      id: home(association, "topics/model-performance"),
      name: "Model performance",
    },
  };
}

function rightsCriterion(ctx: DemoContext) {
  return {
    id: `${schemes(ctx).rights.id}#training-use`,
    name: "The work may be used to train models",
  };
}

function lossCriterion(ctx: DemoContext) {
  return {
    id: `${schemes(ctx).evaluation.id}#validation-loss`,
    name: "Cross-entropy loss on the held-out validation split",
  };
}

/** Loss in nats per token; Recommendation 20 has no such unit, so it is a plain number (C62). */
function lossPerformance(ctx: DemoContext, loss: number) {
  return {
    metric: {
      id: `${schemes(ctx).evaluation.id}#cross-entropy`,
      name: "Cross-entropy loss (nats per token)",
    },
    measure: { value: loss, unit: "C62" },
  };
}

function endorsement(ctx: DemoContext, accreditation: IssuedCredential) {
  const association = ctx.parties.evaluatorsAssociation;
  return {
    name: `Accreditation by ${association.name}`,
    issuingAuthority: partyRef(association),
    endorsementEvidence: link(accreditation, `Accreditation by ${association.name}`),
  };
}

function trainingUseClaim(ctx: DemoContext, claimant: Identity, rights: IssuedCredential) {
  return {
    id: home(claimant, "claims/training-use-permitted"),
    name: "Training use permitted",
    referenceCriteria: [rightsCriterion(ctx)],
    claimDate: ctx.date,
    claimedPerformance: [{ score: { code: "permitted" } }],
    conformityTopic: [topics(ctx).trainingRights],
    evidence: [link(rights, "Training-rights verification")],
  };
}
