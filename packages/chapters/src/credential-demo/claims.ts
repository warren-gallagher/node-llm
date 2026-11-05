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
  /** The terms the data is under; public domain when not given. */
  licence?: DatasetLicence;
}

/** A dataset's licence, and why the rights auditor finds it permits training. */
export interface DatasetLicence {
  id: string;
  name: string;
  /** The auditor's reason the licence permits training, one sentence. */
  rationale: string;
}

const PUBLIC_DOMAIN_LICENCE: DatasetLicence = {
  id: "https://creativecommons.org/publicdomain/mark/1.0/",
  name: "Public Domain Mark 1.0",
  rationale: "The work is in the public domain, so it may be used to train models.",
};

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

/** A change made to an existing model to produce a new one, such as quantization. */
export interface DerivationFacts {
  derivationId: string;
  /** ISO date-time the derivation started. */
  eventDate: string;
  /** What was done, for example "Quantization (int8)". */
  processType: string;
  /** The method and its settings, enough to rerun a deterministic derivation. */
  hyperparameters: object;
  code: { repository: string; commit: string };
  /** Unknown for a derivation recorded after the fact. */
  durationSeconds?: number;
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
  const licence = dataset.licence ?? PUBLIC_DOMAIN_LICENCE;

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
          description: licence.rationale,
          assessmentCriteria: [rightsCriterion(ctx)],
          assessmentDate: ctx.date,
          assessedPerformance: [{ score: { code: "permitted", definition: licence.name } }],
          assessedProduct: [
            { product: productRef(dataset.title, dataset.file), idVerifiedByCAB: true },
          ],
          conformityTopic: [topics(ctx).trainingRights],
          evidence: [
            {
              linkURL: dataset.sourceUrl,
              linkName: "Source data",
              mediaType: dataset.file.mediaType,
            },
            ...(dataset.licence
              ? [{ linkURL: licence.id, linkName: licence.name, mediaType: "text/plain" }]
              : []),
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
        mediaType: dataset.file.mediaType,
        encoding: "utf-8",
        sourceURL: dataset.sourceUrl,
        licence: dataset.licence
          ? { id: dataset.licence.id, name: dataset.licence.name }
          : PUBLIC_DOMAIN,
        tokenCounts: [{ tokenizer: dataset.tokenizer, count: dataset.tokenCount }],
        files: [extensionFile(dataset.file)],
      },
      performanceClaim: [trainingUseClaim(ctx, publisher, rights, dataset)],
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

/** Model Publisher's record of deriving one model from another, by the hashes of both. */
export function derivationDte(
  ctx: DemoContext,
  parent: Pick<ModelFacts, "name" | "weights" | "parameterCount">,
  derived: ModelFacts,
  derivation: DerivationFacts,
  refs: {
    parentPassport: IssuedCredential;
    /** For a fine-tune: the data it trained on, with its passport and rights verification. */
    data?: { dataset: DatasetFacts; passport: IssuedCredential; rights: IssuedCredential };
  },
): SignRequest {
  const builder = ctx.parties.modelPublisher;
  const data = refs.data;
  return {
    name: `${derivation.processType}: ${derived.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: [
      {
        id: home(builder, `events/derivation/${derivation.derivationId}`),
        name: `${derivation.processType} of ${parent.name}`,
        description: `Derives ${derived.name} from ${parent.name}`,
        eventDate: derivation.eventDate,
        activityType: COMMISSIONING,
        inputProduct: [
          {
            product: productRef(parent.name, parent.weights),
            quantity: { value: parent.parameterCount, unit: "C62" },
            // The parent model is read, not used up.
            disposition: "active",
          },
          ...(data
            ? [
                {
                  product: productRef(data.dataset.title, data.dataset.file),
                  quantity: { value: data.dataset.file.byteLength, unit: "AD" },
                  disposition: "active",
                },
              ]
            : []),
        ],
        outputProduct: [
          {
            product: productRef(derived.name, derived.weights),
            quantity: { value: derived.parameterCount, unit: "C62" },
            disposition: "new",
          },
        ],
        madeAtFacility: facility(ctx, "modelPublisher"),
        relatedDocument: [
          link(refs.parentPassport, `Model passport: ${parent.name}`),
          ...(data
            ? [
                link(data.passport, `Dataset passport: ${data.dataset.title}`),
                link(data.rights, `Training rights: ${data.dataset.title}`),
              ]
            : []),
        ],
        trainingRun: {
          type: ["TrainingRun"],
          processType: derivation.processType,
          modelConfig: derived.modelConfig,
          hyperparameters: derivation.hyperparameters,
          code: derivation.code,
          durationSeconds: derivation.durationSeconds,
          quantityBasis: [
            { productId: niUri(parent.weights.sha256), counts: "parameter" },
            { productId: niUri(derived.weights.sha256), counts: "parameter" },
          ],
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
    /** The event that produced the model: its training run, or the derivation. */
    trainingEvent: IssuedCredential;
    evaluation: IssuedCredential;
    /**
     * For a derived model: the model it was derived from, what the derivation was, and for a
     * fine-tune the data it added, whose rights the passport claims as well.
     */
    derivedFrom?: {
      parentName: string;
      parentPassport: IssuedCredential;
      processType: string;
      data?: { dataset: DatasetFacts; rights: IssuedCredential };
    };
  },
): SignRequest {
  const builder = ctx.parties.modelPublisher;
  const party = PARTIES.modelPublisher;
  const parent = refs.derivedFrom;
  // A derived model links its derivation and its parent; the dataset is reached through the parent.
  const lineage = parent
    ? [
        link(refs.trainingEvent, `${parent.processType} of ${parent.parentName}`),
        link(parent.parentPassport, `Model passport: ${parent.parentName}`),
      ]
    : [
        link(refs.trainingEvent, `Training run of ${model.name}`),
        link(refs.datasetPassport, `Dataset passport: ${dataset.title}`),
      ];

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
      materialProvenance: materials(ctx, dataset, parent?.data?.dataset),
      characteristics: {
        type: ["ModelCharacteristics"],
        architecture: model.architecture,
        parameterCount: model.parameterCount,
        config: model.modelConfig,
        tokenizer: dataset.tokenizer,
        files: [extensionFile(model.weights), extensionFile(model.config)],
      },
      relatedDocument: lineage,
      performanceClaim: [
        {
          ...trainingUseClaim(ctx, builder, refs.rights, dataset),
          description: `Carried over from the passport of ${parent ? parent.parentName : dataset.title}`,
        },
        ...(parent?.data
          ? [
              {
                ...trainingUseClaim(ctx, builder, parent.data.rights, parent.data.dataset),
                description: `The data ${parent.parentName} was fine-tuned on`,
              },
            ]
          : []),
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
// Phase 3: retrieval
// ============================================================================

/** A model from outside the demo: its files, the first being its weights, and where they came from. */
export interface ImportedModel {
  name: string;
  description: string;
  architecture: string;
  parameterCount: number;
  modelConfig: object;
  files: FileDigest[];
  source: {
    /** The repository the files were downloaded from, and the revision. */
    repository: string;
    revision: string;
    /** The SHA-256 the repository publishes for the weights, when it publishes one. */
    publishedSha256?: string;
    licence: { id: string; name: string };
    /** Who produced the files, and where, as far as the publisher says. */
    producer: { name: string; country: { countryCode: string; countryName: string } };
  };
}

/**
 * Model Publisher's passport for a model it imported rather than built: it attests where the files
 * came from and their hashes, not how the model was made.
 */
export function importedModelDpp(ctx: DemoContext, model: ImportedModel): SignRequest {
  const [weights] = model.files;
  if (!weights) throw new Error(`${model.name} has no files`);
  const verified = model.source.publishedSha256 === weights.sha256;
  return {
    name: `Model passport: ${model.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(model.name, weights),
      description:
        `${model.description} Imported from ${model.source.repository} at revision ` +
        `${model.source.revision}${verified ? ", matching the SHA-256 it publishes" : ""}. ` +
        "This passport attests where the files came from, not how the model was made.",
      productCategory: [category(ctx, "model")],
      // The repository that published the files stands in for the producer's facility.
      producedAtFacility: { id: model.source.repository, name: model.source.producer.name },
      countryOfProduction: model.source.producer.country,
      characteristics: {
        type: ["ModelCharacteristics"],
        architecture: model.architecture,
        parameterCount: model.parameterCount,
        config: model.modelConfig,
        files: model.files.map(extensionFile),
        source: model.source,
        licence: model.source.licence,
      },
    },
  };
}

/** The instruction-tuned model that answers questions: fine-tuned from an imported model. */
export function instructionModelDpp(
  ctx: DemoContext,
  model: ModelFacts,
  refs: {
    fineTuning: IssuedCredential;
    parentName: string;
    parentPassport: IssuedCredential;
    data: { dataset: DatasetFacts; rights: IssuedCredential };
  },
): SignRequest {
  const builder = ctx.parties.modelPublisher;
  return {
    name: `Model passport: ${model.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(model.name, model.weights),
      description: model.description,
      productCategory: [category(ctx, "model")],
      producedAtFacility: facility(ctx, "modelPublisher"),
      productionDate: ctx.date,
      countryOfProduction: PARTIES.modelPublisher.country,
      characteristics: {
        type: ["ModelCharacteristics"],
        architecture: model.architecture,
        parameterCount: model.parameterCount,
        config: model.modelConfig,
        files: [extensionFile(model.weights), extensionFile(model.config)],
      },
      relatedDocument: [
        link(refs.fineTuning, `Fine-tuning of ${refs.parentName}`),
        link(refs.parentPassport, `Model passport: ${refs.parentName}`),
      ],
      performanceClaim: [
        {
          ...trainingUseClaim(ctx, builder, refs.data.rights, refs.data.dataset),
          description: `The data ${refs.parentName} was fine-tuned on`,
        },
      ],
    },
  };
}

/** A document collection for retrieval: one file, with each passage listed by its digest. */
export interface CollectionFacts {
  title: string;
  description: string;
  file: FileDigest;
  passages: { name: string; mediaType: string; byteLength: number; digestMultibase: string }[];
  licence: { id: string; name: string };
}

/** Book Publisher's passport for the document collection the retrieval system may index. */
export function collectionDpp(ctx: DemoContext, collection: CollectionFacts): SignRequest {
  return {
    name: `Collection passport: ${collection.title}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(collection.title, collection.file),
      description: collection.description,
      productCategory: [category(ctx, "dataset")],
      producedAtFacility: facility(ctx, "bookPublisher"),
      productionDate: ctx.date,
      countryOfProduction: PARTIES.bookPublisher.country,
      characteristics: {
        type: ["DatasetCharacteristics"],
        byteLength: collection.file.byteLength,
        mediaType: collection.file.mediaType,
        encoding: "utf-8",
        licence: collection.licence,
        files: [extensionFile(collection.file), ...collection.passages],
      },
    },
  };
}

/** Model Deployment Inc.'s record of embedding the admitted passages into an index. */
export function indexingDte(
  ctx: DemoContext,
  collection: CollectionFacts,
  embedder: ImportedModel,
  index: { name: string; file: FileDigest; passageCount: number },
  run: DerivationFacts,
  refs: { collectionPassport: IssuedCredential; embedderPassport: IssuedCredential },
): SignRequest {
  const deployer = ctx.parties.modelDeployer;
  const [embedderWeights] = embedder.files;
  if (!embedderWeights) throw new Error(`${embedder.name} has no files`);
  return {
    name: `Indexing: ${index.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: [
      {
        id: home(deployer, `events/indexing/${run.derivationId}`),
        name: `Indexing of ${collection.title}`,
        description: `Embeds ${index.passageCount} admitted passages with ${embedder.name}`,
        eventDate: run.eventDate,
        activityType: COMMISSIONING,
        inputProduct: [
          {
            product: productRef(collection.title, collection.file),
            quantity: { value: collection.file.byteLength, unit: "AD" },
            disposition: "active",
          },
          {
            product: productRef(embedder.name, embedderWeights),
            quantity: { value: embedder.parameterCount, unit: "C62" },
            disposition: "active",
          },
        ],
        outputProduct: [
          {
            product: productRef(index.name, index.file),
            quantity: { value: index.file.byteLength, unit: "AD" },
            disposition: "new",
          },
        ],
        madeAtFacility: facility(ctx, "modelDeployer"),
        relatedDocument: [
          link(refs.collectionPassport, `Collection passport: ${collection.title}`),
          link(refs.embedderPassport, `Model passport: ${embedder.name}`),
        ],
        trainingRun: {
          type: ["TrainingRun"],
          processType: run.processType,
          hyperparameters: run.hyperparameters,
          code: run.code,
          durationSeconds: run.durationSeconds,
        },
      },
    ],
  };
}

/** Model Deployment Inc.'s passport for the index its retrieval system searches. */
export function indexDpp(
  ctx: DemoContext,
  index: { name: string; description: string; file: FileDigest },
  refs: {
    indexing: IssuedCredential;
    collectionPassport: IssuedCredential;
    collectionTitle: string;
  },
): SignRequest {
  return {
    name: `Index passport: ${index.name}`,
    extensions: [extensionContext(ctx.apiUrl)],
    credentialSubject: {
      ...productIdentity(index.name, index.file),
      description: index.description,
      productCategory: [category(ctx, "dataset")],
      producedAtFacility: facility(ctx, "modelDeployer"),
      productionDate: ctx.date,
      countryOfProduction: PARTIES.modelDeployer.country,
      characteristics: {
        type: ["DatasetCharacteristics"],
        byteLength: index.file.byteLength,
        mediaType: index.file.mediaType,
        files: [extensionFile(index.file)],
      },
      relatedDocument: [
        link(refs.indexing, `Indexing of ${refs.collectionTitle}`),
        link(refs.collectionPassport, `Collection passport: ${refs.collectionTitle}`),
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

function facility(ctx: DemoContext, key: "bookPublisher" | "modelPublisher" | "modelDeployer") {
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

/**
 * The data a model was made from. UNTP requires a mass fraction for each material; data has no
 * mass, so with fine-tuning data as well, each dataset's fraction is its share of the two
 * datasets' token counts (rounded so the two add up to 1).
 */
function materials(ctx: DemoContext, dataset: DatasetFacts, fineTuning?: DatasetFacts) {
  if (!fineTuning) return [{ ...material(ctx, dataset), massFraction: 1 }];
  const share =
    Math.round((dataset.tokenCount / (dataset.tokenCount + fineTuning.tokenCount)) * 1e4) / 1e4;
  return [
    { ...material(ctx, dataset), massFraction: share },
    { ...material(ctx, fineTuning), massFraction: Math.round((1 - share) * 1e4) / 1e4 },
  ];
}

function material(ctx: DemoContext, data: DatasetFacts) {
  return {
    name: data.title,
    originCountry: { countryCode: "US", countryName: "United States of America" },
    materialType: category(ctx, "dataset"),
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

function trainingUseClaim(
  ctx: DemoContext,
  claimant: Identity,
  rights: IssuedCredential,
  dataset: DatasetFacts,
) {
  return {
    id: home(claimant, `claims/training-use-permitted/${dataset.file.sha256}`),
    name: `Training use permitted: ${dataset.title}`,
    referenceCriteria: [rightsCriterion(ctx)],
    claimDate: ctx.date,
    claimedPerformance: [{ score: { code: "permitted" } }],
    conformityTopic: [topics(ctx).trainingRights],
    evidence: [link(rights, "Training-rights verification")],
  };
}
