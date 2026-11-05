import { describe, expect, it } from "vitest";
import {
  accreditationDcc,
  datasetDpp,
  derivationDte,
  evaluationDcc,
  extensionContext,
  modelDpp,
  rightsDcc,
  trainingDte,
} from "../../src/credential-demo/claims";
import { niUri } from "../../src/credential-demo/hashing";
import {
  API_URL,
  CTX,
  DATASET,
  DERIVATION,
  DERIVED,
  EVALUATION,
  issued,
  MODEL,
  TRAINING,
} from "./fixtures";

const LINK_TYPES = "https://test.uncefact.org/vocabulary/linkTypes/";
const DATASET_ID = niUri(DATASET.file.sha256);
const MODEL_ID = niUri(MODEL.weights.sha256);

const accreditation = issued("dcc", "accreditation");
const rights = issued("dcc", "rights");
const datasetPassport = issued("dpp", "dataset");
const trainingEvent = issued("dte", "training");
const evaluation = issued("dcc", "evaluation");

// The builders return plain objects; tests read them loosely.
// biome-ignore lint/suspicious/noExplicitAny: test access into built claims
type Loose = any;

/** Every `id` and link URL in a request, which UNTP requires to be URIs. */
function collectUris(value: unknown, found: string[] = []): string[] {
  if (Array.isArray(value)) {
    for (const item of value) collectUris(item, found);
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (["id", "linkURL", "schemeId", "productId", "datasetId"].includes(key)) {
        found.push(item as string);
      } else {
        collectUris(item, found);
      }
    }
  }
  return found;
}

function linkTo(credential: typeof rights) {
  return expect.objectContaining({
    linkURL: credential.credentialId,
    linkType: `${LINK_TYPES}${credential.type}`,
    digestMultibase: credential.digestMultibase,
  });
}

describe("credential demo claims", () => {
  const requests = {
    accreditation: accreditationDcc(CTX, "modelAuditor"),
    rights: rightsDcc(CTX, DATASET, accreditation),
    datasetPassport: datasetDpp(CTX, DATASET, rights),
    trainingEvent: trainingDte(CTX, DATASET, MODEL, TRAINING, { datasetPassport, rights }),
    evaluation: evaluationDcc(CTX, DATASET, MODEL, EVALUATION, accreditation),
    modelPassport: modelDpp(CTX, DATASET, MODEL, EVALUATION, {
      datasetPassport,
      rights,
      trainingEvent,
      evaluation,
    }),
  };

  it("uses only URIs for ids and links", () => {
    for (const request of Object.values(requests)) {
      for (const uri of collectUris(request.credentialSubject)) {
        expect(URL.canParse(uri), uri).toBe(true);
      }
    }
  });

  it("names the AI model extension where it is used", () => {
    expect(requests.accreditation.extensions).toBeUndefined();
    expect(requests.rights.extensions).toBeUndefined();
    for (const key of [
      "datasetPassport",
      "trainingEvent",
      "evaluation",
      "modelPassport",
    ] as const) {
      expect(requests[key].extensions).toEqual([extensionContext(API_URL)]);
    }
    expect(extensionContext(API_URL)).toBe(`${API_URL}/untp/ai-model/0.1.0/context.jsonld`);
  });

  it("accredits the evaluator as an organisation", () => {
    const subject: Loose = requests.accreditation.credentialSubject;
    const modelAuditor = { id: CTX.parties.modelAuditor.did, name: CTX.parties.modelAuditor.name };
    expect(subject.issuedToParty).toEqual(modelAuditor);
    expect(subject.conformityAssessment[0].assessedOrganisation).toEqual(modelAuditor);
    expect(subject.referenceScheme.id).toBe(
      "https://ea.dids.example.com/schemes/evaluator-accreditation",
    );
  });

  it("verifies the dataset's rights under the auditor's accreditation", () => {
    const subject: Loose = requests.rights.credentialSubject;
    expect(subject.issuedToParty.id).toBe(CTX.parties.bookPublisher.did);
    expect(subject.authorisation[0].issuingAuthority.id).toBe(
      CTX.parties.evaluatorsAssociation.did,
    );
    expect(subject.authorisation[0].endorsementEvidence).toEqual(linkTo(accreditation));
    expect(subject.conformityAssessment[0].assessedProduct[0]).toEqual({
      product: { id: DATASET_ID, name: DATASET.title, idGranularity: "item" },
      idVerifiedByCAB: true,
    });
  });

  it("identifies the dataset by its hash and backs its claim with the rights DCC", () => {
    const subject: Loose = requests.datasetPassport.credentialSubject;
    expect(subject.id).toBe(DATASET_ID);
    expect(subject.idGranularity).toBe("item");
    expect(subject.productCategory[0].code).toBe("dataset");
    expect(subject.characteristics).toMatchObject({
      type: ["DatasetCharacteristics"],
      byteLength: 20479,
      tokenCounts: [{ tokenizer: DATASET.tokenizer, count: 5145 }],
    });
    expect(subject.performanceClaim[0].evidence).toEqual([linkTo(rights)]);
  });

  it("records training from the dataset's hash to the weights' hash", () => {
    const [event]: Loose = requests.trainingEvent.credentialSubject;
    expect(event.inputProduct[0]).toEqual({
      product: { id: DATASET_ID, name: DATASET.title, idGranularity: "item" },
      quantity: { value: 20479, unit: "AD" },
      disposition: "active",
    });
    expect(event.outputProduct[0].product.id).toBe(MODEL_ID);
    expect(event.outputProduct[0].quantity).toEqual({ value: MODEL.parameterCount, unit: "C62" });
    expect(event.relatedDocument).toEqual([linkTo(datasetPassport), linkTo(rights)]);
    expect(event.trainingRun.quantityBasis).toEqual([{ productId: MODEL_ID, counts: "parameter" }]);
    expect(event.trainingRun.reproduction.seed).toBe(42);
  });

  it("evaluates the exact weights under the auditor's accreditation", () => {
    const subject: Loose = requests.evaluation.credentialSubject;
    const assessment = subject.conformityAssessment[0];
    expect(subject.attestationType).toBe("testing");
    expect(subject.authorisation[0].endorsementEvidence).toEqual(linkTo(accreditation));
    expect(assessment.assessedProduct[0].product.id).toBe(MODEL_ID);
    expect(assessment.assessedPerformance[0].measure).toEqual({ value: 6.37, unit: "C62" });
    expect(assessment.evaluationRun).toMatchObject({
      type: ["EvaluationRun"],
      datasetId: DATASET_ID,
    });
  });

  it("links the model passport to the whole chain", () => {
    const subject: Loose = requests.modelPassport.credentialSubject;
    expect(subject.id).toBe(MODEL_ID);
    expect(subject.productCategory[0].code).toBe("model");
    expect(subject.characteristics.files.map((file: Loose) => file.name)).toEqual([
      "weights.bin",
      "config.json",
    ]);
    expect(subject.relatedDocument).toEqual([linkTo(trainingEvent), linkTo(datasetPassport)]);
    const [trainingUse, loss] = subject.performanceClaim;
    expect(trainingUse.evidence).toEqual([linkTo(rights)]);
    expect(loss.evidence).toEqual([linkTo(evaluation)]);
    expect(loss.claimedPerformance[0].measure.value).toBe(EVALUATION.validationLoss);
  });
});

describe("credential demo claims for a derived model", () => {
  const parentPassport = issued("dpp", "parent");
  const derivationEvent = issued("dte", "derivation");
  const derivedEvaluation = issued("dcc", "derived-evaluation");
  const event: Loose = derivationDte(CTX, MODEL, DERIVED, DERIVATION, {
    parentPassport,
  }).credentialSubject;
  const passport: Loose = modelDpp(CTX, DATASET, DERIVED, EVALUATION, {
    datasetPassport,
    rights,
    trainingEvent: derivationEvent,
    evaluation: derivedEvaluation,
    derivedFrom: {
      parentName: MODEL.name,
      parentPassport,
      processType: DERIVATION.processType,
    },
  }).credentialSubject;

  it("records the derivation from the parent's hash to the derived model's hash", () => {
    const [made] = event;
    expect(made.inputProduct[0].product.id).toBe(MODEL_ID);
    expect(made.inputProduct[0].disposition).toBe("active");
    expect(made.outputProduct[0].product.id).toBe(niUri(DERIVED.weights.sha256));
    expect(made.relatedDocument).toEqual([linkTo(parentPassport)]);
    expect(made.trainingRun).toMatchObject({
      processType: "Quantization (int8)",
      hyperparameters: DERIVATION.hyperparameters,
    });
  });

  it("links the derivation and the parent passport, not the dataset passport", () => {
    expect(passport.id).toBe(niUri(DERIVED.weights.sha256));
    expect(passport.relatedDocument).toEqual([linkTo(derivationEvent), linkTo(parentPassport)]);
  });

  it("carries the rights claim forward and cites its own evaluation for the loss", () => {
    const [trainingUse, loss] = passport.performanceClaim;
    expect(trainingUse.evidence).toEqual([linkTo(rights)]);
    expect(trainingUse.description).toContain(MODEL.name);
    expect(loss.evidence).toEqual([linkTo(derivedEvaluation)]);
  });

  it("uses only URIs for ids and links", () => {
    for (const uri of [...collectUris(event), ...collectUris(passport)]) {
      expect(URL.canParse(uri), uri).toBe(true);
    }
  });
});
