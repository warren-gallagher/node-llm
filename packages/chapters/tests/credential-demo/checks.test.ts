import { describe, expect, it } from "vitest";
import {
  artifactChecks,
  assessedProducts,
  changeOneWeight,
  changeOneWord,
  claimChecks,
  classifyClaims,
  credentialDigestCheck,
  decodeJwtPayload,
  digestOf,
  evaluationRerunCheck,
  inheritanceCheck,
  LOSS_TOLERANCE,
  modelLinks,
  quantizationRerunChecks,
  readChainClaims,
  weightComparisonCheck,
} from "../../src/credential-demo/checks";
import {
  datasetDpp,
  derivationDte,
  evaluationDcc,
  modelDpp,
  rightsDcc,
  trainingDte,
} from "../../src/credential-demo/claims";
import { niUri } from "../../src/credential-demo/hashing";
import { CTX, DATASET, DERIVATION, DERIVED, EVALUATION, issued, MODEL, TRAINING } from "./fixtures";

const rights = issued("dcc", "rights");
const datasetPassport = issued("dpp", "dataset");
const trainingEvent = issued("dte", "training");
const evaluation = issued("dcc", "evaluation");
const accreditation = issued("dcc", "accreditation");

/** The credentials as a buyer decodes them: the signed request's subject in an envelope. */
const credentials = {
  model: {
    credentialSubject: modelDpp(CTX, DATASET, MODEL, EVALUATION, {
      datasetPassport,
      rights,
      trainingEvent,
      evaluation,
    }).credentialSubject,
  },
  dataset: { credentialSubject: datasetDpp(CTX, DATASET, rights).credentialSubject },
  evaluation: {
    credentialSubject: evaluationDcc(CTX, DATASET, MODEL, EVALUATION, accreditation)
      .credentialSubject,
  },
};
const claims = readChainClaims(credentials.model, credentials.dataset, credentials.evaluation);
const files = { weights: MODEL.weights, config: MODEL.config, dataset: DATASET.file };

describe("credential demo checks", () => {
  it("decodes a JWT's payload", () => {
    const part = (value: unknown) => Buffer.from(JSON.stringify(value)).toString("base64url");
    const jwt = `${part({ alg: "ES256" })}.${part({ credentialSubject: { id: "x" } })}.sig`;
    expect(decodeJwtPayload(jwt)).toEqual({ credentialSubject: { id: "x" } });
    expect(() => decodeJwtPayload("not-a-jwt")).toThrow("not a JWT");
  });

  it("finds the evaluation and the event that produced the model from its passport", () => {
    const links = modelLinks(credentials.model);
    expect(links.evaluation.linkURL).toBe(evaluation.credentialId);
    expect(links.event?.linkURL).toBe(trainingEvent.credentialId);
  });

  it("reads the facts the checks compare", () => {
    expect(claims).toMatchObject({
      modelId: niUri(MODEL.weights.sha256),
      datasetId: niUri(DATASET.file.sha256),
      claimedLoss: EVALUATION.validationLoss,
      evaluation: {
        assessedProductId: niUri(MODEL.weights.sha256),
        loss: EVALUATION.validationLoss,
        datasetId: niUri(DATASET.file.sha256),
        harness: EVALUATION.harness,
      },
    });
    expect(claims.modelFiles.map((file) => file.name)).toEqual(["weights.bin", "config.json"]);
  });

  it("passes every check for the files the credentials describe", () => {
    const checks = [
      ...claimChecks(claims),
      ...artifactChecks(claims, files),
      evaluationRerunCheck(claims, EVALUATION.validationLoss, EVALUATION.harness.version),
    ];
    expect(checks.filter((check) => !check.passed)).toEqual([]);
  });

  it("catches an evaluation of other weights", () => {
    const other = { ...claims, evaluation: { ...claims.evaluation, assessedProductId: "ni:///x" } };
    const failed = claimChecks(other).filter((check) => !check.passed);
    expect(failed.map((check) => check.name)).toEqual([
      "The evaluation assessed the model in the passport",
    ]);
  });

  it("catches changed weights and a changed dataset by their hashes", () => {
    const changed = { ...files, weights: { ...files.weights, sha256: "0".repeat(64) } };
    expect(failedNames(artifactChecks(claims, changed))).toEqual([
      "weights.bin is the model in the passport",
    ]);
    const otherText = { ...files, dataset: { ...files.dataset, sha256: "1".repeat(64) } };
    expect(failedNames(artifactChecks(claims, otherText))).toEqual([
      "The dataset file is the dataset in its passport",
    ]);
  });

  it("accepts a rerun loss within the tolerance, and notes a different harness version", () => {
    const close = evaluationRerunCheck(
      claims,
      EVALUATION.validationLoss + LOSS_TOLERANCE / 2,
      "9.9.9",
    );
    expect(close.passed).toBe(true);
    expect(close.detail).toContain("harness 9.9.9 here");
    const far = evaluationRerunCheck(
      claims,
      EVALUATION.validationLoss + 0.01,
      EVALUATION.harness.version,
    );
    expect(far.passed).toBe(false);
  });

  it("checks that each credential read is the one its link names", () => {
    const jwt = "header.payload.signature";
    const link = { linkURL: "https://x/1", linkType: "dpp", digestMultibase: digestOf(jwt) };
    expect(credentialDigestCheck([{ link, jwt }]).passed).toBe(true);
    expect(credentialDigestCheck([{ link, jwt: `${jwt}x` }]).passed).toBe(false);
  });

  it("changes exactly one weight", () => {
    const weights = Buffer.alloc(12);
    weights.writeFloatLE(0.5, 0);
    weights.writeFloatLE(0.25, 4);
    const changed = changeOneWeight(weights);
    expect(changed.readFloatLE(0)).toBeCloseTo(0.501, 6);
    expect(changed.subarray(4)).toEqual(weights.subarray(4));
    expect(weights.readFloatLE(0)).toBe(0.5);
  });

  it("changes one word of the text", () => {
    expect(changeOneWord("Jack Gisburn painted")).toBe("Jack Gisborne painted");
    expect(() => changeOneWord("no such name")).toThrow();
  });
});

function failedNames(checks: { name: string; passed: boolean }[]): string[] {
  return checks.filter((check) => !check.passed).map((check) => check.name);
}

describe("checking a derived model", () => {
  const event = {
    credentialSubject: derivationDte(CTX, MODEL, DERIVED, DERIVATION, {
      parentPassport: issued("dpp", "parent"),
    }).credentialSubject,
  };
  const derivedClaims = readChainClaims(
    {
      credentialSubject: modelDpp(CTX, DATASET, DERIVED, EVALUATION, {
        datasetPassport,
        rights,
        trainingEvent: issued("dte", "derivation"),
        evaluation,
        derivedFrom: {
          parentName: MODEL.name,
          parentPassport: issued("dpp", "parent"),
          processType: DERIVATION.processType,
        },
      }).credentialSubject,
    },
    credentials.dataset,
    credentials.evaluation,
    event,
  );

  /** The derivation the passport's event records; every test here needs it. */
  function derivationOf(claims: typeof derivedClaims) {
    if (!claims.derivation) throw new Error("no derivation read");
    return claims.derivation;
  }

  it("reads a derivation from an event that made the model from another model", () => {
    expect(derivedClaims.derivation).toEqual({
      processType: "Quantization (int8)",
      hyperparameters: DERIVATION.hyperparameters,
      parentId: niUri(MODEL.weights.sha256),
      derivedId: niUri(DERIVED.weights.sha256),
    });
  });

  it("reads no derivation from a training run", () => {
    const training = {
      credentialSubject: trainingDte(CTX, DATASET, MODEL, TRAINING, { datasetPassport, rights })
        .credentialSubject,
    };
    const trained = readChainClaims(
      credentials.model,
      credentials.dataset,
      credentials.evaluation,
      training,
    );
    expect(trained.derivation).toBeUndefined();
  });

  it("accepts a quantization that reruns to the passport's model from the named parent", () => {
    const checks = quantizationRerunChecks(
      derivationOf(derivedClaims),
      derivedClaims.modelId,
      MODEL.weights.sha256,
      DERIVED.weights.sha256,
    );
    expect(checks.every((check) => check.passed)).toBe(true);
  });

  it("catches a different parent, a different result, and a method it can't rerun", () => {
    const derivation = derivationOf(derivedClaims);
    const rerun = (parent: string, result: string | undefined) =>
      failedNames(quantizationRerunChecks(derivation, derivedClaims.modelId, parent, result));
    expect(rerun("0".repeat(64), DERIVED.weights.sha256)).toEqual([
      "The derivation starts from the parent weights the buyer holds",
    ]);
    expect(rerun(MODEL.weights.sha256, "1".repeat(64))).toEqual([
      "Rerunning the quantization gives the model in the passport",
    ]);
    expect(rerun(MODEL.weights.sha256, undefined)).toEqual([
      "Rerunning the quantization gives the model in the passport",
    ]);
  });
});

describe("the weight comparison", () => {
  const quantization = {
    processType: "Quantization (int8)",
    hyperparameters: {},
    parentId: "ni:///p",
    derivedId: "ni:///d",
  };
  const fineTuning = { ...quantization, processType: "Fine-tuning (instruction data)" };
  const profile = (beyond: number, within: number) => ({
    tensors: 31,
    unchanged: 31 - beyond - within,
    withinRounding: within,
    beyondRounding: beyond,
    largestRelativeChange: 0.01,
  });

  it("expects a quantization to move nothing beyond rounding", () => {
    expect(weightComparisonCheck(quantization, profile(0, 31))?.passed).toBe(true);
    expect(weightComparisonCheck(quantization, profile(29, 2))?.passed).toBe(false);
    expect(weightComparisonCheck(quantization, profile(0, 0))?.passed).toBe(false);
  });

  it("expects a fine-tune to move most tensors beyond rounding", () => {
    expect(weightComparisonCheck(fineTuning, profile(29, 2))?.passed).toBe(true);
    expect(weightComparisonCheck(fineTuning, profile(0, 31))?.passed).toBe(false);
  });

  it("has no expectation for other derivations", () => {
    expect(weightComparisonCheck({ ...quantization, processType: "Merging" }, profile(1, 1))).toBe(
      undefined,
    );
  });
});

// The builders return plain objects; tests read them loosely.
// biome-ignore lint/suspicious/noExplicitAny: test access into built claims
type Loose = any;

describe("classifying a derived model's claims", () => {
  const verdictRights = issued("dcc", "rights");
  const parentEvaluation = issued("dcc", "parent-evaluation");
  const derivedEvaluation = issued("dcc", "derived-evaluation");
  const instructionRights = issued("dcc", "instruction-rights");
  const instructions = {
    ...DATASET,
    title: "Instruction data",
    file: { ...DATASET.file, name: "instruction-data.json", sha256: "7e09".padEnd(64, "0") },
  };
  const parent: Loose = {
    credentialSubject: modelDpp(CTX, DATASET, MODEL, EVALUATION, {
      datasetPassport,
      rights: verdictRights,
      trainingEvent,
      evaluation: parentEvaluation,
    }).credentialSubject,
  };
  const fineTuned: Loose = {
    credentialSubject: modelDpp(CTX, DATASET, DERIVED, EVALUATION, {
      datasetPassport,
      rights: verdictRights,
      trainingEvent: issued("dte", "fine-tuning"),
      evaluation: derivedEvaluation,
      derivedFrom: {
        parentName: MODEL.name,
        parentPassport: issued("dpp", "parent"),
        processType: "Fine-tuning",
        data: { dataset: instructions, rights: instructionRights },
      },
    }).credentialSubject,
  };
  const assessed = {
    [verdictRights.credentialId]: [niUri(DATASET.file.sha256)],
    [instructionRights.credentialId]: [niUri(instructions.file.sha256)],
    [parentEvaluation.credentialId]: [niUri(MODEL.weights.sha256)],
    [derivedEvaluation.credentialId]: [niUri(DERIVED.weights.sha256)],
  };

  it("finds the inherited rights, the new data's rights and the overriding loss claim", () => {
    expect(
      classifyClaims(fineTuned, parent).map(({ origin, parentClaim }) => ({ origin, parentClaim })),
    ).toEqual([
      { origin: "inherited", parentClaim: `Training use permitted: ${DATASET.title}` },
      { origin: "new", parentClaim: undefined },
      { origin: "overrides", parentClaim: "Validation loss" },
    ]);
  });

  it("accepts inheriting claims about the data", () => {
    const check = inheritanceCheck(
      classifyClaims(fineTuned, parent),
      assessed,
      niUri(MODEL.weights.sha256),
    );
    expect(check.passed).toBe(true);
  });

  it("refuses inheriting the parent's evaluation", () => {
    const [rights, , loss] = fineTuned.credentialSubject.performanceClaim;
    const inheritsEvaluation = {
      credentialSubject: {
        ...fineTuned.credentialSubject,
        performanceClaim: [
          rights,
          { ...loss, evidence: parent.credentialSubject.performanceClaim[1].evidence },
        ],
      },
    };
    const classified = classifyClaims(inheritsEvaluation, parent);
    expect(classified[1]).toMatchObject({ origin: "inherited", parentClaim: "Validation loss" });
    const check = inheritanceCheck(classified, assessed, niUri(MODEL.weights.sha256));
    expect(check).toMatchObject({ passed: false });
    expect(check.detail).toContain("Validation loss");
  });

  it("reads the products a DCC assessed", () => {
    const dcc = {
      credentialSubject: rightsDcc(CTX, instructions, issued("dcc", "acc")).credentialSubject,
    };
    expect(assessedProducts(dcc)).toEqual([niUri(instructions.file.sha256)]);
  });
});
