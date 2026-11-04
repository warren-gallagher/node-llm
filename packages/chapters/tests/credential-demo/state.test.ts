import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { digestMultibase, sha256Hex } from "../../src/credential-demo/hashing";
import {
  loadManifest,
  partyIdentities,
  saveManifest,
  toIssued,
} from "../../src/credential-demo/state";

const API = "https://realtime.example.com";
const TENANT = "tenant-1";

function tempManifestPath(): string {
  return join(mkdtempSync(join(tmpdir(), "credential-demo-")), "credentials.json");
}

describe("credential demo manifest", () => {
  it("starts empty when there is no file", () => {
    expect(loadManifest(API, TENANT, tempManifestPath())).toEqual({
      apiUrl: API,
      tenantId: TENANT,
      authorities: {},
      accreditations: {},
      runs: [],
    });
  });

  it("round-trips through the file", () => {
    const path = tempManifestPath();
    const manifest = loadManifest(API, TENANT, path);
    manifest.authorities.bookPublisher = { id: "a1", name: "Book Publisher", did: "did:web:x" };
    saveManifest(manifest, path);
    expect(loadManifest(API, TENANT, path)).toEqual(manifest);
  });

  it("starts again for another platform or tenant", () => {
    const path = tempManifestPath();
    const manifest = loadManifest(API, TENANT, path);
    manifest.authorities.bookPublisher = { id: "a1", name: "Book Publisher", did: "did:web:x" };
    saveManifest(manifest, path);
    expect(loadManifest(API, "tenant-2", path).authorities).toEqual({});
  });

  it("asks for setup when an authority is missing", () => {
    expect(() => partyIdentities(loadManifest(API, TENANT, tempManifestPath()))).toThrow(
      /demo:setup/,
    );
  });

  it("keeps the digest of each signed JWT", () => {
    const issued = toIssued({
      jwt: "header.payload.signature",
      credentialId: "https://dids.example.com/credentials/dcc/2026/1",
      issuer: "did:web:dids.example.com",
      type: "dcc",
      validFrom: "2026-11-04T00:00:00Z",
      validUntil: "2027-11-04T00:00:00Z",
    });
    expect(issued.digestMultibase).toBe(digestMultibase(sha256Hex("header.payload.signature")));
    expect(issued).not.toHaveProperty("jwt");
  });
});
