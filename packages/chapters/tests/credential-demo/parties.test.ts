import { describe, expect, it } from "vitest";
import {
  didDocumentUrl,
  didWebHost,
  NAME_PREFIX,
  PARTIES,
  PARTY_KEYS,
} from "../../src/credential-demo/parties";

describe("credential demo parties", () => {
  it("prefixes every authority name", () => {
    expect(PARTY_KEYS).toHaveLength(5);
    for (const key of PARTY_KEYS) {
      expect(PARTIES[key].name.startsWith(NAME_PREFIX)).toBe(true);
    }
  });

  it("gives the producers a facility and a country", () => {
    for (const key of ["bookPublisher", "modelPublisher"] as const) {
      expect(PARTIES[key].facility).toBeDefined();
      expect(PARTIES[key].country?.countryCode).toHaveLength(2);
    }
  });

  it("finds a did:web's host", () => {
    expect(didWebHost("did:web:a1b2.dids-aq.ngrok.io")).toBe("a1b2.dids-aq.ngrok.io");
    expect(didWebHost("did:web:example.com%3A8443:user:alice")).toBe("example.com:8443");
    expect(() => didWebHost("did:key:z6Mk")).toThrow();
  });

  it("finds a did:web's DID document", () => {
    expect(didDocumentUrl("did:web:example.com")).toBe("https://example.com/.well-known/did.json");
    expect(didDocumentUrl("did:web:example.com%3A8443:user:alice")).toBe(
      "https://example.com:8443/user/alice/did.json",
    );
  });
});
