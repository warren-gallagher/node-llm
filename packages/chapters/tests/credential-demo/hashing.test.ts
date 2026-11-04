import { describe, expect, it } from "vitest";
import { digestMultibase, hashFile, niUri, sha256Hex } from "../../src/credential-demo/hashing";

const VERDICT = new URL("../../../../fixtures/the-verdict.txt", import.meta.url);

describe("credential demo hashing", () => {
  it("computes SHA-256 in hex", () => {
    expect(sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("writes ni URIs as RFC 6920 does (base64url, no padding)", () => {
    // The example in RFC 6920, section 3: the SHA-256 of "Hello World!".
    expect(niUri(sha256Hex("Hello World!"))).toBe(
      "ni:///sha-256;f4OxZX_x_FO5LcGBSKHWXfwtSx-j1ncoSt3SABJtkGk",
    );
  });

  it("writes digestMultibase as a base64url SHA-256 multihash", () => {
    const hex = sha256Hex("abc");
    const value = digestMultibase(hex);
    // "Ei" encodes 0x12 0x20; the next character already carries digest bits.
    expect(value.startsWith("uEi")).toBe(true);
    const multihash = Buffer.from(value.slice(1), "base64url");
    expect([...multihash.subarray(0, 2)]).toEqual([0x12, 0x20]);
    expect(multihash.subarray(2).toString("hex")).toBe(hex);
  });

  it("hashes The Verdict to its known digest", () => {
    expect(hashFile(VERDICT, "text/plain")).toEqual({
      name: "the-verdict.txt",
      mediaType: "text/plain",
      byteLength: 20479,
      sha256: "b41e41a68f0398a3154ae69e2e4c0e2694e17fe0d66730536837f1b01935b31f",
    });
  });
});
