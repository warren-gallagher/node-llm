import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Content hashes for the credential demo. Datasets and models are identified by the SHA-256 of
 * their bytes, written two ways:
 *
 * - as an RFC 6920 named-information URI, `ni:///sha-256;<base64url digest>`, for product ids;
 * - as a multibase-encoded multihash (`u` + base64url of 0x12 0x20 + digest), for
 *   `digestMultibase` values, as in the W3C VC data model.
 */

export interface FileDigest {
  name: string;
  mediaType: string;
  byteLength: number;
  /** SHA-256 of the file, hex. */
  sha256: string;
}

export function sha256Hex(data: Uint8Array | string): string {
  return createHash("sha256").update(data).digest("hex");
}

/** RFC 6920 `ni` URI for a SHA-256 digest given in hex. */
export function niUri(sha256: string): string {
  return `ni:///sha-256;${Buffer.from(sha256, "hex").toString("base64url")}`;
}

/** Multibase (base64url, prefix `u`) of the SHA-256 multihash of a digest given in hex. */
export function digestMultibase(sha256: string): string {
  const multihash = Buffer.concat([Buffer.from([0x12, 0x20]), Buffer.from(sha256, "hex")]);
  return `u${multihash.toString("base64url")}`;
}

export function hashFile(path: string | URL, mediaType: string): FileDigest {
  const bytes = readFileSync(path);
  const filePath = typeof path === "string" ? path : fileURLToPath(path);
  return {
    name: basename(filePath),
    mediaType,
    byteLength: bytes.byteLength,
    sha256: sha256Hex(bytes),
  };
}
