import { createHash, createHmac, randomUUID } from "node:crypto";
import type { DemoEnv } from "./env";

/**
 * A small client for the AffinitiQuest platform API (`realtime`): authorities, and signing and
 * verifying UNTP supply-chain credentials.
 */

export type CredentialType = "dpp" | "dcc" | "dte";

export interface Authority {
  id: string;
  name: string;
  description: string;
  published: { isPublished: boolean; did: string | null };
}

/** What a sign call returns. `credentialId` is where the JWT is published. */
export interface SignedCredential {
  jwt: string;
  credentialId: string;
  issuer: string;
  type: CredentialType;
  validFrom: string;
  validUntil: string;
}

export interface VerificationCheck {
  status: "pass" | "fail" | "skipped" | "warning";
  code: string;
  message: string;
  details?: unknown;
}

export interface VerificationResult {
  verified: boolean;
  credentialId: string;
  credentialType: CredentialType;
  issuer: string;
  checks: Record<string, VerificationCheck | undefined>;
  /** How the parent credential linked to this one (nested results only). */
  link?: {
    url: string;
    linkType?: string;
    linkName?: string;
    digestMultibase?: string;
    path?: string;
  };
  /** Each credential this one links, verified through its own links in turn. */
  upstreamResults?: VerificationResult[];
  errors: { code: string; message: string }[];
  warnings: { code: string; message: string }[];
}

/** A sign request: the UNTP 0.7.0 `credentialSubject` plus optional envelope fields. */
export interface SignRequest {
  credentialSubject: unknown;
  name?: string;
  validFrom?: string;
  validUntil?: string;
  extensions?: string[];
}

/** The platform settings for a new authority: a did:web that signs ES256 JWT credentials. */
const AUTHORITY_PLATFORM = {
  name: "identity-credential",
  network: "ndts",
  environment: "dev",
  signatureSuite: "ES256",
  didMethod: "did:web",
  format: "vc+jwt",
  keyManagement: "internal",
};

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly body: string,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

/**
 * The Authorization header for a v2 API key request: an HS256 JWT signed with the hex-decoded
 * key, bound to this one request by its method (`htm`) and URL without query (`htu`), with a
 * unique `jti` and a 5-minute lifetime. `kid` is the SHA-256 of the key string.
 */
export function requestToken(env: DemoEnv, method: string, url: string, now = Date.now()): string {
  const { origin, pathname } = new URL(url);
  const iat = Math.floor(now / 1000);
  const header = { alg: "HS256", kid: createHash("sha256").update(env.apiKey).digest("hex") };
  const payload = {
    iss: env.tenantId,
    jti: randomUUID(),
    htm: method.toUpperCase(),
    htu: `${origin}${pathname}`,
    iat,
    exp: iat + 300,
  };
  const signingInput = `${base64url(header)}.${base64url(payload)}`;
  const signature = createHmac("sha256", Buffer.from(env.apiKey, "hex"))
    .update(signingInput)
    .digest("base64url");
  return `Bearer ${signingInput}.${signature}`;
}

function base64url(value: unknown): string {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

export class AqClient {
  constructor(private readonly env: DemoEnv) {}

  async getAuthority(id: string): Promise<Authority | undefined> {
    try {
      return await this.request<Authority>("GET", `/api/authority/${id}`);
    } catch (error) {
      if (error instanceof HttpError && error.status === 404) return undefined;
      throw error;
    }
  }

  /** The authority with exactly this name; the search endpoint also returns partial matches. */
  async findAuthorityByName(name: string): Promise<Authority | undefined> {
    const matches = await this.request<Authority[]>(
      "GET",
      `/api/authority?searchTerm=${encodeURIComponent(name)}`,
    );
    return matches.find((authority) => authority.name === name);
  }

  createAuthority(name: string, description: string): Promise<Authority> {
    return this.request<Authority>("POST", "/api/authority", {
      name,
      description,
      domain: "",
      platform: AUTHORITY_PLATFORM,
    });
  }

  /** Publishes the authority's DID document; the did:web is assigned at this point. */
  async publishAuthority(id: string): Promise<Authority> {
    await this.request("POST", `/api/authority/${id}/publish`, {});
    const authority = await this.getAuthority(id);
    if (!authority) throw new Error(`authority ${id} disappeared after publishing`);
    return authority;
  }

  sign(type: CredentialType, authorityId: string, request: SignRequest): Promise<SignedCredential> {
    return this.request("POST", `/api/supply-chain/${type}/sign/${authorityId}`, request);
  }

  /**
   * Verifies a credential and everything it links, through the whole chain (the platform's
   * defaults). Verification needs no authentication, so this sends none, as a buyer would.
   */
  async verify(type: CredentialType, credentialId: string): Promise<VerificationResult> {
    const response = await fetch(`${this.env.apiUrl}/api/supply-chain/${type}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credentialId }),
    });
    return this.parse<VerificationResult>(response, `verify ${type}`);
  }

  /** The credential's JWT, from its did:web URL (which serves an HTML page to browsers). */
  async fetchCredential(credentialId: string): Promise<string> {
    const response = await fetch(credentialId, { headers: { Accept: "application/vc+ld+jwt" } });
    return this.parse<string>(response, `GET ${credentialId}`, false);
  }

  private async request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    const url = `${this.env.apiUrl}${path}`;
    const headers: Record<string, string> = { Authorization: requestToken(this.env, method, url) };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return this.parse<T>(response, `${method} ${path}`);
  }

  private async parse<T>(response: Response, what: string, json = true): Promise<T> {
    const text = await response.text();
    if (!response.ok) {
      throw new HttpError(
        response.status,
        text,
        `${what}: HTTP ${response.status} ${response.statusText}\n${text}`,
      );
    }
    if (!json) return text as T;
    return (text ? JSON.parse(text) : undefined) as T;
  }
}
