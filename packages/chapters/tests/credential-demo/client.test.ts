import { createHash, createHmac } from "node:crypto";
import { describe, expect, it } from "vitest";
import { requestToken } from "../../src/credential-demo/client";

const ENV = {
  apiUrl: "https://realtime.example.com",
  tenantId: "00000000-0000-4000-8000-000000000001",
  apiKey: "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
};

function decode(token: string) {
  const jwt = token.replace(/^Bearer /, "");
  const [header, payload, signature] = jwt.split(".") as [string, string, string];
  return {
    header: JSON.parse(Buffer.from(header, "base64url").toString()),
    payload: JSON.parse(Buffer.from(payload, "base64url").toString()),
    signingInput: `${header}.${payload}`,
    signature,
  };
}

describe("credential demo request token", () => {
  it("is an HS256 JWT signed with the hex-decoded key", () => {
    const { header, signingInput, signature } = decode(
      requestToken(ENV, "POST", `${ENV.apiUrl}/api/authority`),
    );
    expect(header).toEqual({
      alg: "HS256",
      kid: createHash("sha256").update(ENV.apiKey).digest("hex"),
    });
    const expected = createHmac("sha256", Buffer.from(ENV.apiKey, "hex"))
      .update(signingInput)
      .digest("base64url");
    expect(signature).toBe(expected);
  });

  it("binds the request's method and URL, without the query", () => {
    const now = Date.UTC(2026, 10, 4, 12);
    const { payload } = decode(
      requestToken(ENV, "get", `${ENV.apiUrl}/api/authority?searchTerm=Verdict`, now),
    );
    expect(payload).toMatchObject({
      iss: ENV.tenantId,
      htm: "GET",
      htu: `${ENV.apiUrl}/api/authority`,
      iat: now / 1000,
      exp: now / 1000 + 300,
    });
  });

  it("uses a new jti for every request", () => {
    const url = `${ENV.apiUrl}/api/authority`;
    const first = decode(requestToken(ENV, "GET", url)).payload.jti;
    const second = decode(requestToken(ENV, "GET", url)).payload.jti;
    expect(first).not.toBe(second);
  });
});
