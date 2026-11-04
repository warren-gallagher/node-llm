import { accreditationDcc, type DemoContext } from "./claims";
import { AqClient, type Authority } from "./client";
import { loadDemoEnv } from "./env";
import {
  type AccreditedKey,
  didDocumentUrl,
  didWebHost,
  PARTIES,
  PARTY_KEYS,
  type PartyKey,
} from "./parties";
import {
  type AuthorityRecord,
  loadManifest,
  type Manifest,
  partyIdentities,
  saveManifest,
  toIssued,
} from "./state";

/**
 * Credential demo, setup: creates and publishes the five authorities, checks that each DID
 * resolves, then has Evaluators Association accredit the two evaluators. Safe to run again: it
 * reuses the authorities and accreditations the manifest (or the platform) already has.
 *
 * A did:web's host is only known once the authority is published, and each host needs its own
 * ngrok tunnel to the local did:web server. So the first run prints the hosts and stops if any
 * DID doesn't resolve; start the tunnels and run it again.
 */
async function main(): Promise<void> {
  const env = loadDemoEnv();
  const client = new AqClient(env);
  const manifest = loadManifest(env.apiUrl, env.tenantId);

  console.log(`Platform: ${env.apiUrl}\n\nAuthorities:`);
  for (const key of PARTY_KEYS) {
    await ensureAuthority(client, manifest, key);
  }
  saveManifest(manifest);
  printAuthorities(manifest);

  const unresolved = await unresolvedDids(manifest);
  if (unresolved.length > 0) {
    console.error("\nThese DIDs don't resolve yet. Start a tunnel for each host, then run again:");
    for (const { record, reason } of unresolved) {
      console.error(`  ${didWebHost(record.did)}  (${record.name}: ${reason})`);
    }
    process.exitCode = 1;
    return;
  }
  console.log("\nEvery DID resolves.");

  const ctx: DemoContext = {
    apiUrl: env.apiUrl,
    parties: partyIdentities(manifest),
    date: new Date().toISOString().slice(0, 10),
  };
  const issuer = manifest.authorities.evaluatorsAssociation as AuthorityRecord;
  for (const auditor of ["bookAuditor", "modelAuditor"] as AccreditedKey[]) {
    const existing = manifest.accreditations[auditor];
    if (existing) {
      console.log(`Accreditation of ${PARTIES[auditor].name}: reusing ${existing.credentialId}`);
      continue;
    }
    const signed = await client.sign("dcc", issuer.id, accreditationDcc(ctx, auditor));
    manifest.accreditations[auditor] = toIssued(signed);
    saveManifest(manifest);
    console.log(`Accreditation of ${PARTIES[auditor].name}: issued ${signed.credentialId}`);
  }
  console.log("\nSetup is complete. Next: pnpm demo:run");
}

/**
 * Finds the party's authority (by the ID in the manifest, then by name), creates it if there is
 * none, and publishes it if it isn't published yet.
 */
async function ensureAuthority(client: AqClient, manifest: Manifest, key: PartyKey): Promise<void> {
  const party = PARTIES[key];
  const saved = manifest.authorities[key];

  let authority: Authority | undefined;
  if (saved) authority = await client.getAuthority(saved.id);
  authority ??= await client.findAuthorityByName(party.name);
  if (authority) {
    console.log(`  ${party.name}: exists`);
  } else {
    authority = await client.createAuthority(party.name, party.description);
    console.log(`  ${party.name}: created`);
  }
  if (!authority.published.isPublished || !authority.published.did) {
    authority = await client.publishAuthority(authority.id);
    console.log(`  ${party.name}: published`);
  }

  const did = authority.published.did;
  if (!did) throw new Error(`${party.name} has no DID after publishing`);
  if (saved && saved.did !== did) {
    // Credentials issued under the old DID no longer belong to this authority.
    console.warn(`  ${party.name}: DID changed; earlier credentials will be issued again`);
    manifest.accreditations = {};
    manifest.dataset = undefined;
  }
  manifest.authorities[key] = { id: authority.id, name: authority.name, did };
}

function printAuthorities(manifest: Manifest): void {
  const rows = PARTY_KEYS.map((key) => {
    const record = manifest.authorities[key] as AuthorityRecord;
    return [PARTIES[key].role, record.name, didWebHost(record.did), record.did];
  });
  const header = ["Role", "Authority", "Host to tunnel", "DID"];
  const widths = header.map((title, i) =>
    Math.max(title.length, ...rows.map((row) => row[i]?.length ?? 0)),
  );
  console.log();
  for (const row of [header, ...rows]) {
    console.log(
      row
        .map((cell, i) => cell.padEnd(widths[i] ?? 0))
        .join("  ")
        .trimEnd(),
    );
  }
}

/** The authorities whose DID document can't be fetched, or doesn't name the DID. */
async function unresolvedDids(
  manifest: Manifest,
): Promise<{ record: AuthorityRecord; reason: string }[]> {
  const records = PARTY_KEYS.map((key) => manifest.authorities[key] as AuthorityRecord);
  const results = await Promise.all(
    records.map(async (record) => ({ record, reason: await resolveFailure(record.did) })),
  );
  return results.filter((result): result is { record: AuthorityRecord; reason: string } =>
    Boolean(result.reason),
  );
}

async function resolveFailure(did: string): Promise<string | undefined> {
  try {
    const response = await fetch(didDocumentUrl(did), { signal: AbortSignal.timeout(10_000) });
    if (!response.ok) return `HTTP ${response.status}`;
    const document = (await response.json()) as { id?: string };
    return document.id === did ? undefined : `the document is for ${document.id}`;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
