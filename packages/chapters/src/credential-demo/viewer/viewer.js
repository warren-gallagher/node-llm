// The credential viewer's page. The local server verifies the run and lays out the graph; this
// script only draws it, and hashes dropped files. Everything shown comes from credentials or the
// verifier, so it is set as text or attributes, never as HTML.

const SVG = "http://www.w3.org/2000/svg";
const NODE_WIDTH = 230;
const NODE_HEIGHT = 66;
const NAME_LENGTH = 32;
const CHECK_TITLES = {
  proof: "Cryptographic proof",
  credentialStructure: "Credential structure",
  schemaValidation: "Schema",
  upstreamCredentials: "Upstream credentials",
  dccValidity: "DCC validity",
  accreditation: "Accreditation",
  lineage: "Lineage",
  linkIntegrity: "Link integrity",
  revocationStatus: "Revocation",
};

const runSelect = document.getElementById("run");
const verifyButton = document.getElementById("verify");
const overall = document.getElementById("overall");
const graphSvg = document.getElementById("graph");
const graphMessage = document.getElementById("graph-message");
const selectedTitle = document.getElementById("selected-title");
const selected = document.getElementById("selected");
const claimChecks = document.getElementById("claim-checks");
const fileChecks = document.getElementById("file-checks");
const dropZone = document.getElementById("drop-zone");

let current = null;

function el(tag, attributes, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes ?? {})) {
    if (value !== undefined && value !== null) element.setAttribute(name, value);
  }
  for (const child of children) {
    if (child === undefined || child === null) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function svg(tag, attributes, ...children) {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes ?? {})) element.setAttribute(name, value);
  for (const child of children) element.append(child);
  return element;
}

async function api(path, body) {
  const response = await fetch(path, body ? { method: "POST", body: JSON.stringify(body) } : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? `${response.status} ${response.statusText}`);
  return data;
}

function shorten(text, length) {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function isHttpsUrl(value) {
  return typeof value === "string" && value.startsWith("https://");
}

// ============================================================================
// Runs and verification
// ============================================================================

async function loadRuns() {
  const runs = await api("/api/runs");
  runSelect.replaceChildren(...runs.map((run) => el("option", { value: run.runId }, run.runId)));
  if (runs.length === 0) graphMessage.textContent = "No runs yet: run pnpm demo:run.";
}

async function verify() {
  verifyButton.disabled = true;
  graphMessage.textContent = "Verifying through the platform…";
  overall.replaceChildren();
  try {
    current = await api("/api/chain", { runId: runSelect.value });
    graphMessage.textContent = `${current.graph.nodes.length} credentials. Click one to see its checks.`;
    document.getElementById("model-passport").textContent = current.modelPassport;
    overall.append(
      el(
        "span",
        { class: `badge ${current.verified ? "pass" : "fail"}` },
        current.verified ? "VERIFIED" : "NOT VERIFIED",
      ),
    );
    drawGraph(current.graph);
    showChecks(claimChecks, current.checks);
    fileChecks.replaceChildren();
    select(current.graph.nodes[0]);
  } catch (error) {
    graphMessage.replaceChildren(el("div", { class: "error-box" }, error.message));
  } finally {
    verifyButton.disabled = false;
  }
}

// ============================================================================
// The graph
// ============================================================================

function drawGraph(graph) {
  graphSvg.setAttribute("viewBox", `0 0 ${graph.width} ${graph.height}`);
  graphSvg.setAttribute("width", graph.width);
  graphSvg.setAttribute("height", graph.height);
  graphSvg.replaceChildren(
    svg(
      "defs",
      {},
      svg(
        "marker",
        {
          id: "arrow",
          viewBox: "0 0 10 10",
          refX: "10",
          refY: "5",
          markerWidth: "7",
          markerHeight: "7",
          orient: "auto",
        },
        svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--edge)" }),
      ),
    ),
    ...graph.edges.map(drawEdge),
    ...graph.nodes.map(drawNode),
  );
}

function drawEdge(edge) {
  const path = svg("path", { class: "edge", d: edge.path, "marker-end": "url(#arrow)" });
  if (edge.linkName) path.append(svg("title", {}, document.createTextNode(edge.linkName)));
  return path;
}

/** The name in up to two lines, broken between words, the second shortened if it must be. */
function nameLines(name) {
  if (name.length <= NAME_LENGTH) return [name];
  const breakAt = name.lastIndexOf(" ", NAME_LENGTH);
  const cut = breakAt > 0 ? breakAt : NAME_LENGTH;
  return [name.slice(0, cut), shorten(name.slice(cut).trim(), NAME_LENGTH)];
}

function drawNode(node) {
  const colour = `var(--${node.type})`;
  const group = svg(
    "g",
    {
      class: "node",
      "data-id": node.id,
      transform: `translate(${node.x},${node.y})`,
      tabindex: "0",
    },
    svg("title", {}, document.createTextNode(node.name)),
    svg("rect", {
      class: "box",
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      rx: "8",
      stroke: node.verified ? colour : "var(--fail)",
    }),
    svg("rect", { width: "6", height: NODE_HEIGHT, rx: "3", fill: colour }),
    svg(
      "text",
      { class: "type", x: "16", y: "20", fill: colour },
      document.createTextNode(
        `${node.type.toUpperCase()} · ${node.verified ? "verified" : "NOT VERIFIED"}`,
      ),
    ),
    ...nameLines(node.name).map((line, index) =>
      svg("text", { x: "16", y: String(38 + index * 16) }, document.createTextNode(line)),
    ),
  );
  group.addEventListener("click", () => select(node));
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") select(node);
  });
  return group;
}

function select(node) {
  for (const group of graphSvg.querySelectorAll(".node")) {
    group.classList.toggle("selected", group.dataset.id === node.id);
  }
  selectedTitle.textContent = node.name;
  const page = isHttpsUrl(node.id)
    ? el("a", { href: node.id, target: "_blank", rel: "noopener" }, "Open its did:web page")
    : null;
  const messages = el(
    "ul",
    { class: "messages" },
    ...node.errors.map((text) => el("li", { class: "error" }, text)),
    ...node.warnings.map((text) => el("li", { class: "warning" }, text)),
  );
  selected.replaceChildren(
    el(
      "div",
      {},
      el(
        "span",
        { class: `badge ${node.verified ? "pass" : "fail"}` },
        node.verified ? "VERIFIED" : "NOT VERIFIED",
      ),
      " ",
      node.type.toUpperCase(),
      " · ",
      page,
    ),
    el("div", { class: "detail" }, node.id),
    el(
      "ul",
      { class: "checks", style: "margin-top: 8px" },
      ...node.checks.map((check) =>
        el(
          "li",
          {},
          el("span", { class: `status ${check.status}` }, check.status.toUpperCase()),
          el("span", {}, `${CHECK_TITLES[check.name] ?? check.name}: ${check.message}`),
        ),
      ),
    ),
  );
  if (node.errors.length + node.warnings.length > 0) selected.append(messages);
}

// ============================================================================
// The buyer's checks and dropped files
// ============================================================================

function showChecks(list, checks) {
  list.replaceChildren(
    ...checks.map((check) =>
      el(
        "li",
        {},
        el(
          "span",
          { class: `status ${check.passed ? "pass" : "fail"}` },
          check.passed ? "PASS" : "FAIL",
        ),
        el("span", {}, check.name, el("div", { class: "detail" }, check.detail)),
      ),
    ),
  );
}

async function sha256Hex(file) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function checkFiles(files) {
  if (!current) {
    fileChecks.replaceChildren(el("li", {}, "", "Verify a run first."));
    return;
  }
  fileChecks.replaceChildren(el("li", {}, "", `Hashing ${files.length} file(s)…`));
  try {
    const digests = [];
    for (const file of files) {
      digests.push({
        name: file.name,
        mediaType: file.type,
        byteLength: file.size,
        sha256: await sha256Hex(file),
      });
    }
    showChecks(fileChecks, await api("/api/files", { runId: current.runId, files: digests }));
  } catch (error) {
    fileChecks.replaceChildren(el("li", {}, "", el("span", { class: "error-box" }, error.message)));
  }
}

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("over");
  checkFiles([...event.dataTransfer.files]);
});
verifyButton.addEventListener("click", verify);

// A run named in the URL (?run=<id>) is verified as soon as the page loads.
loadRuns()
  .then(() => {
    const runId = new URLSearchParams(window.location.search).get("run");
    if (runId && [...runSelect.options].some((option) => option.value === runId)) {
      runSelect.value = runId;
      return verify();
    }
  })
  .catch((error) => {
    graphMessage.replaceChildren(el("div", { class: "error-box" }, error.message));
  });
