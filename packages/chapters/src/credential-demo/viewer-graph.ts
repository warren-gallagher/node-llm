import type { CredentialType, VerificationCheck, VerificationResult } from "./client";

/**
 * The credential graph for `demo:viewer`, laid out for drawing: the platform's verification tree,
 * with each credential once, in columns by depth from the credential asked about, so every link
 * points from left to right.
 */

export interface GraphNode {
  id: string;
  type: CredentialType;
  name: string;
  verified: boolean;
  checks: { name: string; status: VerificationCheck["status"]; message: string }[];
  errors: string[];
  warnings: string[];
  column: number;
  row: number;
  x: number;
  y: number;
}

export interface GraphEdge {
  from: string;
  to: string;
  linkName?: string;
  /** The SVG path to draw: a curve to the next column, or a lane above the nodes for longer links. */
  path: string;
}

export interface CredentialGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
}

export const NODE_WIDTH = 230;
export const NODE_HEIGHT = 66;
const COLUMN_GAP = 70;
const ROW_GAP = 22;
const MARGIN = 20;
/** The vertical space for each lane that carries a link past a column. */
const LANE_GAP = 14;

/** Lays out the verification tree as a graph, one node per credential. */
export function credentialGraph(root: VerificationResult): CredentialGraph {
  const results = new Map<string, VerificationResult>();
  const links: Link[] = [];
  collect(root, results, links, new Set());

  const columns = longestPathColumns(root.credentialId, links);
  // Links that skip a column run in their own lane above the nodes, so no node hides them.
  const longLinks = links.filter(
    (link) => (columns.get(link.to) ?? 0) - (columns.get(link.from) ?? 0) > 1,
  );
  const top = MARGIN + longLinks.length * LANE_GAP;
  const order = [...results.keys()];
  const byColumn = new Map<number, string[]>();
  for (const id of order) {
    const column = columns.get(id) ?? 0;
    byColumn.set(column, [...(byColumn.get(column) ?? []), id]);
  }

  const nodes: GraphNode[] = [];
  for (const [column, ids] of byColumn) {
    ids.forEach((id, row) => {
      const result = results.get(id) as VerificationResult;
      nodes.push(
        toNode(result, column, row, {
          x: MARGIN + column * (NODE_WIDTH + COLUMN_GAP),
          y: top + row * (NODE_HEIGHT + ROW_GAP),
        }),
      );
    });
  }

  const positions = new Map(nodes.map((node) => [node.id, node]));
  const edges = links.map((link) => {
    const from = positions.get(link.from) as GraphNode;
    const to = positions.get(link.to) as GraphNode;
    const lane = longLinks.indexOf(link);
    return {
      ...link,
      path: lane < 0 ? curve(from, to) : laned(from, to, MARGIN + lane * LANE_GAP),
    };
  });

  const columnCount = Math.max(...[...byColumn.keys()]) + 1;
  const rowCount = Math.max(...[...byColumn.values()].map((ids) => ids.length));
  return {
    nodes,
    edges,
    width: 2 * MARGIN + columnCount * NODE_WIDTH + (columnCount - 1) * COLUMN_GAP,
    height: top + MARGIN + rowCount * NODE_HEIGHT + (rowCount - 1) * ROW_GAP,
  };
}

type Link = Omit<GraphEdge, "path">;

/** A link to the next column: a curve from the right side of one node to the left of the other. */
function curve(from: GraphNode, to: GraphNode): string {
  const [x1, y1] = [from.x + NODE_WIDTH, from.y + NODE_HEIGHT / 2];
  const [x2, y2] = [to.x, to.y + NODE_HEIGHT / 2];
  const bend = (x2 - x1) / 2;
  return `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`;
}

/** A link past one or more columns: up into its lane above the nodes, across, and down. */
function laned(from: GraphNode, to: GraphNode, laneY: number): string {
  const [x1, y1] = [from.x + NODE_WIDTH, from.y + NODE_HEIGHT / 2];
  const [x2, y2] = [to.x, to.y + NODE_HEIGHT / 2];
  const turn = COLUMN_GAP / 2;
  return (
    `M${x1},${y1} C${x1 + turn},${y1} ${x1 + turn / 2},${laneY} ${x1 + turn},${laneY} ` +
    `L${x2 - turn},${laneY} C${x2 - turn / 2},${laneY} ${x2 - turn},${y2} ${x2},${y2}`
  );
}

/** Every credential in the tree once, and every link between two of them once. */
function collect(
  result: VerificationResult,
  results: Map<string, VerificationResult>,
  edges: Link[],
  path: Set<string>,
): void {
  if (!results.has(result.credentialId)) results.set(result.credentialId, result);
  path.add(result.credentialId);
  for (const child of result.upstreamResults ?? []) {
    const edgeKnown = edges.some(
      (edge) => edge.from === result.credentialId && edge.to === child.credentialId,
    );
    if (!edgeKnown) {
      edges.push({
        from: result.credentialId,
        to: child.credentialId,
        linkName: child.link?.linkName,
      });
    }
    if (!path.has(child.credentialId)) collect(child, results, edges, path);
  }
  path.delete(result.credentialId);
}

/** Each credential's column: the longest chain of links from the root to it. */
function longestPathColumns(rootId: string, edges: Link[]): Map<string, number> {
  const columns = new Map<string, number>([[rootId, 0]]);
  // Relax repeatedly; the graph is small, and the path set in `collect` keeps cycles out.
  for (let changed = true; changed; ) {
    changed = false;
    for (const edge of edges) {
      const from = columns.get(edge.from);
      if (from === undefined) continue;
      if ((columns.get(edge.to) ?? -1) < from + 1 && from + 1 <= edges.length) {
        columns.set(edge.to, from + 1);
        changed = true;
      }
    }
  }
  return columns;
}

function toNode(
  result: VerificationResult,
  column: number,
  row: number,
  position: { x: number; y: number },
): GraphNode {
  return {
    id: result.credentialId,
    type: result.credentialType,
    name: result.name ?? result.link?.linkName ?? result.credentialId,
    verified: result.verified,
    checks: Object.entries(result.checks)
      .filter((entry): entry is [string, VerificationCheck] => entry[1] !== undefined)
      .map(([name, check]) => ({ name, status: check.status, message: check.message })),
    errors: result.errors.map((error) => `${error.code}: ${error.message}`),
    warnings: result.warnings.map((warning) => `${warning.code}: ${warning.message}`),
    column,
    row,
    ...position,
  };
}
