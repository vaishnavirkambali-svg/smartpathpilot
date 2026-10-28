"""Multi-stop route planning on a NetworkX graph (Graph, DiGraph, MultiGraph, MultiDiGraph).
Waypoints: index 0 = source (fixed), 1..m = mandatory stops, m+1 = destination (fixed).
All costs come from the road graph (Dijkstra shortest paths between waypoints), never straight-line distance."""
import math
import networkx as nx

EXACT_MAX = 12            # up to this many stops: exact Held-Karp dynamic programming; above: heuristic
INF = math.inf

class RoutingError(Exception): status = 422
class InvalidWeight(RoutingError): status = 422
class Unreachable(RoutingError): status = 404
class TooManyStops(RoutingError): status = 422

def _valid(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0

def weight_callable(G, weight):
    """Dijkstra weight function. `weight` is an edge-attribute name or callable(u, v, edge_data) -> number.
    On multigraphs the cheapest valid parallel edge is used. Missing/negative/NaN/infinite weights raise InvalidWeight."""
    base = weight if callable(weight) else (lambda u, v, d: d.get(weight))
    def f(u, v, d):
        if G.is_multigraph():
            vals = [x for x in (base(u, v, e) for e in d.values()) if _valid(x)]
            if not vals: raise InvalidWeight(f"Edge {u}->{v} has no valid '{weight}' weight.")
            return min(vals)
        w = base(u, v, d)
        if not _valid(w): raise InvalidWeight(f"Edge {u}->{v} has an invalid '{weight}' weight: {w!r}.")
        return w
    return f

def validate_weights(G, weight):
    """Scan every edge once (attribute-name weights) so bad data fails loudly instead of being skipped by Dijkstra."""
    f = weight_callable(G, weight)
    for u, v, d in (G.edges(data=True) if not G.is_multigraph() else ((u, v, G[u][v]) for u, v in set(G.edges()))):
        f(u, v, d)

def best_edge(G, u, v, weight):
    """(key, data) of the cheapest valid edge u->v; key is None on simple graphs."""
    if not G.is_multigraph(): return None, G[u][v]
    base = weight if callable(weight) else (lambda a, b, d: d.get(weight))
    cands = [(base(u, v, d), k, d) for k, d in G[u][v].items() if _valid(base(u, v, d))]
    if not cands: raise InvalidWeight(f"Edge {u}->{v} has no valid weight.")
    w, k, d = min(cands, key=lambda x: (x[0], str(x[1]))); return k, d

def pairwise(G, nodes, weight, sources=None):
    """Cost matrix + node paths between waypoints, from Dijkstra on the graph. INF = unreachable."""
    wf = weight_callable(G, weight); n = len(nodes); cost = [[INF] * n for _ in range(n)]; paths = {}; cache = {}
    for i in (range(n) if sources is None else sources):
        u = nodes[i]
        if u not in cache: cache[u] = nx.single_source_dijkstra(G, u, weight=wf)
        dist, pth = cache[u]
        for j in range(n):
            v = nodes[j]
            if u == v: cost[i][j] = 0.0; paths[(i, j)] = [u]
            elif v in dist: cost[i][j] = dist[v]; paths[(i, j)] = pth[v]
    return cost, paths

def _total(cost, order, end):
    seq = [0] + list(order) + [end]; return sum(cost[a][b] for a, b in zip(seq, seq[1:]))

def held_karp(cost, m):
    """Exact optimal stop order (fixed start 0 and end m+1). Returns (order, states_evaluated). O(2^m * m^2)."""
    if m == 0: return [], 1
    end = m + 1; dp = {}; parent = {}
    for j in range(1, m + 1): dp[(1 << (j - 1), j)] = cost[0][j]
    for mask in range(1, 1 << m):
        for j in range(1, m + 1):
            cur = dp.get((mask, j))
            if cur is None or cur == INF: continue
            for k in range(1, m + 1):
                if mask & (1 << (k - 1)): continue
                nm = mask | (1 << (k - 1)); c = cur + cost[j][k]
                if c < dp.get((nm, k), INF): dp[(nm, k)] = c; parent[(nm, k)] = j
    full = (1 << m) - 1; best, bj = INF, None
    for j in range(1, m + 1):
        c = dp.get((full, j), INF) + cost[j][end]
        if c < best: best, bj = c, j
    if bj is None: return list(range(1, m + 1)), len(dp)
    order, mask, j = [], full, bj
    while j is not None:
        order.append(j); pj = parent.get((mask, j)); mask ^= 1 << (j - 1); j = pj
    return order[::-1], len(dp)

def heuristic(cost, m):
    """Nearest neighbour, then 2-opt and or-opt until no improvement. Deterministic; not guaranteed optimal."""
    end = m + 1; left = set(range(1, m + 1)); cur, order = 0, []
    while left:
        nxt = min(left, key=lambda j: (cost[cur][j], j)); order.append(nxt); left.remove(nxt); cur = nxt
    best = _total(cost, order, end); evaluated = 1; improved = True; passes = 0
    while improved and passes < 100:
        improved = False; passes += 1
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]; c = _total(cost, cand, end); evaluated += 1
                if c < best - 1e-9: order, best, improved = cand, c, True
        for L in (1, 2, 3):
            for i in range(len(order) - L + 1):
                seg, rest = order[i:i + L], order[:i] + order[i + L:]
                for p in range(len(rest) + 1):
                    if p == i: continue
                    cand = rest[:p] + seg + rest[p:]; c = _total(cost, cand, end); evaluated += 1
                    if c < best - 1e-9: order, best, improved = cand, c, True
    return order, evaluated

def _diagnose(cost, m, lab):
    end = m + 1
    for j in range(1, m + 1):
        if all(cost[i][j] == INF for i in range(0, m + 1) if i != j): return f"Stop {j} ({lab(j)}) cannot be reached by road from the source or any other stop."
        if all(cost[j][k] == INF for k in range(1, m + 2) if k != j): return f"Stop {j} ({lab(j)}) has no road connection to the remaining stops or the destination."
    return "No ordering of the stops forms a connected drivable route."

def plan(G, nodes, weight, labels=None, max_mid=20, exact_max=EXACT_MAX):
    """Returns {"strict": r, "optimized": r, "entered_cost": float}; r = {order, legs[(i, j, node_path, cost)], total, method, evaluated}."""
    n = len(nodes)
    if n < 2: raise RoutingError("A source and a destination are required.")
    m = n - 2
    if m > max_mid: raise TooManyStops(f"At most {max_mid} mandatory stops are supported (got {m}).")
    lab = lambda i: (labels[i] if labels and i < len(labels) and labels[i] else f"waypoint {i}")
    for i, u in enumerate(nodes):
        if u not in G: raise Unreachable(f"{lab(i)} is not on the road graph.")
    cost, paths = pairwise(G, nodes, weight, sources=range(n - 1)); end = n - 1
    def result(order, method, evaluated):
        seq = [0] + list(order) + [end]
        return {"order": seq, "legs": [(a, b, paths[(a, b)], cost[a][b]) for a, b in zip(seq, seq[1:])],
                "total": sum(cost[a][b] for a, b in zip(seq, seq[1:])), "method": method, "evaluated": evaluated}
    entered = list(range(1, m + 1)); seq = [0] + entered + [end]
    for a, b in zip(seq, seq[1:]):
        if cost[a][b] == INF: raise Unreachable(f"No drivable route from {lab(a)} to {lab(b)}.")
    if m <= 1: opt, method, ev = entered, "trivial", 1
    elif m <= exact_max: opt, ev = held_karp(cost, m); method = "exact_held_karp"
    else: opt, ev = heuristic(cost, m); method = "heuristic_nn_2opt_oropt"
    if _total(cost, opt, end) == INF: raise Unreachable(_diagnose(cost, m, lab))
    return {"strict": result(entered, "strict_as_entered", 1), "optimized": result(opt, method, ev), "entered_cost": _total(cost, entered, end)}
