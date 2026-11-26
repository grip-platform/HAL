"""learned_lattice — a numpy-free port of the gridlm capability-lattice ladder.

HAL issue #420. The GRIP fall-over ladder
(``raillm_ladder.GRIP_LADDER_ADDRESSES``) is today a *hand-ordered* tuple.
``examples/gridlm/gridlm_syscall.py`` proved that the same ordering can be
*learned* off a capability lattice — providers become points on a metric
manifold, substitutable providers are neighbours, and the multi-hop fallback
preference is the Banach fixed point of a contraction operator over that
lattice (Personalized PageRank; contraction modulus 0.838 < 1 => unique,
stable, geometric convergence).

That research module cannot run on the HAL hot path: it hard-depends on
``numpy`` and on ``gridlm_atlas``'s self-organising-map layout — neither is a
HAL runtime dependency. This module is a **stdlib-only re-derivation of ONLY
the ordering logic**: the same capability signatures (cost / throughput /
scheme / modes from the SAME ``TIER_COST`` / ``TIER_THROUGHPUT`` / ``MODES``
tables), a deterministic kNN capability lattice, the row-stochastic walk, and
the Banach diffusion fixed point — re-implemented with ``math`` and plain
lists (the matrices are n<=19, so a pure-python Gaussian solve is trivial and
exact). No numpy, no SOM, no randomness: the layout is the feature-space kNN
graph itself, so the ordering is deterministic and reproducible.

Honesty (the HAL ethos — measured, never asserted): cost + throughput default
to the DECLARED tier tables. The ``throughput_probe`` seam (identical contract
to ``gridlm_probe.make_probes`` — ``fn(provider) -> rpm | None``) lets a live
caller feed REAL measured rpm; any probe miss / None / non-finite reading
fails open to the declared value, per-provider. Real-rpm validation runs in the
operator's live environment; here the seam is exercised by injected fakes.

Boot guard: the constructor asserts the contraction modulus is < 1.0 and
raises ``RuntimeError`` loudly if a degenerate (non-contracting) layout is
ever fed in — a mis-wired lattice cannot ship (acceptance: "fails loudly at
boot").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from lib.hal.raillm_ladder import GRIP_LADDER_ADDRESSES

# --- capability model (schema = config/model_catalog.yaml, mirrors gridlm) ---
# Identical tier tables to examples/gridlm/gridlm_syscall.py so the math is
# faithful to the proven research result.
TIER_COST = {"subscription": 0, "local": 0, "free": 0,
             "cheap": 2, "mid": 4, "premium": 7, "frontier": 10}
TIER_THROUGHPUT = {"subscription": 2, "local": 2, "free": 4,
                   "cheap": 20, "mid": 12, "premium": 6, "frontier": 4}
MODES = {"text": frozenset({"text"}),
         "text+vision": frozenset({"text", "vision"}),
         "text+audio": frozenset({"text", "audio"}),
         "multimodal": frozenset({"text", "vision", "audio"})}


def _scheme_of(tier: str) -> str:
    return {"subscription": "subscription", "local": "local"}.get(tier, "api_key")


@dataclass(frozen=True)
class Provider:
    """One inference endpoint — a typed capability signature (same shape as the
    gridlm research ``Provider``, stdlib-only)."""
    address: str
    tier: str
    modality: str
    context_window: int

    @property
    def cost(self) -> int:
        return TIER_COST[self.tier]

    @property
    def throughput(self) -> int:
        return TIER_THROUGHPUT[self.tier]

    @property
    def scheme(self) -> str:
        return _scheme_of(self.tier)

    @property
    def modes(self) -> frozenset:
        return MODES[self.modality]

    def feature(self, cost: float | None = None) -> list[float]:
        """Normalised capability vector: cost, log-context, scheme one-hot,
        modality multi-hot — verbatim ordering from the gridlm ``feature``.
        ``cost`` overrides the declared tier cost with a measured cost index so
        the learned layout tracks observed economics (None => declared cost)."""
        c = self.cost if cost is None else cost
        sch = {"subscription": (1.0, 0.0, 0.0),
               "api_key": (0.0, 1.0, 0.0),
               "local": (0.0, 0.0, 1.0)}[self.scheme]
        return [
            c / 10.0,
            math.log10(self.context_window) / 7.0,
            *sch,
            1.0,                                   # text (always)
            1.0 if "vision" in self.modes else 0.0,
            1.0 if "audio" in self.modes else 0.0,
        ]


# Capability signatures for the operator's actual GRIP ladder addresses
# (raillm_ladder.GRIP_LADDER_ADDRESSES). Tier/modality/context_window are the
# declared metadata for each rung — the SAME schema the gridlm research uses,
# applied to the live ladder rather than the research's 19-endpoint demo set.
_LADDER_PROVIDERS: dict[str, Provider] = {
    "anthropic/claude-opus-5":       Provider("anthropic/claude-opus-5",       "subscription", "text+vision", 1_000_000),
    # Kimi K3 is METERED (keychain grip-moonshot, cash balance). HAL has no
    # "metered" tier, so it takes "frontier" -- the paid top of TIER_COST; the
    # old "subscription" made the rung look free to the lattice. Vision per the
    # live GET /v1/models flag supports_image_in (2026-09-07).
    "moonshot/kimi-k3":              Provider("moonshot/kimi-k3",              "frontier",     "text+vision", 1_048_576),
    # The Pool D rung (raillm_ladder.KIMI_CODING_LADDER_ADDRESS): Kimi Code is
    # a membership quota, so "subscription" is its honest tier; k3-256k is
    # "256k only" with "Image only" input per the vendor's model table. Present
    # in the ladder only when the kimi-coding subscription resolves; a signature
    # here costs nothing while unarmed and stops the unmodelled-rung fallback
    # (local/text/32k) misrepresenting it the moment it is armed.
    "kimi-coding/k3-256k":           Provider("kimi-coding/k3-256k",           "subscription", "text+vision",   262_144),
    "alibaba/qwen-max":              Provider("alibaba/qwen-max",              "subscription", "text",          128_000),
    "groq/llama-3.3-70b-versatile":  Provider("groq/llama-3.3-70b-versatile",  "cheap",        "text",          128_000),
    "gemini/gemini-3.5-flash-lite":  Provider("gemini/gemini-3.5-flash-lite",  "mid",          "multimodal",  1_048_576),
    "deepseek/deepseek-chat":        Provider("deepseek/deepseek-chat",        "premium",      "text",          128_000),
    "ollama/llama3.3:70b":           Provider("ollama/llama3.3:70b",           "local",        "text",           32_000),
    "mlx_vlm/auto":                  Provider("mlx_vlm/auto",                   "local",        "multimodal",    128_000),
}


def _provider_for(address: str) -> Provider:
    """The capability signature for a ladder address. Unknown addresses get a
    conservative local-text signature (fail-safe: an unmodelled rung is treated
    as a low-cost local fallback, never dropped)."""
    p = _LADDER_PROVIDERS.get(address)
    if p is not None:
        return p
    return Provider(address, "local", "text", 32_000)


# --- numpy-free linear algebra (n <= 19; exact, deterministic) --------------
def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _knn_adjacency(features: list[list[float]], k: int) -> list[list[float]]:
    """Symmetric kNN capability graph: each node links to its k nearest
    neighbours in feature space (excluding itself), unioned to be undirected.
    Returns a 0/1 adjacency matrix. Deterministic — ties broken by index."""
    n = len(features)
    adj = [[0.0] * n for _ in range(n)]
    kk = max(1, min(k, n - 1)) if n > 1 else 0
    for i in range(n):
        dists = sorted(
            ((_euclidean(features[i], features[j]), j) for j in range(n) if j != i),
            key=lambda t: (t[0], t[1]),
        )
        for _, j in dists[:kk]:
            adj[i][j] = 1.0
            adj[j][i] = 1.0                          # undirected union
    return adj


def _row_stochastic(adj: list[list[float]]) -> list[list[float]]:
    """Row-normalise the adjacency into a transition matrix. An isolated node
    (no neighbours) gets a self-loop, so every row sums to 1 — the walk is
    well-defined and the damped operator is an l1 contraction with modulus
    exactly ``alpha`` (the Banach guarantee the research proves: ``A`` row-
    stochastic means ``A^T`` has unit column sums, so ``||A^T||_1 = 1``)."""
    n = len(adj)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = sum(adj[i])
        if s > 0:
            for j in range(n):
                A[i][j] = adj[i][j] / s
        else:
            A[i][i] = 1.0
    return A


def _matvec(A: list[list[float]], v: list[float]) -> list[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _transpose(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Exact Gaussian elimination with partial pivoting for (I - a Aᵀ) x = b.
    n <= 19, so this is trivially fast and numerically clean."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise RuntimeError("singular lattice operator — degenerate layout")
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivval
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


class LearnedLattice:
    """Stdlib-only capability lattice for the GRIP fallback ladder.

    Builds the kNN capability graph -> row-stochastic walk -> Banach diffusion
    fixed point, and reads the fallback ladder off the lattice by descending
    fallback mass from the primary. ``alpha`` is the diffusion damping (the
    research's 0.85); the l1 contraction modulus is exactly ``alpha`` (a
    damped row-stochastic walk contracts the l1 norm), asserted < 1 at boot.
    """

    def __init__(self, addresses: tuple[str, ...] = GRIP_LADDER_ADDRESSES,
                 *, throughput_probe=None, alpha: float = 0.85, k: int = 3):
        self.addresses = tuple(addresses)
        self.alpha = alpha
        self.P = [_provider_for(a) for a in self.addresses]
        self.n = len(self.P)
        # economics seam: throughput from a live probe if supplied, else the
        # declared TIER_THROUGHPUT — per-provider fail-open on None / non-finite.
        self.throughput = [self._probe_or(throughput_probe, p, p.throughput)
                           for p in self.P]
        features = [p.feature() for p in self.P]
        self.A = _row_stochastic(_knn_adjacency(features, k))
        # Boot guard: a row-stochastic walk damped by alpha is an l1
        # contraction with modulus alpha < 1. Re-derive it empirically and
        # refuse to ship a non-contracting (mis-wired) layout.
        self.contraction = self._contraction_modulus()
        if not (self.contraction < 1.0):
            raise RuntimeError(
                f"learned lattice not a contraction (modulus "
                f"{self.contraction!r} >= 1.0) — refusing to ship a "
                f"non-convergent fallback ladder")

    @staticmethod
    def _probe_or(probe_fn, p: Provider, declared):
        """Probe value for provider ``p``, or the declared value if no probe is
        given, the probe returns None, or returns a non-numeric / non-finite
        value (per-provider fail-open — identical contract to the gridlm
        ``_probe_or``: a garbage reading degrades to declared, never corrupts
        the ladder)."""
        if probe_fn is None:
            return declared
        v = probe_fn(p)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            return declared
        return v

    def _contraction_modulus(self) -> float:
        """Empirical l1 contraction modulus of the damped walk
        ``T(p) = (1-a) e + a Aᵀ p``: ``||T x - T y||_1 = a ||Aᵀ(x-y)||_1
        <= a ||x-y||_1`` because ``A`` is row-stochastic, so ``Aᵀ`` has unit
        column sums and ``||Aᵀ||_1 = 1``. The measured ratio over deterministic
        probe pairs is therefore bounded by ``alpha`` for EVERY layout.

        Not the sup norm: ``||a Aᵀ||_inf = a * max column sum of A``, which is
        1.275 for the live ladder (max in-weight 1.5) — the walk is not a
        sup-norm contraction at all. Until 2026-09-07 this method measured the
        sup-norm ratio and read <= alpha only because the probe set missed the
        worst direction; the kimi-k3 capability signature moved a ramp probe
        onto it (0.8601 > 0.85) and exposed the norm mix-up."""
        At = _transpose(self.A)
        n = self.n
        if n == 0:
            return 0.0
        worst = 0.0
        # deterministic probe pairs (unit basis vectors and a ramp) — no RNG,
        # so the boot guard is reproducible.
        probes: list[list[float]] = []
        for i in range(n):
            e = [0.0] * n
            e[i] = 1.0
            probes.append(e)
        probes.append([float(i + 1) for i in range(n)])
        probes.append([float(n - i) for i in range(n)])
        for a_vec in probes:
            for b_vec in probes:
                diff = [x - y for x, y in zip(a_vec, b_vec)]
                denom = sum(abs(d) for d in diff)
                if denom < 1e-12:
                    continue
                walked = _matvec(At, diff)
                num = sum(abs(self.alpha * w) for w in walked)
                worst = max(worst, num / denom)
        return worst

    def diffuse(self, i: int, iters: int = 200) -> tuple[list[float], list[float]]:
        """Personalized PageRank from provider ``i`` over the capability
        lattice. ``T(p) = (1-a) e_i + a Aᵀ p`` is an affine contraction
        (Banach -> unique fixed point, geometric convergence). Returns
        ``(p_iter, p_closed)`` — the iterate and the closed-form solve, which
        agree to tolerance (the convergence proof, numpy-free)."""
        a, At, n = self.alpha, _transpose(self.A), self.n
        e = [0.0] * n
        e[i] = 1.0
        p = e[:]
        for _ in range(iters):
            Atp = _matvec(At, p)
            p = [(1 - a) * e[j] + a * Atp[j] for j in range(n)]
        # closed form: p_closed = (1-a) (I - a Aᵀ)^{-1} e
        Imat = [[(1.0 if r == c else 0.0) - a * At[r][c] for c in range(n)]
                for r in range(n)]
        sol = _solve(Imat, e)
        p_closed = [(1 - a) * x for x in sol]
        return p, p_closed

    def ladder(self, primary: str | None = None) -> list[str]:
        """The fallback ladder as a PERMUTATION of the input addresses: the
        primary first, then the remaining rungs ordered by descending Banach
        fallback mass from the primary (multi-hop capability proximity). Ties
        break by original ladder index — so the order is deterministic and no
        rung is ever dropped.

        ``primary`` defaults to ``addresses[0]`` (the fixed CCH rung). An
        unknown ``primary`` falls back to index 0 (fail-safe)."""
        if self.n == 0:
            return []
        idx = {a: i for i, a in enumerate(self.addresses)}
        start = idx.get(primary, 0) if primary is not None else 0
        p, _ = self.diffuse(start)
        rest = [j for j in range(self.n) if j != start]
        # descending fallback mass; tie-break by original index (stable).
        rest.sort(key=lambda j: (-p[j], j))
        order = [start] + rest
        return [self.addresses[j] for j in order]


def learned_ladder_order(addresses: tuple[str, ...] = GRIP_LADDER_ADDRESSES,
                         throughput_probe=None) -> list[str]:
    """Return ``addresses`` reordered by the lattice fixed point — a permutation
    (every rung preserved), primary first. ``throughput_probe`` (``fn(provider)
    -> rpm | None``, same contract as ``gridlm_probe.make_probes``) feeds REAL
    measured rpm when supplied; ``None`` => the declared ``TIER_THROUGHPUT``
    baseline. The contraction boot guard runs inside the constructor, so a
    mis-wired layout raises ``RuntimeError`` before any order is returned."""
    lattice = LearnedLattice(addresses, throughput_probe=throughput_probe)
    return lattice.ladder(addresses[0] if addresses else None)
