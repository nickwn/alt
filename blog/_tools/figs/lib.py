"""Shared pieces for the conformal_sphere figures: line-art rendering with hidden
lines removed, and instrumented versions of the sphere-map algorithms.

Run any figure script with the shapegen venv, e.g.
    ~/repos/shapegen/.venv/bin/python blog/_tools/figs/overview.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

sys.path.insert(0, str(Path.home() / "repos/shapegen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapegen.geom.eigen_sphere import (  # noqa: E402
    FOURPI, _gen_eig, spectrum_error)
from shapegen.geom.conformal import ball_automorphism, mobius_normalize  # noqa: E402
from shapegen.geom.mesh import Mesh, icosphere  # noqa: E402
from shapegen.geom.operators import cotan_laplacian, vertex_areas  # noqa: E402
from bunny import closed_bunny  # noqa: E402

IMG = Path(__file__).resolve().parents[2] / "img/conformal_sphere"
# STIX matches the post's math font (STIX Two Math) and the serif lettering of the blog's other figures
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 11,
                     "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- #
# Model and camera
# --------------------------------------------------------------------------- #

def bunny():
    """(X, F): the closed bunny, centred, z up, outward oriented."""
    V, F = closed_bunny()
    V = V[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])       # y-up -> z-up, keeping handedness
    V = V - V.mean(0)
    V = V / np.sqrt(vertex_areas(V, F).sum() / FOURPI)       # total area 4π
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()
    if vol < 0:
        F = F[:, ::-1]
    return V, np.ascontiguousarray(F)


def rot(az, el):
    """World -> camera for z-up worlds: the camera looks along -z, y is up on screen."""
    a, e = np.radians(az), np.radians(el)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = e - np.pi / 2
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return rx @ rz


BUNNY_VIEW = rot(-35, 20)


def bunny_parts(X, F):
    """Per triangle: 0 body, 1 head, 2 ears (thresholds read off renders of the z-up bunny)."""
    c = X[F].mean(1)
    ears = c[:, 2] > 0.97
    head = ~ears & (((c[:, 0] < -0.25) & (c[:, 2] > 0.0)) | (c[:, 2] > 0.62))
    return np.where(ears, 2, np.where(head, 1, 0))


PART_GRAY = np.array([1.0, 0.78, 0.45])        # body, head, ears

# Eigenfunctions kept. The spectrum error is flat from 200 on, the 99th-percentile triangle
# distortion falls from 22 (K=100) to 3.5, and the iteration still takes only ~1.4 s.


# --------------------------------------------------------------------------- #
# Hidden-line rendering
# --------------------------------------------------------------------------- #

class Scene:
    """A mesh seen through camera ``R``: a depth buffer for visibility tests, and
    helpers that draw silhouettes and level sets of vertex functions on it."""

    def __init__(self, X, F, R=BUNNY_VIEW, res=900, box=None):
        self.F = F
        self.P = X @ R.T
        self.R = R
        lo, hi = self.P.min(0), self.P.max(0)
        if box is None:
            c, h = (lo + hi)[:2] / 2, (hi - lo)[:2].max() / 2 * 1.04
            box = (c[0] - h, c[1] - h, 2 * h)
        self.box, self.res = box, res
        self.extent = (hi - lo).max()
        t = self.P[F]
        self.front = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2] > 0
        self._depth()
        from scipy.ndimage import minimum_filter
        self.near = minimum_filter(self.buf, size=3)

    def _px(self, p):
        x0, y0, w = self.box
        return (p[..., :2] - np.array([x0, y0])) / w * (self.res - 1)

    def _depth(self):
        buf = np.full((self.res, self.res), -np.inf)
        for tri in self.P[self.F]:
            q = self._px(tri)
            lo = np.maximum(np.floor(q.min(0)).astype(int), 0)
            hi = np.minimum(np.ceil(q.max(0)).astype(int), self.res - 1)
            if (hi < lo).any():
                continue
            xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            d = (q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0]) + (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1])
            if abs(d) < 1e-12:
                continue
            l0 = ((q[1, 1] - q[2, 1]) * (xs - q[2, 0]) + (q[2, 0] - q[1, 0]) * (ys - q[2, 1])) / d
            l1 = ((q[2, 1] - q[0, 1]) * (xs - q[2, 0]) + (q[0, 0] - q[2, 0]) * (ys - q[2, 1])) / d
            l2 = 1 - l0 - l1
            m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = l0 * tri[0, 2] + l1 * tri[1, 2] + l2 * tri[2, 2]
            sub = buf[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            sub[m] = np.maximum(sub[m], z[m])
        self.buf = buf

    def visible(self, p):
        q = np.clip(np.round(self._px(p)).astype(int), 0, self.res - 1)
        return p[:, 2] >= self.near[q[:, 1], q[:, 0]] - 0.012 * self.extent

    def _runs(self, pts, ok):
        out, cur = [], []
        for p, o in zip(pts, ok):
            if o:
                cur.append(p[:2])
            else:
                if len(cur) > 1:
                    out.append(np.array(cur))
                cur = []
        if len(cur) > 1:
            out.append(np.array(cur))
        return out

    def level_set(self, f, level, keep=None):
        """Visible polylines (screen xy) of ``f == level``; ``keep`` is an optional
        vertex function, and points where it is negative are dropped."""
        F, P = self.F, self.P
        v = f[F] - level
        neg = v < 0
        tris = np.nonzero(neg.any(1) & ~neg.all(1))[0]
        nodes, adj = {}, {}
        for t in tris:
            i, vv = F[t], v[t]
            ends = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if neg[t, a] != neg[t, b]:
                    s = vv[a] / (vv[a] - vv[b])
                    key = (min(i[a], i[b]), max(i[a], i[b]))
                    if key not in nodes:
                        pt = P[i[a]] * (1 - s) + P[i[b]] * s
                        k = 1.0 if keep is None else keep[i[a]] * (1 - s) + keep[i[b]] * s
                        nodes[key] = (pt, k)
                    ends.append(key)
            if len(ends) == 2:
                adj.setdefault(ends[0], []).append(ends[1])
                adj.setdefault(ends[1], []).append(ends[0])
        seen, chains = set(), []
        order = [k for k in adj if len(adj[k]) == 1] + list(adj)
        for s in order:
            if s in seen:
                continue
            chain, k = [s], s
            seen.add(s)
            while True:
                nxt = [n for n in adj[k] if n not in seen]
                if not nxt:
                    if s in adj[k] and len(chain) > 2:
                        chain.append(s)
                    break
                k = nxt[0]
                seen.add(k)
                chain.append(k)
            chains.append(chain)
        out = []
        for ch in chains:
            pts = np.array([nodes[k][0] for k in ch])
            ok = self.visible(pts)
            if keep is not None:
                ok &= np.array([nodes[k][1] >= 0 for k in ch])
            out += self._runs(pts, ok)
        return out

    def silhouette(self):
        F, P = self.F, self.P
        e = {}
        for t, f in enumerate(F):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                e.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append(self.front[t])
        segs = []
        for (a, b), s in e.items():
            if len(s) == 2 and s[0] != s[1]:
                m = (P[a] + P[b]) / 2
                if self.visible(m[None])[0]:
                    segs.append(P[[a, b], :2])
        return segs

    def wire(self, ax, color="0.45", lw=0.25):
        """Flat white triangles with gray edges, painter-sorted."""
        idx = np.nonzero(self.front)[0]
        idx = idx[np.argsort(self.P[self.F[idx]][:, :, 2].mean(1))]
        ax.add_collection(PolyCollection(self.P[self.F[idx]][:, :, :2], facecolors="white",
                                         edgecolors=color, linewidths=lw))

    def fill(self, ax, colors):
        """Flat-shaded front triangles, painter-sorted; ``colors`` is one gray level
        (0 black, 1 white) per triangle."""
        idx = np.nonzero(self.front)[0]
        idx = idx[np.argsort(self.P[self.F[idx]][:, :, 2].mean(1))]
        c = np.repeat(np.asarray(colors, float)[idx, None], 3, 1)
        ax.add_collection(PolyCollection(self.P[self.F[idx]][:, :, :2], facecolors=c,
                                         edgecolors=c, linewidths=0.3))

    def draw(self, ax, sets, lw=1.4, sil=2.2):
        """``sets``: list of (polylines, linewidth, linestyle) from :meth:`level_set`."""
        ax.add_collection(LineCollection(self.silhouette(), colors="k", linewidths=sil,
                                         capstyle="round"))
        for lines, w, ls in sets:
            ax.add_collection(LineCollection(lines, colors="k", linewidths=w or lw, linestyles=ls,
                                             capstyle="round", joinstyle="round"))
        x0, y0, w = self.box
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0, y0 + w)
        ax.set_aspect("equal")
        ax.axis("off")


# --------------------------------------------------------------------------- #
# Grids on the sphere, pulled back through a map S (unit vectors per vertex)
# --------------------------------------------------------------------------- #

def pole_rotation(S, v0, v1):
    """Rotation of the sphere putting vertex ``v0``'s image at the north pole and
    ``v1``'s image on the prime meridian (so different maps can be compared)."""
    z = S[v0]
    x = S[v1] - (S[v1] @ z) * z
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def aligned(S, v0, v1):
    return S @ pole_rotation(S, v0, v1).T


def view_poles(X, R=BUNNY_VIEW):
    """Two visible vertices near the front/top of the camera view: pole and meridian anchor."""
    P = X @ R.T
    v0 = int(np.argmax(P[:, 2] + 0.8 * P[:, 1]))
    v1 = int(np.argmax(P[:, 2] - 0.8 * P[:, 0]))
    return v0, v1


# Longitudes every 30 degrees, latitudes spaced like a Mercator chart (equal steps of
# log tan(pi/4 + lat/2)), so that every cell of the grid is a curvilinear square on the
# sphere itself, getting smaller toward the poles. A conformal map keeps them square.
LONS = np.radians(np.arange(0, 360, 30))
LATS = 2 * np.arctan(np.exp(np.arange(-4, 5) * np.pi / 6)) - np.pi / 2


def grid_sets(scene, S, lw=1.5):
    """Latitude/longitude curves of ``S`` as drawable sets for ``scene``."""
    sets = []
    for lat in LATS:
        sets.append((scene.level_set(S[:, 2], np.sin(lat)), lw, "-"))
    for lon in LONS:
        n = np.array([-np.sin(lon), np.cos(lon), 0.0])
        d = np.array([np.cos(lon), np.sin(lon), 0.0])
        sets.append((scene.level_set(S @ n, 0.0, keep=S @ d), lw, "-"))
    return sets


def circle_sets(scene, S, n=110, r=0.115, lw=1.3):
    """Equal spherical circles (centres on a Fibonacci lattice) pulled back through ``S``."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    ph = np.pi * (1 + 5 ** 0.5) * k
    cs = np.stack([np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z], 1)
    return [(scene.level_set(S @ c, np.cos(r)), lw, "-") for c in cs]


def unit_sphere(R=None, subdiv=5):
    """(scene, S): a fine icosphere seen through camera R (default: z up, slightly from above)."""
    m = icosphere(subdiv)
    return Scene(m.vertices, m.faces, rot(0, 22) if R is None else R, box=(-1.06, -1.06, 2.12)), m.vertices


def levels_sets(scene, f, levels, lw=1.1, zero_lw=2.2):
    sets = []
    for c in levels:
        sets.append((scene.level_set(f, c), zero_lw if c == 0 else lw, "-" if c >= 0 else (0, (4, 3))))
    return sets


def save(fig, name):
    fig.savefig(IMG / name, transparent=True, bbox_inches="tight", pad_inches=0.04)
    print("wrote", IMG / name)


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #

def corner_cotangents(X, F):
    """(nF, 3) cotangent of the angle at each corner of each triangle."""
    out = np.empty((len(F), 3))
    for k in range(3):
        u = X[F[:, (k + 1) % 3]] - X[F[:, k]]
        v = X[F[:, (k + 2) % 3]] - X[F[:, k]]
        out[:, k] = (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
    return out


def energy_density(S, F, cot, m0):
    """The post's density step: 1/2 <dS ^ *dS> per triangle is 1/4 sum_k cot_k |dS(e_k)|^2 (e_k
    opposite corner k); a third goes to each corner; divide by the vertex area to get w."""
    e = 0.25 * sum(cot[:, k] * ((S[F[:, (k + 2) % 3]] - S[F[:, (k + 1) % 3]]) ** 2).sum(1)
                   for k in range(3))
    return np.bincount(F.ravel(), weights=np.repeat(e / 3, 3), minlength=len(m0)) / m0


def splu_spd(A):
    """Sparse LU of a symmetric positive definite matrix (symmetric ordering, no pivoting)."""
    import scipy.sparse.linalg as spla
    return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True))


def setup(X, F):
    """What the iteration needs from the input mesh: L0, the vertex areas, the corner cotangents,
    and one factorization of L0 + M0, the preconditioner for every eigensolve."""
    import scipy.sparse.linalg as spla
    L0, M0 = cotan_laplacian(X, F)
    L0 = L0.tocsr()
    lu = splu_spd(L0 + M0)
    P = spla.LinearOperator(L0.shape, matvec=lu.solve, matmat=lu.solve, dtype=float)
    return dict(L0=L0, m0=M0.diagonal(), P=P, F=F, X=X, cot=corner_cotangents(X, F))


def eigenstep(st, w, f, inner=1):
    """``inner`` LOBPCG iterations toward the three lowest non-constant eigenvectors of
    (L0, diag(m0 w)), warm-started at the map f and kept orthogonal to constants; scaled so that
    each coordinate has int f_c^2 w dA = 4 pi / 3."""
    import warnings
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # lobpcg warns that one iteration is not converged;
        # it is given a copy because it normalizes its starting block in place
        _, f = spla.lobpcg(st["L0"], f.copy(), B=sp.diags(st["m0"] * w), M=st["P"],
                           Y=np.ones((len(w), 1)), largest=False, maxiter=inner, tol=1e-12)
    return f * np.sqrt(FOURPI / 3)


def apply_gauge(st, f, w):
    """The first-order boost that puts the input-area centroid int f dA at the origin; it
    multiplies the density by 1 - 2 <a, f>."""
    m0 = st["m0"]
    a = -np.linalg.solve(FOURPI * np.eye(3) - f.T @ (m0[:, None] * f), m0 @ f)
    return w * (1 - 2 * f @ a)


def conformal_iterates(st, gauge=True, m_hist=6, max_iter=200, tol=1e-7, stop_at_tol=True,
                       floor=1e-3, inner=1):
    """The post's ``conformal_sphere`` iteration (see the code listing there), instrumented:
    eigenvector step, density from the energy density of the map, first-order Möbius gauge,
    Anderson acceleration of the density with memory ``m_hist``, and the floor on the density.

    Returns the maps ``f_k`` (n x 3, before projection), the residual history and per-step times.
    """
    m0, F, cot = st["m0"], st["F"], st["cot"]

    def normalize(w):
        return w * FOURPI / (m0 @ w)

    X = st["X"]
    f = X - (m0[:, None] * X).sum(0) / m0.sum()      # first guess for the eigenvectors
    w = normalize(np.ones(len(m0)))
    Gs, Fs, maps, res_hist, times = [], [], [], [], []
    t0 = time.perf_counter()
    for _ in range(max_iter):
        f = eigenstep(st, w, f, inner)
        gw = normalize(energy_density(f, F, cot, m0))
        if gauge:
            gw = normalize(np.maximum(apply_gauge(st, f, gw), floor * gw.mean()))
        r = gw - w
        res = float(np.linalg.norm(r) / np.linalg.norm(w))
        maps.append(f)
        res_hist.append(res)
        times.append(time.perf_counter() - t0)
        if stop_at_tol and res < tol:
            break
        Gs.append(gw)
        Fs.append(r)
        Gs, Fs = Gs[-(m_hist + 1):], Fs[-(m_hist + 1):]
        if m_hist == 0 or len(Fs) == 1:
            w = gw
            continue
        dF = np.diff(Fs, axis=0).T
        dG = np.diff(Gs, axis=0).T
        gamma = np.linalg.lstsq(dF, r, rcond=None)[0]
        w = normalize(np.maximum(gw - dG @ gamma, floor * gw.mean()))
    return maps, res_hist, times


def to_sphere(R3):
    return R3 / np.linalg.norm(R3, axis=1, keepdims=True)


def cmcf_steps(X, F, dt=2.0, max_steps=300, stop=3e-4, refactor_after=25):
    """Conformalized mean-curvature flow (Kazhdan et al.): fixed L0, the vertex areas of the
    current surface, backward Euler, then recentre on the area-weighted centroid and rescale
    to area 4 pi. Each step solves (M_k + dt L0) V = M_k V_k with conjugate gradients,
    preconditioned by a factorization that is reused until CG needs more than
    ``refactor_after`` iterations. Yields (step, positions, seconds, move) after every step,
    where move is the area-weighted rms displacement of the step; stops once move < stop."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    t0 = time.perf_counter()
    L0, _ = cotan_laplacian(X, F)
    L0 = L0.tocsc()
    n = len(X)
    V = X.copy()
    lu = None
    t = time.perf_counter() - t0
    for k in range(1, max_steps + 1):
        t0 = time.perf_counter()
        m = vertex_areas(V, F)
        A = (sp.diags(m) + dt * L0).tocsc()
        rhs = m[:, None] * V
        if lu is None:
            lu = splu_spd(A)
            Vn = lu.solve(rhs)
        else:
            P = spla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
            Vn = np.empty_like(V)
            worst = 0
            for c in range(3):
                its = [0]
                Vn[:, c], _ = spla.cg(A, rhs[:, c], x0=V[:, c], rtol=1e-10, M=P,
                                      callback=lambda _: its.__setitem__(0, its[0] + 1))
                worst = max(worst, its[0])
            if worst > refactor_after:
                lu = None
        mn = vertex_areas(Vn, F)
        Vn -= (mn[:, None] * Vn).sum(0) / mn.sum()
        Vn /= np.sqrt(vertex_areas(Vn, F).sum() / FOURPI)
        move = float(np.sqrt((mn * ((Vn - V) ** 2).sum(1)).sum() / FOURPI))
        V = Vn
        t += time.perf_counter() - t0
        yield k, V, t, move
        if stop is not None and move < stop:
            return


def triangle_distortion(X, F, S):
    """Per triangle: ratio of the larger to the smaller singular value of the affine
    map from the input triangle to its image (1 = locally conformal)."""
    def frames(P):
        e1 = P[F[:, 1]] - P[F[:, 0]]
        e2 = P[F[:, 2]] - P[F[:, 0]]
        a = np.linalg.norm(e1, axis=1)
        u = e1 / a[:, None]
        b = (e2 * u).sum(1)
        c = np.linalg.norm(e2 - b[:, None] * u, axis=1)
        return a, b, c
    a0, b0, c0 = frames(X)
    a1, b1, c1 = frames(S)
    out = np.empty(len(F))
    for i in range(len(F)):
        A0 = np.array([[a0[i], b0[i]], [0, c0[i]]])
        A1 = np.array([[a1[i], b1[i]], [0, c1[i]]])
        s = np.linalg.svd(A1 @ np.linalg.inv(A0), compute_uv=False)
        out[i] = s[0] / max(s[1], 1e-300)
    return out


def final_map(st, X, F, f):
    """Unit-sphere map from the converged map: project, fix orientation, Möbius-centre."""
    S = to_sphere(f)
    nrm = np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]])
    if ((nrm * S[F].mean(1)).sum(1) < 0).sum() > len(F) // 2:
        S = S * np.array([1.0, 1.0, -1.0])
    return mobius_normalize(Mesh(X, F), S)


def conformal_sphere_timed(X, F, on_step=None, **kw):
    """The whole method with a time breakdown: (S, info). on_step(seconds, f) after each iteration."""
    T = {}
    t0 = time.perf_counter()
    st = setup(X, F)
    T["setup"] = time.perf_counter() - t0
    maps, res, times = conformal_iterates(st, **kw)
    T["iterate"] = times[-1]
    T["iters"] = len(maps)
    if on_step is not None:
        for t, f in zip(times, maps):
            on_step(T["setup"] + t, f)
    t1 = time.perf_counter()
    S = final_map(st, X, F, maps[-1])
    T["final"] = time.perf_counter() - t1
    T["total"] = T["setup"] + T["iterate"] + T["final"]
    T["f"] = maps[-1]
    return S, T


def bunny_map(X, F):
    """The converged conformal_sphere map of the bunny and the quantities behind it."""
    S, T = conformal_sphere_timed(X, F)
    return None, S, T
