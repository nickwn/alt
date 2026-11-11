"""The test model for every figure: the Stanford bunny, holes capped and
remeshed to a closed genus-0 surface of about 10k vertices.

The scanned bunny (data/bunny.obj, 2503 vertices) has open holes at its base.
Each boundary loop is capped with a fan around its centroid, and the result is
remeshed isotropically (via shapegen's pymeshlab wrapper) to ~20k faces so the
cotan Laplacian is well behaved.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "repos/shapegen"))
from shapegen.geom.mesh import remesh_isotropic  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data/bunny.obj"
CACHE = Path(__file__).resolve().parents[1] / "data/bunny_closed.npz"


def read_obj(path):
    V, F = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            F.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return np.array(V), np.array(F)


def boundary_loops(F):
    nxt = {}
    edges = {(int(a), int(b)) for f in F for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    for a, b in edges:
        if (b, a) not in edges:
            nxt.setdefault(a, []).append(b)
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v][0]
        loops.append(loop)
    return loops


def closed_bunny(n_faces=20000):
    if CACHE.exists():
        d = np.load(CACHE)
        return d["V"], d["F"]
    V, F = read_obj(DATA)
    Vs, Fs = [V], [F]
    n = len(V)
    for loop in boundary_loops(F):
        Vs.append(V[loop].mean(0)[None])
        Fs.append(np.array([[loop[(i + 1) % len(loop)], loop[i], n] for i in range(len(loop))]))
        n += 1
    V, F = np.vstack(Vs), np.vstack(Fs)
    assert not boundary_loops(F), "cap orientation inconsistent"
    V, F = remesh_isotropic(V, F, n_faces=n_faces)
    np.savez(CACHE, V=V, F=F)
    return V, F


if __name__ == "__main__":
    V, F = closed_bunny()
    E = {tuple(sorted((int(a), int(b)))) for f in F for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    print(len(V), len(F), "euler", len(V) - len(E) + len(F), "boundary loops", len(boundary_loops(F)))
