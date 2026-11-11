"""Prints every number the post quotes about the bunny (vertex count, iterations, timings, spectrum errors,
convergence of the variants, the cMCF comparison, triangle distortion). Timings are best of three."""
from lib import *

X, F = bunny()
print("vertices", len(X), "faces", len(F))
S, T = min((conformal_sphere_timed(X, F) for _ in range(3)), key=lambda r: r[1]["total"])
print(f"conformal_sphere: setup {T['setup']:.3f}s, {T['iters']} iterations {T['iterate']:.3f}s "
      f"({1000 * T['iterate'] / T['iters']:.1f} ms each), final {T['final']:.3f}s, total {T['total']:.3f}s, "
      f"spectrum error {spectrum_error(X, F, S):.5f}")
r = np.linalg.norm(T["f"], axis=1)
print(f"  |f| before projection from {r.min():.4f} to {r.max():.4f}")
st = setup(X, F)
for name, kw in [("plain, gauge", dict(m_hist=0)), ("plain, no gauge", dict(gauge=False, m_hist=0)),
                 ("Anderson, no gauge", dict(gauge=False)), ("Anderson, gauge", dict())]:
    _, res, _ = conformal_iterates(st, max_iter=200, tol=1e-10, **kw)
    res = np.array(res)
    k7 = int(np.argmax(res < 1e-7)) + 1 if (res < 1e-7).any() else None
    rate = np.exp(np.mean(np.log(res[31:56] / res[30:55]))) if len(res) > 56 else float("nan")
    print(f"  {name}: {len(res)} iterations, final {res[-1]:.2g}, min {res.min():.2g}, 1e-7 at {k7}, rate {rate:.4f}")
S_r, T_r = conformal_sphere_timed(X, F)
print("cMCF (stop when the rms move per step < 3e-4), best of three:")
for dt in (0.05, 0.2, 0.5, 1.0, 2.0):
    best = None
    for _ in range(3):
        for k, V, t, move in cmcf_steps(X, F, dt=dt):
            pass
        if best is None or t < best[1]:
            best = (k, t, V)
    k, t, V = best
    print(f"  dt={dt:g}: {k} steps, {t:.3f}s, error {spectrum_error(X, F, final_map(None, X, F, V)):.5f}")
for k, V, t, move in cmcf_steps(X, F):
    pass
Sc = final_map(None, X, F, V)


def flips(S):
    n = np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]])
    return int(((n * S[F].mean(1)).sum(1) < 0).sum())


for name, M in [("conformal_sphere", S), ("cMCF dt=2", Sc)]:
    d = triangle_distortion(X, F, M)
    print(f"{name}: distortion median {np.median(d):.3f} p99 {np.percentile(d, 99):.3f} max {d.max():.3g}; flipped triangles {flips(M)}")
