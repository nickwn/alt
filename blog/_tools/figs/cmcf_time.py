"""cmcf_time.png: spectrum error against wall-clock time, cMCF (several step sizes) vs conformal_sphere.
Every point is scored after the same finish (projection, orientation, Möbius centring), which is not timed."""
from lib import *

X, F = bunny()
pts_s = []
S, T = conformal_sphere_timed(X, F, on_step=lambda t, f: pts_s.append((t, f)))
pts_s = [(t, spectrum_error(X, F, final_map(None, X, F, f))) for t, f in pts_s]
fig, ax = plt.subplots(figsize=(8.2, 4.7))
end = pts_s[-1][0]
for dt, mk in [(0.2, "s"), (1.0, "v"), (2.0, "^")]:
    pts = [(t, spectrum_error(X, F, final_map(None, X, F, V))) for k, V, t, move in cmcf_steps(X, F, dt=dt)]
    ax.semilogy(*zip(*pts), "-" + mk, color="0.45", mfc="none", lw=1.0, ms=4, label=f"cMCF, $\\delta = {dt:g}$")
    end = max(end, pts[-1][0])
    print(f"cMCF dt={dt}: {len(pts)} steps, {pts[-1][0]:.2f}s, error {pts[-1][1]:.5f}")
ax.semilogy(*zip(*pts_s), "-o", color="k", lw=2.4, ms=4, label="conformal_sphere")
ax.set_xlim(0, end * 1.03)
ax.set_ylim(top=1)
ax.set_xlabel("seconds")
ax.set_ylabel("spectrum error")
ax.legend(frameon=False, loc="upper right")
save(fig, "cmcf_time.png")
print(f"conformal_sphere: {T['iters']} iterations, {pts_s[-1][0]:.2f}s, error {pts_s[-1][1]:.5f}")
