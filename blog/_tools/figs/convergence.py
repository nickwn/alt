"""convergence.png: relative change of the density per iteration, plain/Anderson x gauge/no gauge."""
from lib import *

X, F = bunny()
st = setup(X, F)
runs = [("plain, no gauge", dict(gauge=False, m_hist=0), ":", 1.8),
        ("Anderson, no gauge", dict(gauge=False, m_hist=6), "-.", 1.3),
        ("plain, gauge", dict(gauge=True, m_hist=0), "--", 1.8),
        ("Anderson, gauge", dict(gauge=True, m_hist=6), "-", 2.6)]
fig, ax = plt.subplots(figsize=(8.3, 4.7))
for name, kw, ls, lw in runs:
    _, res, _ = conformal_iterates(st, max_iter=200, tol=1e-10, **kw)
    ax.semilogy(np.arange(1, len(res) + 1), res, "k", ls=ls, lw=lw, label=name)
ax.axhline(1e-7, color="0.6", lw=1, zorder=0)
ax.text(200, 1.4e-7, "tolerance used in the post", ha="right", va="bottom", fontsize=10, color="0.4")
ax.set_xlabel("iteration")
ax.set_ylabel("relative change of the density")
ax.legend(frameon=False, loc="upper right")
save(fig, "convergence.png")
