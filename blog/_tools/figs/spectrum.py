"""spectrum.png: the 36 lowest eigenvalues of (L0, sphere area) for two maps of the bunny."""
from lib import *
from lib import _gen_eig

X, F = bunny()
_, S_cs, info = bunny_map(X, F)
for k, V, t, move in cmcf_steps(X, F):
    pass
S_cm = final_map(None, X, F, V)
L0, _ = cotan_laplacian(X, F)
n = 36
ideal = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(6)])[:n]
fig, ax = plt.subplots(figsize=(8, 4.7))
ax.step(np.arange(n), ideal, where="mid", color="0.6", lw=2.2, label=r"sphere: $\ell(\ell+1)$")
for name, S, mk, fc in [(f"cMCF, $\\delta = 2$, {k} steps", S_cm, "^", "none"),
                        ("conformal_sphere", S_cs, "o", "k")]:
    lam, _ = _gen_eig(L0, vertex_areas(S, F), n)
    ax.plot(np.arange(n), lam, mk, mfc=fc, mec="k", color="k", ms=6,
            label=f"{name} (error {spectrum_error(X, F, S):.4f})", ls="none")
ax.set_xlabel("eigenvalue index")
ax.set_ylabel("eigenvalue of ($L_0$, sphere area)")
ax.legend(frameon=False, loc="upper left")
save(fig, "spectrum.png")
