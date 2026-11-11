"""coords.png: level sets of the three l=1 eigenfunctions (x, y, z of the converged map).

Top: on the bunny. Bottom: the same levels on the unit sphere, in one fixed camera, so the three differ only in which coordinate is cut. Levels are uniform in
the coordinate, which cuts the sphere into equal-area bands (Archimedes), so the
spacing of the curves on the bunny shows the area scale w. Solid = positive, dashed =
negative, heavy = zero; the positive side is shaded."""
from lib import *

LEVELS = np.round(np.arange(-0.8, 0.81, 0.2), 2)
X, F = bunny()
st, S, info = bunny_map(X, F)
S = aligned(S, *view_poles(X))
mesh = Scene(X, F)

sph, U = unit_sphere(rot(-35, 25))                  # one camera for all three, so the axes differ
fig, axs = plt.subplots(2, 3, figsize=(12, 7.6), gridspec_kw={"hspace": 0.02, "wspace": 0.02})
for c, name in enumerate("xyz"):
    mesh.fill(axs[0, c], np.where(S[F, c].mean(1) > 0, 0.86, 1.0))
    mesh.draw(axs[0, c], levels_sets(mesh, S[:, c], LEVELS))
    sph.fill(axs[1, c], np.where(U[sph.F, c].mean(1) > 0, 0.86, 1.0))
    sph.draw(axs[1, c], levels_sets(sph, U[:, c], LEVELS))
    axs[0, c].set_title(f"${name}$", fontsize=16, pad=2)
save(fig, "coords.png")
