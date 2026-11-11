"""map.png: the bunny with a lat/lon grid carried back through the conformal map, an arrow, and the sphere."""
from lib import *

X, F = bunny()
st, S, info = bunny_map(X, F)
S = aligned(S, *view_poles(X))
mesh = Scene(X, F)
sph, U = unit_sphere()

fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.35})
mesh.draw(axs[0], grid_sets(mesh, S, lw=1.2))
sph.draw(axs[1], grid_sets(sph, U, lw=1.2))
axs[0].text(0.5, -0.04, r"$\Sigma$", transform=axs[0].transAxes, ha="center", va="top", fontsize=20)
axs[1].text(0.5, -0.04, r"$S^2$", transform=axs[1].transAxes, ha="center", va="top", fontsize=20)
fig.canvas.draw()
a0, a1 = axs[0].get_position(), axs[1].get_position()
fig.add_artist(matplotlib.patches.FancyArrowPatch(
    (a0.x1 + 0.01, 0.62), (a1.x0 - 0.01, 0.62), transform=fig.transFigure,
    connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=22, lw=2.2, color="k"))
fig.text((a0.x1 + a1.x0) / 2, 0.86, r"$f$", ha="center", fontsize=22)
save(fig, "map.png")
