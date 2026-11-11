"""iterates.png: the fixed-point iteration after its first step (w = 1, started from the input's
coordinates), at three intermediate steps and at convergence.

The intermediate steps are spaced evenly in how far the image of the map is from a sphere (the
spread of |f| between the vertices), which is where the visible change happens; equal step numbers
or equal progress in the density would put them after the map is already round. Rows: lat/lon grid
pulled back to the bunny (the iterate projected to the sphere, for display only); the image of the
bunny under the raw eigenfunctions with the same grid; equal spherical circles pulled back to the bunny."""
from lib import *

X, F = bunny()
st = setup(X, F)
Ys, res, _ = conformal_iterates(st)
v0, v1 = view_poles(X)
final = final_map(st, X, F, Ys[-1])
anchor = aligned(final, v0, v1)
# The intermediate columns are spaced evenly in how far the image of the map is from a sphere (the
# spread of |f|): the first steps at which that spread is down to 3/4, 1/2 and 1/4 of its start.
spread = np.array([np.ptp(np.linalg.norm(Y, axis=1)) for Y in Ys])
idx = [int(np.argmax(spread <= spread[0] * q)) for q in (0.75, 0.5, 0.25)]
# Ys[k] is the map after the (k+1)-th eigenvector step, which the convergence plot calls iteration k+1
picks = [(0, "iteration 1 ($w = 1$)")] + [(k, f"iteration {k + 1}") for k in idx] + \
        [(len(Ys) - 1, f"iteration {len(Ys)} (converged)")]
print("picked steps", [k for k, _ in picks], "of", len(Ys))

mesh = Scene(X, F)
R3 = [Ys[k] for k, _ in picks]
# the middle row is drawn at true scale, the same in every column, with the unit circle dashed
half = 1.03 * max(np.abs((r @ rot(-35, 20).T)[:, :2]).max() for r in R3)
box = (-half, -half, 2 * half)
circle = np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200))], 1)
fig, axs = plt.subplots(3, 5, figsize=(18, 11), gridspec_kw={"wspace": 0.02, "hspace": 0.2})
for j, ((k, title), raw) in enumerate(zip(picks, R3)):
    S = to_sphere(raw)
    S = aligned(S if k == len(Ys) - 1 else S, v0, v1)
    mesh.draw(axs[0, j], grid_sets(mesh, S, lw=1.3))
    img = Scene(raw, F, rot(-35, 20), box=box)
    axs[1, j].plot(*circle.T, color="0.55", lw=1.2, ls=(0, (4, 3)))
    img.draw(axs[1, j], grid_sets(img, S, lw=1.3))
    r = np.linalg.norm(raw, axis=1)
    axs[1, j].text(0.5, -0.03, f"$|\\,f\\,|$ from {r.min():.3f} to {r.max():.3f}", transform=axs[1, j].transAxes,
                   ha="center", va="top", fontsize=13)
    mesh.draw(axs[2, j], circle_sets(mesh, S, lw=1.2))
    axs[0, j].set_title(title, fontsize=15)
for r, label in enumerate(["grid on the bunny", "image of the map", "circles on the bunny"]):
    axs[r, 0].text(-0.02, 0.5, label, transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=14)
save(fig, "iterates.png")
