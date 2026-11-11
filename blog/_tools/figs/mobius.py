"""mobius.png: a Möbius zoom of the sphere gives a Möbius zoom of the bunny.

Top: a checkerboard on the disk of radius ``lam`` in the stereographic plane about the head's direction
n (at lam = 1, the hemisphere around n), with its cells scaled by ``lam``; changing lam is a boost
along n. Bottom: each pattern pulled back to the bunny through the same map f. Pulling the zoomed
pattern back through f is pulling the original pattern back through a boosted map psi o f, so every
bottom panel is another conformal map of the bunny."""
from lib import *

ZOOMS = [0.25, 0.5, 1.0, 2.0, 4.0]
GRID = 0.25             # 8 cells across the unit disk
SHADE = np.array([1.0, 0.62, 0.90])     # cell 0, cell 1, outside the disk
RES = 1400              # render resolution per panel, downsampled on save

X, F = bunny()
st, S, info = bunny_map(X, F)
S = aligned(S, *view_poles(X))
mesh = Scene(X, F)

head = bunny_parts(X, F) > 0
n = S[F[head]].mean((0, 1))
n /= np.linalg.norm(n)
e1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(n, e1)

# Sphere camera: looking at n from 25 degrees below it, so n sits above the centre of the disk.
tilt = np.radians(25)
cz = np.cos(tilt) * n - np.sin(tilt) * e2
cy = np.cross(cz, e1)
cy *= np.sign(cy @ n)
R_sph = np.array([np.cross(cy, cz), cy, cz])
sph, _ = unit_sphere(R_sph)


def zoom_checker(p, lam):
    """0/1 squares on the disk of radius lam in the stereographic plane about n; 2 outside it."""
    u = np.stack([p @ e1, p @ e2], 1) / np.maximum(1 + p @ n, 1e-9)[:, None] / lam
    c = np.floor(u / GRID).astype(int).sum(1) % 2
    return np.where((u ** 2).sum(1) > 1, 2, c)


def raster(P, box, res):
    """Front-facing rasterization of the mesh (P, F): per pixel the triangle id (-1 if none)
    and the first two barycentric coordinates."""
    x0, y0, w = box
    q_all = (P[..., :2] - np.array([x0, y0])) / w * (res - 1)
    t = P[F]
    front = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2] > 0
    buf = np.full((res, res), -np.inf)
    tid = np.full((res, res), -1)
    bar = np.zeros((res, res, 2))
    for k in np.nonzero(front)[0]:
        q, z3 = q_all[F[k]], P[F[k], 2]
        lo = np.maximum(np.floor(q.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(q.max(0)).astype(int), res - 1)
        if (hi < lo).any():
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        d = (q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0]) + (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1])
        if abs(d) < 1e-12:
            continue
        l0 = ((q[1, 1] - q[2, 1]) * (xs - q[2, 0]) + (q[2, 0] - q[1, 0]) * (ys - q[2, 1])) / d
        l1 = ((q[2, 1] - q[0, 1]) * (xs - q[2, 0]) + (q[0, 0] - q[2, 0]) * (ys - q[2, 1])) / d
        m = (l0 >= -1e-6) & (l1 >= -1e-6) & (1 - l0 - l1 >= -1e-6)
        z = l0 * z3[0] + l1 * z3[1] + (1 - l0 - l1) * z3[2]
        sl = (slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
        upd = m & (z > buf[sl])
        buf[sl][upd] = z[upd]
        tid[sl][upd] = k
        bar[sl][upd] = np.stack([l0, l1], -1)[upd]
    return tid, bar


def show(ax, img, box):
    x0, y0, w = box
    rgba = np.ones(img.shape + (4,))
    rgba[..., :3] = np.nan_to_num(img, nan=1.0)[..., None]
    rgba[..., 3] = ~np.isnan(img)
    ax.imshow(rgba, extent=(x0, x0 + w, y0, y0 + w), origin="lower", interpolation="antialiased")


def dot(ax, p):
    ax.plot(p[0], p[1], "o", ms=9, mfc="k", mec="w", mew=2)


# Bunny pixels: where f sends each visible point of the bunny.
tid, bar = raster(mesh.P, mesh.box, RES)
on_bunny = tid >= 0
T = S[F[tid[on_bunny]]]
l = bar[on_bunny]
f_px = l[:, :1] * T[:, 0] + l[:, 1:] * T[:, 1] + (1 - l.sum(1, keepdims=True)) * T[:, 2]
f_px /= np.linalg.norm(f_px, axis=1, keepdims=True)
v_n = int(np.argmax(S @ n))             # the vertex f sends closest to n

# Sphere pixels: the visible point of the unit sphere under each pixel.
x0, y0, w = sph.box
u = x0 + (np.arange(RES) + 0.5) / RES * w
xx, yy = np.meshgrid(u, u)
on_sphere = xx ** 2 + yy ** 2 < 1
s_px = np.stack([xx, yy, np.sqrt(np.maximum(1 - xx ** 2 - yy ** 2, 0))], -1)[on_sphere] @ R_sph

fig, axs = plt.subplots(2, len(ZOOMS), figsize=(15, 6.6), gridspec_kw={"hspace": 0.08, "wspace": 0.02})
for j, lam in enumerate(ZOOMS):
    img = np.full(on_sphere.shape, np.nan)
    img[on_sphere] = SHADE[zoom_checker(s_px, lam)]
    show(axs[0, j], img, sph.box)
    sph.draw(axs[0, j], [])
    dot(axs[0, j], R_sph @ n)

    img = np.full(on_bunny.shape, np.nan)
    img[on_bunny] = SHADE[zoom_checker(f_px, lam)]
    show(axs[1, j], img, mesh.box)
    mesh.draw(axs[1, j], [])
    if mesh.visible(mesh.P[v_n][None])[0]:
        dot(axs[1, j], mesh.P[v_n])

    lab = {0.25: "¼", 0.5: "½"}.get(lam, "%g" % lam)
    axs[0, j].set_title("zoom " + lab, fontsize=22)
for a, name in ((axs[0, 0], "zoomed on the sphere"), (axs[1, 0], "pulled back by $f$")):
    a.text(-0.07, 0.5, name, transform=a.transAxes, rotation=90, ha="center", va="center", fontsize=19)
save(fig, "mobius.png")
