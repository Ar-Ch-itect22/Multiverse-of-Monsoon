"""Physics exam: live checks of how the model behaves, run on the chosen start day. Not graded.

Every check goes through the same call the graded server uses. Rows that fail are returned and shown, not hidden.
The first group are rules the model must obey. The second group are things real weather tends to do; the model
was never told them, so a fail there is a finding about the model, not a bug.
"""
from __future__ import annotations

import numpy as np

NC, NLAT, NLON = 6, 32, 36
T2M, RAIN, U, V, MSL, Q = 0, 1, 2, 3, 4, 5
LAT = 6.5 + np.arange(NLAT)
LON = 65.5 + np.arange(NLON)
BAY = ((LAT >= 8) & (LAT <= 20))[:, None] & ((LON >= 82) & (LON <= 94))[None, :]        # the organizers' Bay of Bengal box
SEA = ((LAT >= 8) & (LAT <= 20))[:, None] & ((LON >= 65) & (LON <= 75))[None, :]        # the organizers' Arabian Sea box


def run(engine, init: np.ndarray, sc) -> list:
    sig = engine.sigma.numpy().reshape(NC, 1, 1)
    rows = []

    def add(group, name, ok, detail):
        rows.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})

    def sim(delta=None, steps=10):
        return engine.simulate(init, sc, steps, delta)

    def size(d):                                                           # response size, rain left out (it is not in these units)
        z = d / sig
        return float(np.sqrt((np.delete(z, RAIN, axis=-3) ** 2).mean()))

    def box_delta(mask, **kw):
        d = np.zeros((NC, NLAT, NLON), "f4")
        for c, (add_v, mul_v) in kw.items():
            d[int(c[1:])][mask] = add_v + mul_v * init[-1][int(c[1:])][mask]
        return d

    base = sim()
    # ---- rules the model must obey -------------------------------------------------------------------------
    add("Rules", "A zero change gives the same answer", float(np.abs(sim(np.zeros((NC, NLAT, NLON), "f4")) - base).max()) == 0.0,
        "largest difference: 0" if float(np.abs(sim(np.zeros((NC, NLAT, NLON), "f4")) - base).max()) == 0.0 else "differs")
    add("Rules", "The same input twice gives the same answer", float(np.abs(sim() - base).max()) == 0.0, "bit for bit")
    ok = np.isfinite(base).all() and (base[:, RAIN] >= 0).all() and (base[:, T2M] > 233.15).all() and (base[:, T2M] < 333.15).all()
    add("Rules", "Ten days stay valid", ok, "no bad numbers, no negative rain, temperature in range")

    one = np.zeros((NC, NLAT, NLON), "f4")
    one[T2M, 16, 18] = 3.0
    d = sim(one, 3) - base[:3]
    ii, jj = np.indices((NLAT, NLON))
    dist = np.maximum(np.abs(ii - 16), np.abs(jj - 18))
    reach = int(engine.net.reach)
    leak = max(float(np.abs(d[k][:, dist > reach * (k + 1)]).max(initial=0.0)) for k in range(3))
    far = [int(dist[np.abs(d[k]).max(0) > 0].max()) for k in range(3)]
    add("Rules", "A change to one square cannot jump across the map", leak == 0.0,
        f"furthest square touched on days 1, 2, 3: {far[0]}, {far[1]}, {far[2]} (limit {reach}, {2 * reach}, {3 * reach})")

    long = sim(None, 30)
    ok = np.isfinite(long).all() and (long[:, RAIN] >= 0).all()
    ratio = [float(long[29, c].std() / max(long[0, c].std(), 1e-6)) for c in range(NC)]
    add("Rules", "Thirty days in a row do not blow up or go flat", ok and min(ratio) > 0.4 and max(ratio) < 2.5,
        "map spread on day 30 against day 1, per field: " + ", ".join(f"{r:.2f}" for r in ratio))

    # ---- behaviour we hope for -----------------------------------------------------------------------------
    warm = sim(box_delta(BAY, c0=(3.0, 0.0), c5=(0.0, 0.15)))
    cool = sim(box_delta(BAY, c0=(-3.0, 0.0), c5=(0.0, -0.15)))
    a, b = (warm[0] - base[0]) / sig, (cool[0] - base[0]) / sig
    corr = float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))
    add("Behaviour", "Opposite changes give opposite responses", corr < -0.5, f"match between the two day-1 responses: {corr:+.2f} (-1 is a perfect mirror)")

    sizes = [size(sim(box_delta(BAY, c0=(k, 0.0)), 2)[1] - base[1]) for k in (1.0, 2.0, 3.0, 4.0)]
    add("Behaviour", "A bigger change gives a bigger response", all(sizes[k] < sizes[k + 1] for k in range(3)),
        "response size for +1, +2, +3, +4 deg C: " + ", ".join(f"{s:.3f}" for s in sizes))

    peak = [size(warm[k] - base[k]) for k in range(10)]
    add("Behaviour", "A response fades in time instead of growing without limit", peak[9] < max(peak),
        f"size on day 1: {peak[0]:.3f}, largest: {max(peak):.3f} on day {int(np.argmax(peak)) + 1}, day 10: {peak[9]:.3f}")

    wet = sim(box_delta(BAY, c5=(0.0, 0.15)), 2)
    dr = float((wet[:, RAIN][:, BAY] - base[:2, RAIN][:, BAY]).mean())
    add("Behaviour", "More moisture in the air gives more rain there", dr > 0, f"rain in the box, days 1 to 2: {dr:+.2f} mm/day")

    hot = sim(box_delta(BAY, c0=(3.0, 0.0)), 2)
    dp = float((hot[:, MSL][:, BAY] - base[:2, MSL][:, BAY]).mean()) / 100.0
    add("Behaviour", "Warmer air over the sea lowers the pressure there", dp < 0, f"pressure in the box, days 1 to 2: {dp:+.2f} hPa")

    calm = sim(box_delta(SEA, c2=(0.0, -0.4), c3=(0.0, -0.4)), 2)
    sp = lambda x: np.hypot(x[:, U], x[:, V])[:, SEA].mean()                # noqa: E731
    dw = float(sp(calm) - sp(base[:2]))
    add("Behaviour", "A stalled wind stays weaker for a while", dw < 0, f"wind speed in the box, days 1 to 2: {dw:+.2f} m/s")

    cut = float((calm[:, RAIN] - base[:2, RAIN])[:, :, 6:14][:, 8:20].mean())  # the west coast strip, lat 14.5 to 25.5, lon 71.5 to 78.5
    add("Behaviour", "A stalled monsoon wind brings less rain to the west coast", cut < 0, f"rain over the west coast strip, days 1 to 2: {cut:+.2f} mm/day")
    return rows
