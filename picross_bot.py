#!/usr/bin/env python3
"""
Picross Touch auto-solver: reads the puzzle off your screen, solves it, and fills it in.

    python picross_bot.py        then, in the game:
      F8 solve   F7 check/fix the clues first   F10 auto mode on/off
      F6 teach a "next level" button             F9 stop          (Ctrl+C here to quit)

See README.md for setup and how it works. Sections below: solver, reading the screen
(board finding, digit extraction, recognition, reading clues), clue editor, mouse
input, taught buttons, app.
"""
import argparse
import itertools
import os
import re
import queue
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITS_PATH = os.path.join(HERE, "picross_digits.npz")
DEBUG_PATH = os.path.join(HERE, "picross_last.png")
CAPTURE_PATH = os.path.join(HERE, "picross_capture.png")
BUTTONS_DIR = os.path.join(HERE, "picross_buttons")

UNKNOWN, EMPTY, FILLED = -1, 0, 1

# Reader tuning
INK = 40        # colour difference (summed over B,G,R) from local background that counts as text
GLYPH = 32      # glyphs are normalised onto a GLYPH x GLYPH canvas before matching
MATCH = 0.30    # max chamfer distance (canvas pixels) to accept a template match...
MARGIN = 0.6    # ...and it must be under MARGIN x the distance to the best *other* label
GAP = 0.35      # in row clues, digits closer than GAP x digit height form one number
HEIGHT_TOL = 1.3  # only compare against stored digits within this size ratio
CLF_MIN = 0.5     # classifier answers at least this confident are used provisionally...
FAMILIAR = 0.6    # ...if the glyph is also within this distance of that digit's examples
RESCALE_HEIGHTS = np.geomspace(7, 100, 18)   # digit heights (px) to synthesise when learning


class ReadError(Exception):
    pass


# =============================================================== solver

def line_clues(cells):
    """Run lengths of filled cells, e.g. [1,1,0,1] -> [2, 1]."""
    out, run = [], 0
    for v in cells:
        if v == FILLED:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def solve_line(line, clues):
    """Fix every cell of `line` that is forced by `clues`. Returns the new line,
    or None if no arrangement fits. O(len(line) * len(clues))."""
    n, k = len(line), len(clues)
    if k == 0:
        return None if FILLED in line else [EMPTY] * n

    pe = [0] * (n + 1)                      # prefix count of known-empty cells
    for i, v in enumerate(line):
        pe[i + 1] = pe[i] + (v == EMPTY)
    fits = []                               # fits[j][i]: block j can sit at cells i..i+L-1
    for L in clues:
        f = [False] * (n + 1)
        for i in range(n - L + 1):
            if pe[i + L] == pe[i] and (i + L == n or line[i + L] != FILLED):
                f[i] = True
        fits.append(f)

    # state (j, i): j blocks placed, first i cells consumed (incl. separator)
    fwd = [[False] * (n + 1) for _ in range(k + 1)]
    fwd[0][0] = True
    for j in range(k + 1):
        fj = fwd[j]
        for i in range(n + 1):
            if not fj[i]:
                continue
            if i < n and line[i] != FILLED:
                fj[i + 1] = True
            if j < k and fits[j][i]:
                fwd[j + 1][min(i + clues[j] + 1, n)] = True

    bwd = [[False] * (n + 1) for _ in range(k + 1)]
    bwd[k][n] = True
    for j in range(k, -1, -1):
        bj = bwd[j]
        for i in range(n, -1, -1):
            if bj[i]:
                continue
            if i < n and line[i] != FILLED and bj[i + 1]:
                bj[i] = True
            elif j < k and fits[j][i] and bwd[j + 1][min(i + clues[j] + 1, n)]:
                bj[i] = True
    if not bwd[0][0]:
        return None

    can_empty = [False] * n
    fill_diff = [0] * (n + 1)
    for j in range(k + 1):
        for i in range(n + 1):
            if not fwd[j][i]:
                continue
            if i < n and line[i] != FILLED and bwd[j][i + 1]:
                can_empty[i] = True
            if j < k and fits[j][i]:
                L = clues[j]
                if bwd[j + 1][min(i + L + 1, n)]:
                    fill_diff[i] += 1
                    fill_diff[i + L] -= 1
                    if i + L < n:
                        can_empty[i + L] = True

    out, run = list(line), 0
    for i in range(n):
        run += fill_diff[i]
        cf, ce = run > 0, can_empty[i]
        if cf and not ce:
            out[i] = FILLED
        elif ce and not cf:
            out[i] = EMPTY
        elif not cf and not ce:
            return None
    return out


def _propagate(grid, rows, cols):
    R, C = len(rows), len(cols)
    dirty_r, dirty_c = set(range(R)), set(range(C))
    while dirty_r or dirty_c:
        while dirty_r:
            r = dirty_r.pop()
            new = solve_line(grid[r], rows[r])
            if new is None:
                return False
            for c in range(C):
                if new[c] != grid[r][c]:
                    grid[r][c] = new[c]
                    dirty_c.add(c)
        while dirty_c:
            c = dirty_c.pop()
            new = solve_line([grid[r][c] for r in range(R)], cols[c])
            if new is None:
                return False
            for r in range(R):
                if new[r] != grid[r][c]:
                    grid[r][c] = new[r]
                    dirty_r.add(r)
    return True


def solve(rows, cols, max_guesses=20000):
    """Solve a nonogram. rows/cols are lists of clue lists. Returns a grid of 0/1 or None."""
    budget = [max_guesses]

    def search(grid):
        if not _propagate(grid, rows, cols):
            return None
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v == UNKNOWN:
                    for guess in (FILLED, EMPTY):
                        budget[0] -= 1
                        if budget[0] < 0:
                            return None
                        g2 = [x[:] for x in grid]
                        g2[r][c] = guess
                        res = search(g2)
                        if res is not None:
                            return res
                    return None
        return grid

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    grid = search([[UNKNOWN] * len(cols) for _ in rows])
    if grid is None:
        return None
    if [line_clues(r) for r in grid] != rows:
        return None
    if [line_clues([grid[r][c] for r in range(len(rows))]) for c in range(len(cols))] != cols:
        return None
    return grid


# =============================================================== reading the screen

class Geometry:
    def __init__(self, box, clue, rows, cols):
        self.x0, self.y0, self.x1, self.y1 = map(float, box)
        self.cx, self.cy = map(float, clue)
        self.rows, self.cols = rows, cols
        self.cw = (self.x1 - self.x0) / cols
        self.ch = (self.y1 - self.y0) / rows
        self.cell = min(self.cw, self.ch)

    def center(self, r, c):
        return self.x0 + (c + 0.5) * self.cw, self.y0 + (r + 0.5) * self.ch

    # Strips span the whole cell: wide numbers like "10" reach the grid lines, and the
    # lines are erased before digits are looked for (see components()).
    def col_strip(self, c):  # (xa, ya, xb, yb) of the clues above column c
        gap = max(2.0, 0.06 * self.ch)
        return (int(self.x0 + c * self.cw), int(self.cy),
                int(self.x0 + (c + 1) * self.cw), int(self.y0 - gap))

    def row_strip(self, r):  # clues left of row r
        gap = max(2.0, 0.06 * self.cw)
        return (int(self.cx), int(self.y0 + r * self.ch),
                int(self.x0 - gap), int(self.y0 + (r + 1) * self.ch))


def _count_cells(profile, lo, hi):
    """Count cells along one axis from a brightness profile across the grid.
    lo/hi: the grid edges (from calibration) as indices into `profile`.
    Returns (cells, evenly_spaced)."""
    level = np.median(profile[lo:hi])               # empty cells dominate inside the box
    dev = np.abs(profile - level)
    thr = max(12.0, 0.3 * dev.max())
    idx = np.flatnonzero(dev > thr)
    centers = [g.mean() for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(g)]
    if len(centers) >= 2:
        edge = max(6.0, 0.3 * np.diff(centers).min())   # bands this close to an edge are the border
    else:
        edge = 6.0
    interior = [c for c in centers if lo + edge < c < hi - edge]
    pts = np.array([lo] + interior + [hi], dtype=float)
    d = np.diff(pts)
    even = bool(d.max() < 1.5 * d.min())
    if len(d) >= 3 and not even:
        # a faint line was missed (a double-width gap): count by the typical spacing
        return max(1, int(round((hi - lo) / np.median(d)))), False
    return len(d), even


def detect_size(gray, box, pad=12):
    """Grid size from the lines between cells. Works for single-row/column boards too."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    H, W = gray.shape
    g = gray.astype(np.float32)
    ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
    cols, even_x = _count_cells(g[y0:y1, ax0:min(W, x1 + pad)].mean(axis=0), x0 - ax0, x1 - ax0)
    rows, even_y = _count_cells(g[ay0:min(H, y1 + pad), x0:x1].mean(axis=1), y0 - ay0, y1 - ay0)
    w, h = x1 - x0, y1 - y0
    px, py = w / cols, h / rows
    if abs(px - py) > 0.2 * min(px, py):
        # Cells should be square; one axis probably missed some faint lines.
        # Trust the axis whose lines are evenly spaced and closer together.
        good = [p for p, n, ok in ((px, cols, even_x), (py, rows, even_y)) if ok and n > 1]
        if not good:
            raise ReadError("couldn't find grid lines - recalibrate (F7) or pass --size")
        p = min(good)
        rows, cols = max(1, round(h / p)), max(1, round(w / p))
    if not (1 <= rows <= 100 and 1 <= cols <= 100):
        raise ReadError(f"grid size looks wrong ({rows}x{cols}) - pass --size")
    return rows, cols


def _first_band(frac, thr=0.85):
    """(start, end) of the first run of positions where frac > thr."""
    idx = np.flatnonzero(frac > thr)
    if len(idx) == 0:
        raise ReadError("no separator line")
    start = end = int(idx[0])
    while end + 1 < len(frac) and frac[end + 1] > thr:
        end += 1
    return start, end


def _board_layout(gray, frame):
    """Inside a board's outer frame, find where the clues start and where the grid is.
    In Picross Touch the row clues sit in one wide cell per row and the column clues in
    one tall cell per column, so the first line running the full height (from the left)
    and the first running the full width (from the top) are the clue/grid separators."""
    x0, y0, x1, y1 = frame
    g = gray[y0:y1, x0:x1].astype(np.float32)
    dark = g < np.median(g) - 50                  # lines, frame, digit outlines

    def inner(f):                                 # step past the frame on both sides
        lo, hi = 0, len(f)
        while lo < hi and f[lo] > 0.6:
            lo += 1
        while hi > lo and f[hi - 1] > 0.6:
            hi -= 1
        return lo, hi
    ix0, ix1 = inner(dark.mean(axis=0))
    iy0, iy1 = inner(dark.mean(axis=1))
    if ix1 - ix0 < 30 or iy1 - iy0 < 30:
        raise ReadError("frame has no inside")
    d = dark[iy0:iy1, ix0:ix1]
    sx0, sx1 = _first_band(d.mean(axis=0))
    sy0, sy1 = _first_band(d.mean(axis=1))
    if sx0 < 5 or sy0 < 5 or ix1 - ix0 - sx1 < 10 or iy1 - iy0 - sy1 < 10:
        raise ReadError("no clue area")
    grid = (x0 + ix0 + sx1 + 1, y0 + iy0 + sy1 + 1, x0 + ix1, y0 + iy1)
    return grid, (x0 + ix0, y0 + iy0)


def locate_board(img):
    """Find the puzzle on a screenshot from its black outer frame.
    Returns (grid box, clue-area corner, frame box), all in image coordinates."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n, lab, st, _ = cv2.connectedComponentsWithStats((gray < 50).astype(np.uint8), connectivity=8)
    frames = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if w < 60 or h < 60 or area > 0.5 * w * h:   # too small, or solid (not a frame)
            continue
        # The frame's edges are the rows and columns of the blob that run (nearly) its
        # full length. Using those rather than the blob's bounding box ignores anything
        # stuck to the outside of the frame, like the star badge on workshop levels.
        sub = lab[y:y + h, x:x + w] == i
        row_cover, col_cover = sub.mean(axis=1), sub.mean(axis=0)
        rows = np.flatnonzero(row_cover >= 0.8 * row_cover.max())
        cols = np.flatnonzero(col_cover >= 0.8 * col_cover.max())
        if len(rows) < 2 or len(cols) < 2:
            continue
        fx0, fx1, fy0, fy1 = x + cols[0], x + cols[-1] + 1, y + rows[0], y + rows[-1] + 1
        if fx1 - fx0 < 60 or fy1 - fy0 < 60:          # needs two edges far apart each way
            continue
        frames.append(((fx1 - fx0) * (fy1 - fy0), (int(fx0), int(fy0), int(fx1), int(fy1))))
    for _, frame in sorted(frames, reverse=True):
        try:
            grid, clue = _board_layout(gray, frame)
            return grid, clue, frame
        except ReadError:
            continue
    raise ReadError("couldn't find the puzzle on screen")


def ink_mask(strip, bg=None):
    """Pixels that differ from the strip's dominant (background) colour.
    Each clue strip sits on one flat-coloured cell, so its median colour is the
    background whatever the theme or highlight; digits cover well under half of it.
    Returns (mask, background colour)."""
    if bg is None:
        bg = np.median(strip.reshape(-1, 3), axis=0).astype(np.int16)
    diff = np.abs(strip.astype(np.int16) - bg).sum(axis=2)
    return (diff > INK).astype(np.uint8), bg


def normalize(glyph):
    """Scale a glyph mask to fit the canvas (keeping aspect ratio) -> uint8 0/1."""
    h, w = glyph.shape
    s = (GLYPH - 4) / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    g = cv2.resize(glyph.astype(np.float32), (nw, nh), interpolation=interp)
    canvas = np.zeros((GLYPH, GLYPH), np.uint8)
    oy, ox = (GLYPH - nh) // 2, (GLYPH - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = g >= 0.35
    return canvas


def _dt2(mask):
    """Squared distance from every pixel to the nearest ink pixel of `mask`."""
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3) ** 2


def glyph_dist(a, b):
    """Symmetric RMS chamfer distance between two glyph masks (tolerant of stroke width)."""
    if not a.any() or not b.any():
        return float("inf")
    return 0.5 * (np.sqrt(_dt2(b)[a > 0].mean()) + np.sqrt(_dt2(a)[b > 0].mean()))


def _glyph(crop, mask):
    """The shape used for recognition: the darker part of a digit (its outline).
    Picross Touch draws white digits with thick black outlines that also fill the holes,
    so the whole digit is a featureless blob - but the outline keeps every detail
    (an 8's two inner slits, a 3's notches). Splitting the digit's own pixels into dark
    and light (Otsu) also works for greyed-out clues and for plain dark text."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    vals = gray[mask > 0].reshape(-1, 1)
    if len(vals) >= 10 and int(vals.max()) - int(vals.min()) > 30:
        t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (mask > 0) & (gray <= t)
        if dark.sum() >= 0.15 * (mask > 0).sum():
            return normalize(dark)
    return normalize(mask > 0)


def _erase_lines(mask, cell):
    """Remove grid lines from an ink mask: thin straight runs longer than any digit
    (95% of a cell, or of the strip if it's shorter). A digit touching a line would
    otherwise merge with it into one long blob and be thrown away as a line.
    Only *thin* runs count: a wide "10" touching the lines on both sides makes a long
    horizontal band too, but a thick one."""
    H, W = mask.shape
    thin = max(3, int(0.2 * cell))   # thick 5-block / clue separators are ~15% of a cell
    lines = np.zeros_like(mask)
    for vertical, length in ((True, int(0.95 * min(H, cell))), (False, int(0.95 * min(W, cell)))):
        if length < 10:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length) if vertical else (length, 1))
        runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(runs, connectivity=4)
        for i in range(1, n):
            x, y, w, h = (int(v) for v in st[i, :4])
            if (w if vertical else h) > thin:                  # too thick to be a line
                continue
            # A real line has background beside it; a sliver where touching digits
            # happen to form a straight run has digit ink on both sides.
            if vertical:
                sides = [mask[y:y + h, x - 1] if x > 0 else None,
                         mask[y:y + h, x + w] if x + w < W else None]
            else:
                sides = [mask[y - 1, x:x + w] if y > 0 else None,
                         mask[y + h, x:x + w] if y + h < H else None]
            if any(s is None or s.mean() < 0.5 for s in sides):
                lines[lab == i] = 1
    # also take the anti-aliased fringe on each side of a line
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return mask & (1 - lines)


def components(strip, cell, ox=0, oy=0):
    """Candidate glyphs in one clue strip (BGR image), as dicts in strip coordinates."""
    H, W = strip.shape[:2]
    if H < 3 or W < 3:
        return []
    mask, bg = ink_mask(strip)
    mask = _erase_lines(mask, cell)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 6 or h < 0.2 * cell or h > 1.05 * cell or w > 2.5 * h:
            continue
        if (x == 0 and x + w == W) or (y == 0 and y + h == H):   # separator lines
            continue
        out.append(_make_comp(strip, x, y, w, h, lab[y:y + h, x:x + w] == i, ox, oy, bg))
    return out


def _make_comp(strip, x, y, w, h, blob, ox, oy, bg):
    """A candidate glyph: its position in the strip, its shape, and its original pixels
    (kept so the shape can be re-rendered at other sizes)."""
    m = max(2, h // 6)
    crop = strip[max(0, y - m):y + h + m, max(0, x - m):x + w + m].copy()
    return dict(x=x, y=y, w=w, h=h, ox=ox, oy=oy, label=None, crop=crop, bg=bg, blob=blob,
                glyph=_glyph(strip[y:y + h, x:x + w], blob))


def _split_wide(rd, img):
    """Split digits that touch (a small "11" or "10" often comes out as one blob).
    A blob much wider than this board's typical digit, and wider than it is tall, is
    two digits: cut it at its thinnest column near the middle."""
    ratios = [c["w"] / c["h"] for s in rd.col_comps + rd.row_comps for c in s]
    if not ratios:
        return
    limit = max(1.0, 1.5 * float(np.median(ratios)))

    def split(comps):
        out = []
        for c in comps:
            if c["w"] <= limit * c["h"]:
                out.append(c)
                continue
            strip = img[c["oy"]:, c["ox"]:]                    # strip coordinates, as before
            region = strip[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
            masks = _split_by_fill(region, c["blob"])
            if masks is None:                                  # fall back to the thinnest column
                masks = []
                for px, pw in _split_touching(c["blob"], limit):
                    m = np.zeros_like(c["blob"])
                    m[:, px:px + pw] = c["blob"][:, px:px + pw]
                    masks.append(m)
            if len(masks) == 1:
                out.append(c)
                continue
            for m in masks:
                rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
                if len(rows) == 0:
                    continue
                py, px = int(rows[0]), int(cols[0])
                ph, pw = int(rows[-1] - py + 1), int(cols[-1] - px + 1)
                out.append(_make_comp(strip, c["x"] + px, c["y"] + py, pw, ph,
                                      m[py:py + ph, px:px + pw], c["ox"], c["oy"], c["bg"]))
        return out
    rd.col_comps = [split(s) for s in rd.col_comps]
    rd.row_comps = [split(s) for s in rd.row_comps]


def _split_by_fill(region, blob):
    """Split touching outlined digits along their real boundary. Their outlines merge
    where they touch, but each digit keeps its own light fill: find those fills, then give
    every pixel of the blob to the nearest one. Returns part masks, left to right, or
    None if the fills don't show separate side-by-side digits (e.g. plain dark text)."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    vals = gray[blob].reshape(-1, 1)
    if len(vals) < 20 or int(vals.max()) - int(vals.min()) <= 30:
        return None
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # fills are *inside* a digit; light pixels on the blob's outer edge are just
    # anti-aliasing (plain dark text has nothing but those)
    inside = cv2.erode(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    light = (inside & (gray > t)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(light, connectivity=4)
    if n < 3:
        return None
    H = blob.shape[0]
    # A digit's fill runs most of its height. Shorter pieces (like the flag of a small
    # "1", cut off from its stem) belong to a neighbouring fill, not a digit of their own.
    fills = sorted((i for i in range(1, n) if st[i, 3] >= 0.5 * H), key=lambda i: cen[i][0])
    if not 2 <= len(fills) <= 4:
        return None
    for a, b in zip(fills, fills[1:]):                  # must sit side by side
        if st[a, 0] + st[a, 2] > st[b, 0] + 0.3 * min(st[a, 2], st[b, 2]):
            return None
    dist = np.stack([cv2.distanceTransform((lab != i).astype(np.uint8), cv2.DIST_L2, 3) for i in fills])
    owner = np.argmin(dist, axis=0)
    parts = [blob & (owner == k) for k in range(len(fills))]
    if any(p.any(axis=0).sum() < 0.25 * H for p in parts):   # a sliver isn't a digit
        return None
    return parts


def _split_touching(blob, limit):
    """[(x offset, width), ...] of the parts of a too-wide blob, cut at thin columns."""
    h, w = blob.shape
    if w <= limit * h:
        return [(0, w)]
    ink = blob.sum(axis=0)
    lo, hi = int(0.25 * w), int(0.75 * w)
    if hi <= lo:
        return [(0, w)]
    cut = lo + int(np.argmin(ink[lo:hi]))
    if cut <= 0:
        return [(0, w)]
    return (_split_touching(blob[:, :cut], limit)
            + [(cut + px, pw) for px, pw in _split_touching(blob[:, cut:], limit)])


def _rescaled(comp):
    """Data augmentation: re-render a labelled glyph at other sizes, the way the game
    would draw it on bigger or smaller boards. Yields (glyph, height)."""
    crop, bg, h0 = comp["crop"], comp["bg"], comp["h"]
    for target in RESCALE_HEIGHTS:
        s = target / h0
        if abs(np.log(s)) < 0.05 or s > 2.0:           # same size, or unrealistic blow-up
            continue
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img = cv2.resize(crop, None, fx=s, fy=s, interpolation=interp)
        mask, _ = ink_mask(img, bg)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))              # the digit is the biggest blob
        x, y, w, h = (int(v) for v in st[i, :4])
        if h >= 5:
            yield _glyph(img[y:y + h, x:x + w], lab[y:y + h, x:x + w] == i), h


# =============================================================== digit recognition

def _hog(g, cell=8, bins=9):
    """HOG (histogram of oriented gradients) for a stack of glyphs (N, G, G), in NumPy.
    1. smooth slightly, 2. brightness gradient at each pixel (direction + strength),
    3. per 8x8 cell, a histogram of gradient directions (0-180 degrees, 9 bins),
    4. normalise each 2x2 block of cells, so it describes shape, not stroke darkness."""
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")              # 1-2-1 smoothing
    p = (p[:, :, :-2] + 2 * p[:, :, 1:-1] + p[:, :, 2:]) / 4
    g = (p[:, :-2, :] + 2 * p[:, 1:-1, :] + p[:, 2:, :]) / 4
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    mag = np.hypot(gx, gy)
    pos = (np.degrees(np.arctan2(gy, gx)) % 180) / (180 / bins) - 0.5   # split between 2 bins
    lo = np.floor(pos)
    frac = pos - lo
    b0, b1 = lo.astype(int) % bins, (lo.astype(int) + 1) % bins
    N, G = g.shape[0], g.shape[1]
    C = G // cell
    hist = np.zeros((N, C, C, bins), np.float32)
    for b in range(bins):
        vote = mag * ((b0 == b) * (1 - frac) + (b1 == b) * frac)
        hist[..., b] = vote.reshape(N, C, cell, C, cell).sum(axis=(2, 4))
    blocks = []
    for by in range(C - 1):
        for bx in range(C - 1):
            v = hist[:, by:by + 2, bx:bx + 2, :].reshape(N, -1)
            v = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True) + 1e-6)
            v = np.minimum(v, 0.2)                                    # "L2-Hys" normalisation
            blocks.append(v / np.sqrt((v ** 2).sum(axis=1, keepdims=True) + 1e-6))
    return np.hstack(blocks)


class DigitClassifier:
    """A trained digit classifier: HOG features + multinomial logistic regression,
    learned from every labelled digit in the DigitBook.

    Template matching recognises a digit only if it's close to one it has stored.
    This instead learns what separates the digits from each other, so it can say
    "that's a 5" for a size it never saw exactly. Its confidence isn't trustworthy on
    very small digits, so its answers are only used provisionally: the puzzle has to
    confirm them (see analyse)."""
    def __init__(self):
        self.classes = None

    @staticmethod
    def features(glyphs):
        """Features for one glyph or a stack of them: HOG (which way the outline's edges
        point in each 8x8 patch, so it describes shape without caring about exact pixel
        positions) plus a coarse 8x8 picture of the glyph."""
        g = np.asarray(glyphs, np.float32)
        single = g.ndim == 2
        if single:
            g = g[None]
        f = np.hstack([_hog(g), g.reshape(len(g), 8, GLYPH // 8, 8, GLYPH // 8).mean(axis=(2, 4)).reshape(len(g), -1)])
        return f[0] if single else f

    @staticmethod
    def _augment(glyph):
        """Slightly thicker, thinner and shifted copies, so the classifier doesn't
        depend on exact stroke width or position."""
        k = np.ones((2, 2), np.uint8)
        out = [glyph, cv2.dilate(glyph, k), cv2.erode(glyph, k)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            out.append(np.roll(np.roll(glyph, dy, axis=0), dx, axis=1))
        return [g for g in out if g.any()]

    def train(self, labels, glyphs, iters=300, max_samples=9000):
        classes = sorted(set(labels))
        if len(classes) < 2:
            self.classes = None
            return
        samples, y = [], []
        for lab, g in zip(labels, glyphs):
            for a in self._augment(g):
                samples.append(a)
                y.append(classes.index(lab))
        y = np.array(y)
        if len(samples) > max_samples:                # keep training quick for big memories
            keep = np.random.default_rng(0).choice(len(samples), max_samples, replace=False)
            samples, y = [samples[i] for i in keep], y[keep]
        X = self.features(np.stack(samples)).astype(np.float32)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1), np.float32)])
        K = len(classes)
        Y = np.eye(K, dtype=np.float32)[y]
        counts = np.bincount(y, minlength=K)          # rare digits count as much as common ones
        w = (len(y) / (K * counts[y])).astype(np.float32)[:, None]
        W, V = np.zeros((Z.shape[1], K), np.float32), np.zeros((Z.shape[1], K), np.float32)
        for _ in range(iters):                        # gradient descent with momentum
            L = Z @ W
            L -= L.max(axis=1, keepdims=True)
            P = np.exp(L)
            P /= P.sum(axis=1, keepdims=True)         # softmax: a probability per digit
            G = Z.T @ ((P - Y) * w) / len(Z) + 1e-3 * W
            V = 0.9 * V - 0.05 * G
            W += V
        self.W, self.classes = W, classes

    def predict(self, glyph):
        """(label, probability) or (None, 0) if untrained."""
        if self.classes is None or not glyph.any():
            return None, 0.0
        z = np.append((self.features(glyph) - self.mu) / self.sd, 1.0).astype(np.float32)
        L = z @ self.W
        p = np.exp(L - L.max())
        p /= p.sum()
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])


class DigitBook:
    """Learned glyph templates: label is a digit string ('7', '12') or '' for not-a-number.
    Each template remembers the pixel height it was seen (or synthesised) at."""
    VERSION = 3          # bump when the glyph representation changes; older files are ignored
                         # (3: touching digits are split along their fills; older memories
                         #  may hold cut-off shapes)

    def __init__(self, path=None):
        self.labels, self.glyphs, self.heights, self._d2 = [], [], [], []
        self._cache = None
        self.outdated = False
        self.revision = 0                     # bumped on every change; retrains the classifier
        self._clf, self._clf_revision = DigitClassifier(), -1
        if path and os.path.exists(path):
            data = np.load(path)
            if "version" not in data.files or int(data["version"]) != self.VERSION:
                self.outdated = True         # made by an older version of this script
                return
            for lab, g, h in zip(data["labels"], data["glyphs"], data["heights"]):
                self.add(str(lab), g.astype(np.uint8), int(h))

    def add(self, label, glyph, height=-1):
        if glyph.any():
            self.labels.append(label)
            self.glyphs.append(glyph)
            self.heights.append(height)
            self._d2.append(_dt2(glyph))
            self._cache = None
            self.revision += 1

    def classify(self, glyph):
        """(label, probability) from the trained classifier, retraining it first if
        anything was learned since last time."""
        if self._clf_revision != self.revision:
            self._clf.train(self.labels, self.glyphs)
            self._clf_revision = self.revision
        return self._clf.predict(glyph)

    def learn(self, label, comp):
        """Store a labelled glyph plus copies re-rendered at other sizes."""
        self.add(label, comp["glyph"], comp["h"])
        if "crop" in comp:
            for g, h in _rescaled(comp):
                lab, d, _ = self.nearest(g, h)
                if not (lab == label and d < 0.15):     # skip near-duplicates
                    self.add(label, g, h)

    def relearn(self, label, comp, tol=0.15):
        """A correction from you: delete templates that look like this glyph but carry a
        different label (they caused the misread), then learn it properly."""
        idx, d = self._dists(comp["glyph"], comp["h"])
        bad = {int(i) for i, dist in zip(idx, d) if dist < tol and self.labels[i] != label}
        if bad:
            keep = [i for i in range(len(self.labels)) if i not in bad]
            for name in ("labels", "glyphs", "heights", "_d2"):
                lst = getattr(self, name)
                setattr(self, name, [lst[i] for i in keep])
            self._cache = None
            self.revision += 1
        self.learn(label, comp)

    def _dists(self, glyph, height):
        """Indices of templates of a similar size, and their distances to `glyph`."""
        if not self.glyphs or not glyph.any():
            return np.array([], int), np.array([])
        if self._cache is None:
            M = np.stack(self.glyphs).astype(np.float32)
            self._cache = (M, np.stack(self._d2), np.maximum(M.sum(axis=(1, 2)), 1),
                           np.array(self.labels), np.array(self.heights, dtype=float))
        M, D2, n, labels, hs = self._cache
        idx = np.flatnonzero((hs < 0) | ((hs <= height * HEIGHT_TOL) & (hs >= height / HEIGHT_TOL)))
        if len(idx) == 0:
            return idx, np.array([])
        q = glyph.astype(np.float32)
        d = 0.5 * (np.sqrt((D2[idx] * q).sum(axis=(1, 2)) / q.sum())
                   + np.sqrt((M[idx] * _dt2(glyph)).sum(axis=(1, 2)) / n[idx]))
        return idx, d

    def dist_to(self, label, glyph, height):
        """Distance from `glyph` to the closest stored example of `label` (similar size)."""
        idx, d = self._dists(glyph, height)
        if len(idx) == 0:
            return np.inf
        sel = d[self._cache[3][idx] == label]
        return float(sel.min()) if len(sel) else np.inf

    def nearest(self, glyph, height):
        """(best label, its distance, distance to the best *other* label)."""
        idx, d = self._dists(glyph, height)
        if len(idx) == 0:
            return None, np.inf, np.inf
        labels = self._cache[3]
        i = int(np.argmin(d))
        best = labels[idx[i]]
        others = d[labels[idx] != best]
        return str(best), float(d[i]), float(others.min()) if len(others) else np.inf

    def match(self, glyph, height):
        """Label for `glyph`, or None unless one template is close and clearly closest."""
        lab, d, runner_up = self.nearest(glyph, height)
        return lab if d <= MATCH and d <= MARGIN * runner_up else None

    def save(self, path):
        np.savez_compressed(path, labels=np.array(self.labels, dtype=str),
                            glyphs=np.stack(self.glyphs).astype(np.uint8),
                            heights=np.array(self.heights, dtype=np.int32),
                            version=np.array(self.VERSION))


# =============================================================== reading the clues

def _group_vertical(comps):
    """Column clues: numbers are stacked; digits of one number sit side by side."""
    groups = []
    for c in sorted(comps, key=lambda c: c["y"] + c["h"] / 2):
        if groups:
            g = groups[-1]
            top, bot = min(e["y"] for e in g), max(e["y"] + e["h"] for e in g)
            ov = min(bot, c["y"] + c["h"]) - max(top, c["y"])
            if ov > 0.5 * min(bot - top, c["h"]):
                g.append(c)
                continue
        groups.append([c])
    return [sorted(g, key=lambda c: c["x"]) for g in groups]


def _group_horizontal(comps, gap):
    """Row clues: numbers run left to right; digits of one number are close together."""
    groups = []
    for c in sorted(comps, key=lambda c: c["x"]):
        if groups:
            p = groups[-1][-1]
            space = c["x"] - (p["x"] + p["w"])
            ov = min(p["y"] + p["h"], c["y"] + c["h"]) - max(p["y"], c["y"])
            if space <= gap * max(p["h"], c["h"]) and ov > 0.5 * min(p["h"], c["h"]):
                groups[-1].append(c)
                continue
        groups.append([c])
    return groups


def _numbers(groups):
    out = []
    for g in groups:
        s = "".join(c["label"] for c in g)
        if s:
            out.append(int(s))
    return [] if out == [0] else out


def _plausible(rows, cols):
    for lines, size in ((rows, len(cols)), (cols, len(rows))):
        for cl in lines:
            if 0 in cl or (cl and sum(cl) + len(cl) - 1 > size):
                return False
    total = sum(map(sum, rows))
    return total > 0 and total == sum(map(sum, cols))


class Reading:
    def __init__(self, geom):
        self.geom = geom
        self.col_comps, self.row_comps = [], []
        self.unknown = []
        self.asked = self.auto = 0         # shapes labelled by you / worked out automatically
        self.corrected = 0                 # misread shapes fixed using the puzzle's logic
        self.confirmed = 0                 # near-miss matches the puzzle confirmed
        self.fingerprint = None
        self.tones, self.filled = [], None  # empty-cell colours; which cells are already filled
        self.rows = self.cols = self.solution = None
        self.error = None


def _cluster(comps, tol=MATCH):
    """Group same-looking glyphs so each distinct shape is handled once. Biggest groups first."""
    clusters = []
    for comp in comps:
        for cl in clusters:
            h0 = cl[0]["h"]
            if (h0 / HEIGHT_TOL <= comp["h"] <= h0 * HEIGHT_TOL
                    and glyph_dist(cl[0]["glyph"], comp["glyph"]) <= tol):
                cl.append(comp)
                break
        else:
            clusters.append([comp])
    return sorted(clusters, key=len, reverse=True)


LABEL_OPTIONS = [str(d) for d in range(10)] + [""]


def _infer(rd, clusters, max_shapes=3):
    """Work out unknown shapes from the puzzle itself: try every digit for each unknown
    shape and keep the one labelling where the clues add up and the puzzle solves.
    Returns that labelling, or None if there isn't exactly one."""
    if not clusters or len(clusters) > max_shapes:
        return None
    col_groups = [_group_vertical(s) for s in rd.col_comps]
    row_groups = [_group_horizontal(s, GAP) for s in rd.row_comps]
    found = []
    for combo in itertools.product(LABEL_OPTIONS, repeat=len(clusters)):
        for cl, lab in zip(clusters, combo):
            for comp in cl:
                comp["label"] = lab
        rows = [_numbers(g) for g in row_groups]
        cols = [_numbers(g) for g in col_groups]
        if _plausible(rows, cols) and solve(rows, cols, max_guesses=2000) is not None:
            found.append(combo)
            if len(found) > 1:
                break
    for cl in clusters:
        for comp in cl:
            comp["label"] = None
    return found[0] if len(found) == 1 else None


def _resolve_unknown(rd, book, ask):
    """Label unknown shapes: deduce them when few are left, otherwise ask (most common first).
    Shapes are grouped tightly so one answer never covers two different digits."""
    clusters = _cluster(rd.unknown, tol=0.15)

    def label(cl, lab):
        for comp in cl:
            comp["label"] = lab
            comp["dist"] = glyph_dist(cl[0]["glyph"], comp["glyph"])   # how sure, for _self_correct
        book.learn(lab, cl[0])

    while clusters:
        combo = _infer(rd, clusters)
        if combo is not None:
            for cl, lab in zip(clusters, combo):
                label(cl, lab)
            rd.auto += len(clusters)
            break
        if ask is None:
            break
        cl = clusters.pop(0)
        ans = ask(cl[0], len(cl))
        if ans is None:                     # skipped: can't finish this puzzle
            break
        label(cl, ans)
        rd.asked += 1
        # your answer may cover other shapes too (same digit, slightly different render)
        still = []
        for other in clusters:
            labels = {book.match(c["glyph"], c["h"]) for c in other}
            if len(labels) == 1 and None not in labels:
                lab = labels.pop()
                for comp in other:
                    comp["label"], comp["dist"] = lab, 0.0
            else:
                still.append(other)
        clusters = still
    rd.unknown = [c for c in rd.unknown if c["label"] is None]


def analyse(img, box, clue, book, size=None, ask=None):
    """img: BGR screenshot; box: grid (x0,y0,x1,y1); clue: clue-area top-left (x,y).
    ask(comp, count) -> label or None is called for shapes that can't be worked out."""
    if size:
        R, C = size
    else:
        R, C = detect_size(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box)
    g = Geometry(box, clue, R, C)
    rd = Reading(g)
    for c in range(C):
        xa, ya, xb, yb = g.col_strip(c)
        rd.col_comps.append(components(img[ya:yb, xa:xb], g.cell, xa, ya))
    for r in range(R):
        xa, ya, xb, yb = g.row_strip(r)
        rd.row_comps.append(components(img[ya:yb, xa:xb], g.cell, xa, ya))
    _split_wide(rd, img)
    for i, strip in enumerate(rd.col_comps):        # for the questions: where each shape is
        for c in strip:
            c["where"] = f"column {i + 1} (counting from the left)"
    for i, strip in enumerate(rd.row_comps):
        for c in strip:
            c["where"] = f"row {i + 1} (counting from the top)"
    # empty cells share the clue cells' colours (both the plain and the lighter tone)
    rd.tones = [np.median(img[ya:yb, xa:xb].reshape(-1, 3), axis=0)
                for xa, ya, xb, yb in [g.col_strip(c) for c in range(C)] + [g.row_strip(r) for r in range(R)]
                if xb > xa and yb > ya]
    rd.filled = cell_states(img, g, rd.tones)
    # identifies "this same board again" (size + where every glyph sits)
    rd.fingerprint = (R, C, tuple(tuple((c["x"] // 4, c["y"] // 4) for c in s)
                                  for s in rd.col_comps + rd.row_comps))

    # Sure template matches (a digit seen before at this size) are taken as read.
    # Anything else is read by the trained classifier, or failing that by the closest
    # template, but only *provisionally*: if the clues then add up and the puzzle solves,
    # the puzzle has confirmed them, and they're learned at this size.
    # "dist" records how doubtful each read is, so the least certain get rechecked first.
    tentative = []
    for comp in (c for strip in rd.col_comps + rd.row_comps for c in strip):
        lab, d, runner_up = book.nearest(comp["glyph"], comp["h"])
        if lab is not None and d <= MATCH and d <= MARGIN * runner_up:
            comp["label"], comp["dist"] = lab, d
            continue
        clab, p = book.classify(comp["glyph"])
        # A classifier always picks *some* digit it knows, even for one it has never
        # seen. So its answer only counts if the glyph also looks reasonably like that
        # digit's stored examples; a never-seen digit stays unknown and gets worked out
        # from the puzzle or asked about.
        familiar = book.dist_to(clab, comp["glyph"], comp["h"]) if clab is not None else np.inf
        if p >= CLF_MIN and familiar <= FAMILIAR:
            comp["label"], comp["dist"] = clab, familiar
            tentative.append(comp)
        else:
            comp["label"], comp["dist"] = None, 0.0
            rd.unknown.append(comp)
    if rd.unknown:
        _resolve_unknown(rd, book, ask)
        if rd.unknown:
            return rd

    if _finish(rd) or (_self_correct(rd, book) and _finish(rd)):
        _confirm(tentative, book, rd)
        return rd
    if tentative:
        # Some provisional reads are wrong. Stop trusting them a bit at a time, most
        # doubtful first (a third, then two thirds, then all), working the distrusted ones
        # out from the puzzle or asking about them, until the puzzle checks out.
        shapes = _cluster(tentative, tol=0.15)
        shapes.sort(key=lambda cl: -max(c["dist"] for c in cl))
        done = 0
        for frac in (1 / 3, 2 / 3, 1.0):
            k = max(1, int(np.ceil(len(shapes) * frac)))
            if k <= done:
                continue
            rd.unknown = [c for cl in shapes[done:k] for c in cl]
            for comp in rd.unknown:
                comp["label"] = None
            done = k
            _resolve_unknown(rd, book, ask)
            if rd.unknown:
                return rd
            if _finish(rd) or (_self_correct(rd, book) and _finish(rd)):
                _confirm([c for cl in shapes[done:] for c in cl], book, rd)
                return rd
    rd.error = ("clues don't add up or have no solution - open picross_last.png "
                "to see what was read")
    return rd


def _confirm(tentative, book, rd):
    """The puzzle checked out, so the provisional matches were right: learn them at this
    size so they match exactly next time."""
    shapes = _cluster([c for c in tentative if c["label"] is not None], tol=0.15)
    for cl in shapes:
        book.learn(cl[0]["label"], cl[0])
    rd.confirmed += len(shapes)


def cell_states(img, g, tones):
    """Which grid cells are already filled (or crossed out): a cell counts as empty when
    its centre is the colour of an empty cell. Returns a rows x cols 0/1 array, or None
    if no cell looks empty at all (colours not recognised - then assume a blank board)."""
    tones = np.array(tones, dtype=float)
    out = np.zeros((g.rows, g.cols), dtype=int)
    k = max(1.0, 0.2 * g.cell)
    for r in range(g.rows):
        for c in range(g.cols):
            x, y = g.center(r, c)
            patch = img[int(y - k):int(y + k) + 1, int(x - k):int(x + k) + 1].reshape(-1, 3)
            colour = np.median(patch, axis=0)
            out[r, c] = np.abs(tones - colour).sum(axis=1).min() > 45
    return None if out.all() else out


FIXES = {}   # clue lines you corrected but the bot couldn't learn, keyed by board fingerprint


def _finish(rd):
    """Turn labelled glyphs into clues and solve. True if that worked."""
    fixes = FIXES.get(rd.fingerprint, {})
    rd.cols = [_numbers(_group_vertical(s)) for s in rd.col_comps]
    for (kind, i), nums in fixes.items():
        if kind == "c":
            rd.cols[i] = nums
    rd.rows = None
    for gap in (GAP, 0.25, 0.45, 0.6, 0.8, 0.15):
        rows = [_numbers(_group_horizontal(s, gap)) for s in rd.row_comps]
        for (kind, i), nums in fixes.items():
            if kind == "r":
                rows[i] = nums
        if rd.rows is None:
            rd.rows = rows
        if _plausible(rows, rd.cols):
            sol = solve(rows, rd.cols)
            if sol is not None:
                rd.rows, rd.solution = rows, sol
                return True
    return False


def _self_correct(rd, book, suspects=6):
    """The clues don't add up, so some shape was misread. Re-examine the least certain
    shapes - each one alone, then in pairs - and keep a relabelling that makes the
    puzzle consistent and solvable. The correction is learned.
    Shapes are grouped tightly here (near-identical renders only), so a misread glyph
    isn't lumped in with the correctly read ones it resembles."""
    shapes = _cluster([c for s in rd.col_comps + rd.row_comps for c in s], tol=0.12)
    # Prime suspects: identical-looking glyphs that were given different labels;
    # after those, the shakiest matches.
    shapes.sort(key=lambda cl: (len({c["label"] for c in cl}) == 1,
                                -max(c.get("dist", 0.0) for c in cl)))
    shapes = shapes[:suspects]
    before = {id(c): c["label"] for cl in shapes for c in cl}
    for k in (1, 2):
        for group in itertools.combinations(shapes, k):
            combo = _infer(rd, list(group))
            if combo is not None:
                for cl, lab in zip(group, combo):
                    for comp in cl:
                        comp["label"] = lab
                    book.learn(lab, cl[0])
                for cl in shapes:
                    for comp in cl:
                        if comp["label"] is None:
                            comp["label"] = before[id(comp)]
                rd.corrected += k
                return True
            for cl in group:                       # _infer cleared these; put them back
                for comp in cl:
                    comp["label"] = before[id(comp)]
    return False


def save_debug(img, rd, path=DEBUG_PATH, crop=None):
    """Annotated copy of what was read; `crop` (x0,y0,x1,y1) trims it to the board."""
    dbg = img.copy()
    g = rd.geom
    red, green, blue = (0, 0, 255), (0, 170, 0), (200, 80, 0)
    font_scale = max(0.7, g.cell / 40)
    for c in range(g.cols + 1):
        x = int(round(g.x0 + c * g.cw))
        cv2.line(dbg, (x, int(g.y0)), (x, int(g.y1)), red, 1)
    for r in range(g.rows + 1):
        y = int(round(g.y0 + r * g.ch))
        cv2.line(dbg, (int(g.x0), y), (int(g.x1), y), red, 1)
    for comp in (c for s in rd.col_comps + rd.row_comps for c in s):
        x, y = comp["x"] + comp["ox"], comp["y"] + comp["oy"]
        known = comp["label"] is not None
        cv2.rectangle(dbg, (x, y), (x + comp["w"], y + comp["h"]), green if known else red, 1)
        if known and comp["label"]:
            cv2.putText(dbg, comp["label"], (x, max(8, y - 1)), cv2.FONT_HERSHEY_PLAIN,
                        font_scale, blue, 1, cv2.LINE_AA)
    if rd.solution:
        for r in range(g.rows):
            for c in range(g.cols):
                if rd.solution[r][c]:
                    x, y = g.center(r, c)
                    cv2.circle(dbg, (int(x), int(y)), max(2, int(g.cell * 0.15)), blue, -1)
    if crop:
        x0, y0, x1, y1 = crop
        dbg = dbg[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20]
    cv2.imwrite(path, dbg)


def _ascii(glyph):
    small = cv2.resize(glyph.astype(np.float32), (GLYPH, GLYPH // 2), interpolation=cv2.INTER_AREA)
    return "\n".join("      " + "".join("#" if v > 0.4 else "." for v in row) for row in small)


def _make_terminal_ask():
    first = [True]

    def ask(comp, count):
        if first[0]:
            print("\a\n  New number shape(s). Type the digit(s) drawn - just this shape, so for")
            print("  the 1 of a 13, type 1. (Two digits drawn as one shape: type both, e.g. 11.")
            print("  x = not a number, Enter = skip.)\n")
            first[0] = False
        print(_ascii(comp["glyph"]))
        where = f" in {comp['where']}" if comp.get("where") else ""
        ans = input(f"    seen {count}x, e.g.{where}  ->  ").strip().lower()
        print()
        if ans == "x":
            return ""
        return ans if ans.isdigit() else None
    return ask


# =============================================================== clue editor

def _clue_problem(rows, cols):
    """Why these clues can't be right, or None if they're plausible."""
    for lines, size, name in ((cols, len(rows), "column"), (rows, len(cols), "row")):
        for i, cl in enumerate(lines):
            if cl and sum(cl) + len(cl) - 1 > size:
                return f"{name} {i + 1} is too long for the board"
    tr, tc = sum(map(sum, rows)), sum(map(sum, cols))
    if tr != tc:
        return f"rows add up to {tr} but columns to {tc}"
    return None


def _fmt_lines(prefix, lines, width=96):
    items = [f"{prefix}{i + 1}: {' '.join(map(str, l)) or '0'}" for i, l in enumerate(lines)]
    w = max(map(len, items)) + 3
    per = max(1, width // w)
    return "\n".join("    " + "".join(it.ljust(w) for it in items[i:i + per])
                     for i in range(0, len(items), per))


def edit_clues(rd, read=input):
    """Show the clues as read and let you correct lines. Returns the set of lines you
    changed (plus the solution in rd), or None if you cancelled."""
    rows, cols = [list(r) for r in rd.rows], [list(c) for c in rd.cols]
    changed = set()
    print(f"\n  {rd.geom.rows}x{rd.geom.cols} board. Clues as I read them:")
    print("  Columns, left to right (each read top to bottom):")
    print(_fmt_lines("c", cols))
    print("  Rows, top to bottom (each read left to right):")
    print(_fmt_lines("r", rows))
    print("  Fix a line by typing it, e.g.  c4 2 1  or  r10 0  (0 = empty line).")
    print("  Enter = done, q = cancel.")
    while True:
        problem = _clue_problem(rows, cols)
        s = read(f"  [{problem or 'clues add up'}] > ").strip().lower()
        if s == "q":
            return None
        if s == "":
            if problem:
                print("  Those still can't be right - fix another line, or q to cancel.")
                continue
            sol = solve(rows, cols)
            if sol is None:
                print("  Those clues have no solution - check them again.")
                continue
            rd.rows, rd.cols, rd.solution, rd.error = rows, cols, sol, None
            return changed
        m = re.fullmatch(r"([rc])\s*(\d+)\s*[:=]?\s*([\d\s]*)", s)
        if not m:
            print("  Type it like  c4 2 1  or  r10 3")
            continue
        kind, i = m.group(1), int(m.group(2)) - 1
        target = cols if kind == "c" else rows
        if not 0 <= i < len(target):
            print(f"  There's no {'column' if kind == 'c' else 'row'} {i + 1}.")
            continue
        target[i] = [int(x) for x in m.group(3).split() if int(x)]
        changed.add((kind, i))


def learn_from_fixes(rd, book, changed):
    """Match your corrected numbers to the glyphs on screen and relearn them.
    Lines that can't be matched up (a digit wasn't seen at all, say) are remembered
    as-is for this board. Returns how many lines were learned."""
    learned, unmatched = 0, {}
    for kind, i in changed:
        nums = (rd.cols if kind == "c" else rd.rows)[i]
        comps = [c for c in (rd.col_comps if kind == "c" else rd.row_comps)[i] if c["label"] != ""]
        groups = _group_vertical(comps) if kind == "c" else _group_horizontal(comps, GAP)
        pairs = []
        if len(groups) == len(nums):
            for grp, num in zip(groups, nums):
                text = str(num)
                if len(grp) == len(text):
                    pairs += list(zip(grp, text))
                elif len(grp) == 1:                    # digits drawn touching: one glyph
                    pairs.append((grp[0], text))
                else:
                    pairs = None
                    break
        if pairs:
            for comp, label in pairs:
                if comp["label"] != label:
                    book.relearn(label, comp)
                comp["label"] = label
            learned += 1
        else:
            unmatched[(kind, i)] = nums
    if unmatched:
        FIXES.setdefault(rd.fingerprint, {}).update(unmatched)
    return learned


# =============================================================== mouse input

def _runs(cells):
    out, start = [], None
    for i, v in enumerate(list(cells) + [0]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def fill(need, g, offset, stop, delay=0.02, click_only=False):
    """Fill every cell marked in `need` (rows x cols 0/1), dragging along runs."""
    import pyautogui
    R, C = g.rows, g.cols
    row_runs = [[(r, c) for c in range(a, b + 1)] for r in range(R) for a, b in _runs(need[r])]
    col_runs = [[(r, c) for r in range(a, b + 1)] for c in range(C)
                for a, b in _runs([need[r][c] for r in range(R)])]
    paths = row_runs if len(row_runs) <= len(col_runs) else col_runs
    if click_only:
        paths = [[cell] for p in paths for cell in p]

    def pt(r, c):
        x, y = g.center(r, c)
        return offset[0] + x, offset[1] + y

    for path in paths:
        if stop.is_set():
            return False
        pyautogui.moveTo(*pt(*path[0]))
        pyautogui.mouseDown()
        try:
            time.sleep(delay)
            for cell in path[1:]:
                pyautogui.moveTo(*pt(*cell))
                time.sleep(delay)
        finally:
            pyautogui.mouseUp()
        time.sleep(delay)
    return True


def click(x, y):
    import pyautogui
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown()
    time.sleep(0.05)
    pyautogui.mouseUp()


# =============================================================== taught buttons

class Buttons:
    """Pictures of buttons you taught (F6). In auto mode, whichever one is on screen
    gets clicked, which is how the bot moves on to the next level."""
    def __init__(self, folder=BUTTONS_DIR):
        self.folder = folder
        self.items = []
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.endswith(".png"):
                    img = cv2.imread(os.path.join(folder, name))
                    if img is not None:
                        self.items.append((name, img))

    def teach(self, screen, x, y, half=32):
        """Remember the patch of `screen` around image point (x, y)."""
        H, W = screen.shape[:2]
        x0, y0 = max(0, x - half), max(0, y - half)
        patch = screen[y0:min(H, y + half), x0:min(W, x + half)].copy()
        os.makedirs(self.folder, exist_ok=True)
        name = f"button_{len(self.items) + 1:02d}_{x - x0}_{y - y0}.png"   # click offset in name
        cv2.imwrite(os.path.join(self.folder, name), patch)
        self.items.append((name, patch))
        return name

    def find(self, screen, threshold=0.9):
        """(name, x, y) of the best-matching taught button on screen, or None."""
        best = None
        for name, patch in self.items:
            if patch.shape[0] > screen.shape[0] or patch.shape[1] > screen.shape[1]:
                continue
            res = cv2.matchTemplate(screen, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, (px, py) = cv2.minMaxLoc(res)
            if score >= threshold and (best is None or score > best[0]):
                ox, oy = (int(v) for v in name[:-4].split("_")[-2:])
                best = (score, name, px + ox, py + oy)
        return best[1:] if best else None


# =============================================================== app

def _dpi_aware():
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class Screen:
    """Screenshots of the monitor the mouse is on."""
    def __init__(self):
        import mss
        import pyautogui
        MSS = getattr(mss, "MSS", None) or mss.mss         # newer mss renamed the class
        self.sct = MSS()
        mx, my = pyautogui.position()
        mons = self.sct.monitors[1:] or self.sct.monitors
        self.mon = next((m for m in mons if m["left"] <= mx < m["left"] + m["width"]
                         and m["top"] <= my < m["top"] + m["height"]), mons[0])
        self.offset = (self.mon["left"], self.mon["top"])

    def grab(self):
        return np.ascontiguousarray(np.array(self.sct.grab(self.mon))[:, :, :3])


def solve_on_screen(book, args, stop, mode="solve", say=print, seen=None):
    """Find, read, solve and fill the puzzle on screen.
    mode: 'solve' (F8), 'check' (F7: let you fix the clues first), 'auto' (F10: quiet,
    and asks for help at most once per board - `seen` remembers which).
    Returns 'filled', 'done' (already solved), 'none' (no puzzle found), or 'stuck'."""
    import pyautogui
    scr = Screen()
    img = scr.grab()
    try:
        frame = locate_board(img)[2]
    except ReadError:
        if mode != "auto":
            cv2.imwrite(CAPTURE_PATH, img)
            say(f"  Couldn't find the puzzle on screen. Screenshot saved as {CAPTURE_PATH}")
        return "none"
    # move the mouse off the board so no hover highlight affects the read, then look again
    park = (scr.offset[0] + max(5, frame[0] - 15), scr.offset[1] + max(5, frame[1] - 15))
    pyautogui.moveTo(*park)
    time.sleep(0.15)
    img = scr.grab()
    if mode != "auto":
        cv2.imwrite(CAPTURE_PATH, img)

    t0 = time.perf_counter()
    ask = _make_terminal_ask()
    try:
        box, clue, frame = locate_board(img)
        rd = analyse(img, box, clue, book, args.size, ask=None if mode == "auto" else ask)
        if mode == "auto" and (rd.unknown or rd.error) and rd.filled is not None and not rd.filled.any():
            if rd.fingerprint in seen:          # already asked about this board
                return "stuck"
            seen.add(rd.fingerprint)
            say("\a  This puzzle needs your help - auto mode waits for your answers here.")
            if rd.unknown:
                rd = analyse(img, box, clue, book, args.size, ask=ask)
    except ReadError as e:
        if mode != "auto":
            say(f"  {e}. Screenshot saved as {CAPTURE_PATH}")
        return "none"
    if rd.asked or rd.auto or rd.corrected or rd.confirmed:
        book.save(DIGITS_PATH)
    save_debug(img, rd, crop=frame)
    g = rd.geom
    if rd.auto:
        say(f"  Worked out {rd.auto} new number shape(s) from the puzzle itself.")
    if rd.confirmed:
        say(f"  Learned {rd.confirmed} known digit(s) at a new size (the puzzle confirmed them).")
    if rd.corrected:
        say(f"  Caught and fixed {rd.corrected} misread shape(s).")
    if rd.unknown:
        if rd.asked or mode != "auto":
            say("  Some shapes are still unlabelled - press F8 to try again.")
        return "stuck"

    used_terminal = rd.asked > 0
    if rd.error or mode == "check":
        if mode == "auto" and rd.fingerprint not in seen:
            return "stuck"
        if rd.error:
            say(f"  The clues I read for this {g.rows}x{g.cols} board don't work.")
        changed = edit_clues(rd)
        if changed is None:
            say("  Cancelled.")
            return "stuck"
        if changed:
            learned = learn_from_fixes(rd, book, changed)
            book.save(DIGITS_PATH)
            say(f"  Thanks - learned from {learned} of the {len(changed)} line(s) you fixed.")
        used_terminal = True

    sol = np.array(rd.solution)
    have = rd.filled if rd.filled is not None else np.zeros_like(sol)
    if (have & (1 - sol)).any():
        say("  Some cells are filled or crossed out that shouldn't be - clear them (or reset "
            "the puzzle) and press F8 again.")
        return "stuck"
    need = sol & (1 - have)
    if not need.any():
        return "done"
    say(f"  {g.rows}x{g.cols} read + solved in {time.perf_counter() - t0:.2f}s"
        + (f", {int(have.sum())} cell(s) were already filled" if have.any() else ""))
    if args.dry_run:
        for row in sol:
            say("   " + "".join("#" if v else "." for v in row))
        return "stuck"
    if used_terminal:
        # you were typing in the terminal: bring the game back to the front by clicking
        # the board's frame (which does nothing in the game), then fill
        say("  Filling in 2 seconds...")
        time.sleep(2)
        click(scr.offset[0] + frame[0] + 2, scr.offset[1] + frame[1] + 2)
        time.sleep(0.3)
    done = fill(need, g, scr.offset, stop, args.delay, args.click)
    pyautogui.moveTo(*park)
    say("  Filled." if done else "  Stopped.")
    return "filled" if done else "stuck"


def auto_step(book, buttons, args, stop, note, seen):
    """One look at the screen in auto mode: fill a fresh puzzle, or click a taught button."""
    status = solve_on_screen(book, args, stop, mode="auto", say=note, seen=seen)
    if status == "filled":
        return
    scr = Screen()
    hit = buttons.find(scr.grab())
    if hit:
        name, x, y = hit
        note(f"  Clicking {name}")
        click(scr.offset[0] + x, scr.offset[1] + y)
        time.sleep(0.5)
    elif status == "done" and not buttons.items:
        note("  Puzzle solved. Teach me the button that moves to the next level: "
             "hover it and press F6.", once=True)


def run(args):
    _dpi_aware()
    import pyautogui
    from pynput import keyboard
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True

    book = DigitBook(DIGITS_PATH)
    buttons = Buttons()
    cmds, stop = queue.Queue(), threading.Event()
    keys = {keyboard.Key.f6: "teach", keyboard.Key.f7: "check",
            keyboard.Key.f8: "solve", keyboard.Key.f10: "auto"}

    def on_press(key):
        if key == keyboard.Key.f9:
            stop.set()
            cmds.put("stop")
        elif key in keys:
            cmds.put(keys[key])

    keyboard.Listener(on_press=on_press).start()
    print("Picross bot ready.")
    print("  F8 = solve the puzzle on screen     F7 = check/fix the clues before solving")
    print("  F6 = teach a button (hover it first) F10 = auto mode on/off")
    print("  F9 = stop                            Ctrl+C = quit")
    if book.outdated:
        print("Ignoring picross_digits.npz from an older version - I'll relearn the digits.")
    print(f"Known number templates: {len(book.labels)}, taught buttons: {len(buttons.items)}")
    if book.labels:
        print("Training the digit classifier...", end=" ", flush=True)
        t0 = time.perf_counter()
        book.classify(np.zeros((GLYPH, GLYPH), np.uint8))   # trains it now, not on your first F8
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    auto, last_note, seen = False, [None], set()

    def note(msg, once=False):
        if not (once and msg == last_note[0]):
            print(msg)
        last_note[0] = msg

    try:
        while True:
            try:
                cmd = cmds.get(timeout=args.interval if auto else 0.25)
            except queue.Empty:
                cmd = None
            try:
                if cmd == "stop":
                    if auto:
                        print("  Auto mode off.")
                    auto = False
                elif cmd == "auto":
                    auto = not auto
                    stop.clear()
                    print(f"  Auto mode {'on - watching the screen' if auto else 'off'}.")
                elif cmd == "teach":
                    scr = Screen()
                    x, y = pyautogui.position()
                    pyautogui.moveTo(scr.offset[0] + 5, scr.offset[1] + 5)   # unhover it
                    time.sleep(0.25)
                    name = buttons.teach(scr.grab(), x - scr.offset[0], y - scr.offset[1])
                    pyautogui.moveTo(x, y)
                    print(f"  Saved {name}. In auto mode I'll click it whenever it's on screen.")
                elif cmd in ("solve", "check"):
                    auto = False
                    stop.clear()
                    if solve_on_screen(book, args, stop, mode=cmd) == "done":
                        print("  This puzzle is already solved.")
                elif cmd is None and auto and not stop.is_set():
                    auto_step(book, buttons, args, stop, note, seen)
            except pyautogui.FailSafeException:
                auto = False
                print("  Stopped (mouse hit a screen corner).")
            except Exception as e:
                print(f"  Error: {e}")
            if cmd in ("solve", "check", "teach"):
                while not cmds.empty():     # drop keys pressed while busy
                    cmds.get_nowait()
    except KeyboardInterrupt:
        print("bye")


def main():
    ap = argparse.ArgumentParser(description="Picross Touch auto-solver")
    ap.add_argument("--size", help="skip grid-size detection: ROWSxCOLS, e.g. 15x15")
    ap.add_argument("--delay", type=float, default=0.02,
                    help="seconds between mouse moves (raise it if the game skips cells)")
    ap.add_argument("--interval", type=float, default=1.0, help="auto mode: seconds between looks")
    ap.add_argument("--click", action="store_true", help="click each cell instead of dragging")
    ap.add_argument("--dry-run", action="store_true", help="solve and print, don't touch the mouse")
    args = ap.parse_args()
    if args.size:
        parts = args.size.lower().split("x")
        args.size = (int(parts[0]), int(parts[-1]))
    run(args)


if __name__ == "__main__":
    main()
