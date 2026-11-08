#!/usr/bin/env python3
"""Self-test for picross_bot: no game or screen needed.

    python selftest.py

1. Solver: random puzzles up to 40x40 must solve.
2. Reader: mock boards drawn in Picross Touch's style (teal frame, white digits with
   black outlines) are played in sequence, as you would play real levels. The bot must
   find each board, read its clues and solve it. Its questions are answered from the
   known answer, the way you'd answer them at the terminal. A misread is a failure.
3. Cell states: already-filled and crossed-out cells must be recognised.
"""
import sys
import time

import cv2
import numpy as np

import picross_bot as pb


def render_board(truth, cell, slots=None, filled=None, crossed=None):
    """Draw a mock board in Picross Touch's style.
    Returns (image, grid box, clue-area corner, row clues, column clues).
    If `slots` is a list, (x, y, text) of every drawn number is appended to it.
    `filled` / `crossed`: optional rows x cols masks of cells to draw filled / X'd."""
    R, C = truth.shape
    rows = [pb.line_clues(truth[r]) or [0] for r in range(R)]
    cols = [pb.line_clues(truth[:, c]) or [0] for c in range(C)]
    teal, cellc, light = (122, 118, 0), (233, 231, 204), (244, 243, 230)
    slot_v, slot_h = 0.8 * cell, 0.7 * cell
    clue_h = int(max(map(len, cols)) * slot_v + 0.3 * cell)
    clue_w = int(max(map(len, rows)) * slot_h + 0.4 * cell)
    sep, line, frame, m = 7, 3, 12, 40
    cx, cy = m + frame, m + frame
    gx0, gy0 = cx + clue_w + sep, cy + clue_h + sep
    gx1, gy1 = gx0 + C * cell, gy0 + R * cell
    img = np.full((gy1 + frame + m, gx1 + frame + m, 3), (228, 225, 191), np.uint8)
    cv2.rectangle(img, (m, m), (gx1 + frame - 1, gy1 + frame - 1), (0, 0, 0), -1)
    cv2.rectangle(img, (m + 4, m + 4), (gx1 + frame - 5, gy1 + frame - 5), teal, -1)
    img[cy:cy + clue_h, cx:cx + clue_w] = 255                       # picture preview box
    for c in range(C):
        img[cy:cy + clue_h, gx0 + c * cell:gx0 + (c + 1) * cell] = light if c % 2 else cellc
    for r in range(R):
        img[gy0 + r * cell:gy0 + (r + 1) * cell, cx:cx + clue_w] = light if r % 2 else cellc
    for r in range(R):
        for c in range(C):
            box = (slice(gy0 + r * cell, gy0 + (r + 1) * cell), slice(gx0 + c * cell, gx0 + (c + 1) * cell))
            img[box] = light if (r + c) % 2 else cellc
            if filled is not None and filled[r][c]:
                img[box] = (110, 95, 30)
            if crossed is not None and crossed[r][c]:
                a, b, k = gx0 + c * cell + cell // 4, gy0 + r * cell + cell // 4, cell // 2
                cv2.line(img, (a, b), (a + k, b + k), (60, 60, 200), max(2, cell // 10))
                cv2.line(img, (a + k, b), (a, b + k), (60, 60, 200), max(2, cell // 10))
    for i in range(1, C):
        x, w = gx0 + i * cell, (5 if i % 5 == 0 else line)
        img[cy:gy1, x - w // 2:x - w // 2 + w] = teal
    for i in range(1, R):
        y, w = gy0 + i * cell, (5 if i % 5 == 0 else line)
        img[y - w // 2:y - w // 2 + w, cx:gx1] = teal

    font, t = cv2.FONT_HERSHEY_DUPLEX, max(2, int(cell / 14))
    places = []
    for c, cl in enumerate(cols):
        for k, v in enumerate(reversed(cl)):
            places.append((gx0 + (c + 0.5) * cell, gy0 - sep - (k + 0.5) * slot_v, str(v)))
    for r, cl in enumerate(rows):
        for k, v in enumerate(reversed(cl)):
            places.append((gx0 - sep - 0.2 * cell - (k + 0.5) * slot_h, gy0 + (r + 0.5) * cell, str(v)))
    for x, y, text in places:
        scale = 0.5 * cell / 22
        (tw, th), _ = cv2.getTextSize(text, font, scale, t + 4)
        if tw > 0.85 * slot_h:
            scale *= 0.85 * slot_h / tw
            (tw, th), _ = cv2.getTextSize(text, font, scale, t + 4)
        org = (int(x - tw / 2), int(y + th / 2))
        cv2.putText(img, text, org, font, scale, (0, 0, 0), t + 4, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, (255, 255, 255), t, cv2.LINE_AA)
    if slots is not None:
        slots.extend(places)
    clean = lambda lines: [[] if x == [0] else x for x in lines]
    return img, (gx0, gy0, gx1, gy1), (cx, cy), clean(rows), clean(cols)


def answer_from(slots):
    """Answers the bot's questions the way you would, using where each number was drawn."""
    def ask(comp, count):
        x = comp["ox"] + comp["x"] + comp["w"] / 2
        y = comp["oy"] + comp["y"] + comp["h"] / 2
        sx, _, text = min(slots, key=lambda s: (s[0] - x) ** 2 + (s[1] - y) ** 2)
        if len(text) == 1 or abs(x - sx) < 0.2 * comp["w"]:       # one digit, or digits merged
            return text
        return text[0] if x < sx else text[-1]                    # one digit of a two-digit number
    return ask


def test_solver(rng):
    print("Solver:")
    ok_all = True
    for R, C in [(10, 10), (15, 15), (20, 20), (25, 30), (40, 40)]:
        truth = (rng.random((R, C)) < 0.62).astype(int)
        rows = [pb.line_clues(truth[r]) for r in range(R)]
        cols = [pb.line_clues(truth[:, c]) for c in range(C)]
        t = time.perf_counter()
        ok = pb.solve(rows, cols) is not None
        ok_all &= ok
        print(f"  {R}x{C}: {'ok' if ok else 'FAIL'}  ({(time.perf_counter() - t) * 1000:.0f} ms)")
    return ok_all


def test_reader(rng):
    print("Reader (simulated play from an empty digit memory):")
    plan = [(1, 3, 85), (5, 5, 170), (5, 5, 170), (5, 5, 70), (10, 10, 90), (10, 10, 45),
            (15, 15, 60), (15, 15, 32), (20, 20, 45), (20, 20, 25), (20, 20, 25)]
    ok_all, total_asked = True, 0
    book = pb.DigitBook()
    for R, C, cell in plan:
        truth = (rng.random((R, C)) < 0.6).astype(int)
        if R * C <= 3:
            truth[:] = 1
        slots = []
        img, box, clue, rows, cols = render_board(truth, cell, slots)
        found = pb.locate_board(img)[:2] == (box, clue)       # it must find the board itself
        rd = pb.analyse(img, box, clue, book, ask=answer_from(slots))
        total_asked += rd.asked
        misread = rd.solution is not None and (rd.rows != rows or rd.cols != cols)
        ok = found and rd.solution is not None and not misread
        ok_all &= ok
        status = "ok" if ok else ("board not found" if not found else
                                  "MISREAD" if misread else f"FAIL ({rd.error})")
        print(f"  {R}x{C} @ {cell}px: asked {rd.asked}, worked out {rd.auto}, "
              f"confirmed {rd.confirmed}, fixed {rd.corrected} -> {status}")
        if not ok:
            pb.save_debug(img, rd, f"selftest_fail_{R}x{C}.png")
    print(f"  questions asked in total: {total_asked}")
    return ok_all


def test_cell_states(rng):
    print("Cell states (filled and crossed-out cells):")
    truth = (rng.random((10, 10)) < 0.6).astype(int)
    filled = truth * (rng.random(truth.shape) < 0.5)
    crossed = np.zeros_like(truth)
    crossed[0, :] = 1 - truth[0, :]
    slots = []
    img, box, clue, rows, cols = render_board(truth, 50, slots, filled=filled, crossed=crossed)
    rd = pb.analyse(img, box, clue, pb.DigitBook(), ask=answer_from(slots))
    ok = rd.filled is not None and np.array_equal(rd.filled, filled | crossed)
    print(f"  {'ok' if ok else 'FAIL'}")
    return ok


def main():
    rng = np.random.default_rng(7)
    results = [test_solver(rng), test_reader(rng), test_cell_states(rng)]
    print("All good." if all(results) else "Some checks FAILED.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
