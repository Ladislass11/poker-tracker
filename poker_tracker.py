# -*- coding: utf-8 -*-
"""
Трекер раздач для покера Majestic.
Сам делает снимки экрана, распознаёт ваши карты и карты стола,
записывает каждую раздачу в hands.csv и считает статистику:
совпадает ли то, что раздаёт игра, с теорией вероятностей.

Запуск:
  python poker_tracker.py            - следить за экраном (Ctrl+C - стоп)
  python poker_tracker.py --hints    - то же + окно с подсказками поверх игры
  python poker_tracker.py --report   - показать отчёт по записанным раздачам
  python poker_tracker.py --check    - один снимок экрана с разметкой (check.png)
  python poker_tracker.py --test ПАПКА - распознать скриншоты из папки
"""
import argparse, csv, math, os, sys, time
from collections import Counter

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "hands.csv")
RANKS = "23456789TJQKA"
SUITS = "shdc"
SUIT_SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RANK_SHOW = {"T": "10"}

# ---------------------------------------------------------------- распознавание
CW, CH = 100, 148
_t = np.load(os.path.join(HERE, "templates.npz"))
R_KEYS = [k[2:] for k in _t.files if k.startswith("r_")]
S_KEYS = [k[2:] for k in _t.files if k.startswith("s_")]
R_T = np.stack([_t["r_" + k] for k in R_KEYS])
S_T = np.stack([_t["s_" + k] for k in S_KEYS])


def _white_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 150) & (hsv[..., 1] < 45)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _card_boxes(img):
    """Ищет белые прямоугольники размером с карту (включая слипшиеся ряды)."""
    H = img.shape[0]
    m = _white_mask(img)
    k = max(5, int(H * 0.0055)) | 1  # стираем тонкие белые линии на сукне, которые "приклеиваются" к картам
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < H * 0.07 or h > H * 0.14:
            continue
        if cv2.contourArea(c) / (w * h) < 0.8:
            continue
        n = round(w / (h * 0.68 + 4))
        if n < 1 or n > 5:
            continue
        cw = (w - (n - 1) * 5) / n
        if not 0.58 < cw / h < 0.78:
            continue
        for i in range(n):
            out.append((int(x + i * (cw + 5)), y, int(cw), h))
    return out


def _ink(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(int)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(int)
    border = np.concatenate([g[5:-5, 3:8].ravel(), g[5:-5, -8:-3].ravel()])
    sb = np.concatenate([hsv[5:-5, 3:8, 1].ravel(), hsv[5:-5, -8:-3, 1].ravel()])
    return ((g < np.median(border) - 45) | (hsv[..., 1] > np.median(sb) + 70)).astype(np.uint8)


def _tight(a, size):
    ys, xs = np.nonzero(a)
    if len(ys) < 15:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(a.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def _is_red(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)).sum()
    dark = ((hsv[..., 2] < 100) & (hsv[..., 1] < 80)).sum()
    return red > dark


def _is_back(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] > 110) & (hsv[..., 0] < 165) & (hsv[..., 1] > 50)).mean() > 0.08


def read_card(crop):
    """Возвращает код карты ('Ah', 'Tc' ...) или None, если это не карта."""
    if _is_back(crop):
        return None
    m = _ink(crop)
    r = _tight(m[int(CH * .06):int(CH * .52), int(CW * .15):int(CW * .85)], (20, 28))
    s = _tight(m[int(CH * .55):int(CH * .92), int(CW * .2):int(CW * .8)], (20, 20))
    if r is None or s is None:
        return None
    dr = np.abs(R_T - r).mean((1, 2))
    ds = np.abs(S_T - s).mean((1, 2))
    red = _is_red(crop)
    for i, k in enumerate(S_KEYS):  # цвет масти должен совпасть
        if (k in "hd") != red:
            ds[i] += 1
    if dr.min() > 0.15 or ds.min() > 0.10:
        return None
    return R_KEYS[int(dr.argmin())] + S_KEYS[int(ds.argmin())]


def read_screen(img):
    """Находит открытые карты. Возвращает (стол, остальные карты [(код, x, y)])."""
    H, W = img.shape[:2]
    board, other = [], []
    for (x, y, w, h) in _card_boxes(img):
        crop = cv2.resize(img[y:y + h, x:x + w], (CW, CH), interpolation=cv2.INTER_AREA)
        code = read_card(crop)
        if not code:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        if 0.36 < cx < 0.64 and 0.25 < cy < 0.47:
            board.append((x, code))
        else:
            other.append((code, cx, cy))
    board = [c for _, c in sorted(board)]
    return board, other


def card_str(c):
    return RANK_SHOW.get(c[0], c[0]) + SUIT_SYM[c[1]]


# ---------------------------------------------------------------- трекер
class Tracker:
    """Склеивает снимки экрана в раздачи."""

    def __init__(self, writer):
        self.writer = writer
        self.hand_no = self._count_existing()
        self.reset()
        self.last_read = None
        self.stable = 0

    def _count_existing(self):
        if not os.path.exists(CSV_PATH):
            return 0
        with open(CSV_PATH, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)

    def reset(self):
        self.hole = None
        self.board = []

    def feed(self, img):
        board, other = read_screen(img)
        key = (tuple(board), tuple(sorted(o[0] for o in other)))
        # берём только кадр, повторившийся 2 раза подряд (не во время анимации)
        if key != self.last_read:
            self.last_read, self.stable = key, 1
            return
        self.stable += 1
        if self.stable != 2:
            return

        if not board:
            if self.board:  # стол очистился - раздача закончилась
                self.finish()
            # до флопа открыты только ваши две карты
            if len(other) == 2 and abs(other[0][2] - other[1][2]) < 0.03 and abs(other[0][1] - other[1][1]) < 0.08:
                hole = tuple(sorted(o[0] for o in other))
                if hole != self.hole:
                    if self.hole:
                        self.finish()
                    self.hole = hole
                    print(f"\nРаздача {self.hand_no + 1}: ваши карты {card_str(hole[0])} {card_str(hole[1])}", flush=True)
            return
        if len(board) < 3:
            return
        if self.board and board[:3] != self.board[:3]:  # новый флоп без пустого стола между
            self.finish()
        if len(board) > len(self.board) and board[:len(self.board)] == self.board:
            self.board = board
            name = {3: "Флоп", 4: "Тёрн", 5: "Ривер"}.get(len(board), "Стол")
            print(f"  {name}: {' '.join(card_str(c) for c in board)}", flush=True)

    def finish(self):
        if self.hole or self.board:
            cards = list(self.hole or []) + self.board
            if len(set(cards)) != len(cards):  # одна карта дважды = ошибка распознавания
                print("  (раздача пропущена: ошибка распознавания)")
            else:
                self.hand_no += 1
                h = list(self.hole) if self.hole else ["", ""]
                b = self.board + [""] * (5 - len(self.board))
                self.writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + h + b)
                self.writer.f.flush()
        self.reset()


class CsvWriter:
    def __init__(self, path):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "hole1", "hole2", "flop1", "flop2", "flop3", "turn", "river"])

    def writerow(self, r):
        self.w.writerow(r)


def grab_screen(sct, monitor):
    shot = np.array(sct.grab(sct.monitors[monitor]))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def run_live(monitor, fps):
    import mss
    w = CsvWriter(CSV_PATH)
    tr = Tracker(w)
    print("Трекер запущен. Играйте как обычно, окно можно свернуть.")
    print("Раздачи пишутся в hands.csv. Остановить: Ctrl+C, потом run_report.bat\n")
    with (mss.MSS() if hasattr(mss, 'MSS') else mss.mss()) as sct:
        try:
            while True:
                t0 = time.time()
                tr.feed(grab_screen(sct, monitor))
                time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        except KeyboardInterrupt:
            tr.finish()
            print(f"\nОстановлено. Всего записано раздач: {tr.hand_no}")


def run_check(monitor):
    import mss
    with (mss.MSS() if hasattr(mss, 'MSS') else mss.mss()) as sct:
        img = grab_screen(sct, monitor)
    annotate(img, os.path.join(HERE, "check.png"))


def annotate(img, out_path):
    H, W = img.shape[:2]
    board, other = read_screen(img)
    vis = img.copy()
    for (x, y, w, h) in _card_boxes(img):
        crop = cv2.resize(img[y:y + h, x:x + w], (CW, CH), interpolation=cv2.INTER_AREA)
        code = read_card(crop)
        col = (0, 200, 0) if code else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 3)
        cv2.putText(vis, code or "?", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 3)
    cv2.imwrite(out_path, vis)
    print("Стол:", " ".join(card_str(c) for c in board) or "-")
    print("Другие открытые карты:", " ".join(card_str(o[0]) for o in other) or "-")
    print("Картинка с разметкой:", out_path)


def run_test(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        board, other = read_screen(img)
        print(f"{f}: стол [{' '.join(card_str(c) for c in board)}]  другие [{' '.join(card_str(o[0]) for o in other)}]")


# ---------------------------------------------------------------- отчёт
def _z_verdict(k, n, p):
    if n == 0:
        return "нет данных", None
    if n * p < 5 or n * (1 - p) < 5:
        return "мало данных", None
    z = (k - n * p) / math.sqrt(n * p * (1 - p))
    a = abs(z)
    v = "норма" if a < 2 else ("немного необычно" if a < 3 else "ПОДОЗРИТЕЛЬНО")
    return v, z


def _chi_verdict(counts, n):
    k = len(counts)
    if n < 5 * k:
        return "мало данных", None
    e = n / k
    chi = sum((c - e) ** 2 / e for c in counts)
    df = k - 1
    z = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))  # Уилсон-Хилферти
    v = "норма" if z < 2 else ("немного необычно" if z < 3 else "ПОДОЗРИТЕЛЬНО")
    return v, z


def load_hands(path=CSV_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(path=CSV_PATH, out=None):
    hands = load_hands(path)
    lines = []
    P = lines.append
    P(f"ОТЧЁТ ПО РАЗДАЧАМ ({len(hands)} шт.)")
    P("Колонки: факт / теория / сколько раздач / вывод")
    P("Вывод 'норма' = отклонение в пределах обычного везения (|z|<2).\n")

    rows = []

    def add(name, k, n, p):
        v, z = _z_verdict(k, n, p)
        fact = f"{100 * k / n:5.1f}%" if n else "  -  "
        zs = f"z={z:+.1f}" if z is not None else ""
        rows.append(f"{name:<44}{fact:>8}{100 * p:8.1f}%{n:>7}   {v} {zs}")

    with_hole = [h for h in hands if h["hole1"]]
    pp = [h for h in with_hole if h["hole1"][0] == h["hole2"][0]]
    suited = [h for h in with_hole if h["hole1"][1] == h["hole2"][1]]
    P("--- Ваши стартовые карты ---")
    add("Карманная пара", len(pp), len(with_hole), 3 / 51)
    add("Одномастные карты", len(suited), len(with_hole), 12 / 51)
    add("Туз хотя бы одной картой", sum(1 for h in with_hole if "A" in (h["hole1"][0] + h["hole2"][0])), len(with_hole), 1 - (48 * 47) / (52 * 51))
    lines += rows; rows.clear()

    flopped = [h for h in with_hole if h["flop3"]]
    pp_f = [h for h in flopped if h["hole1"][0] == h["hole2"][0]]
    up_f = [h for h in flopped if h["hole1"][0] != h["hole2"][0]]
    su_f = [h for h in flopped if h["hole1"][1] == h["hole2"][1]]
    P("\n--- Флоп (видно и после вашего фолда) ---")
    add("Пара: сет или каре на флопе", sum(1 for h in pp_f if h["hole1"][0] in [h[c][0] for c in ("flop1", "flop2", "flop3")]), len(pp_f), 1 - 17296 / 19600)
    add("Разные карты: попал хотя бы одной", sum(1 for h in up_f if {h["hole1"][0], h["hole2"][0]} & {h[c][0] for c in ("flop1", "flop2", "flop3")}), len(up_f), 1 - 13244 / 19600)
    fd = [h for h in su_f if sum(1 for c in ("flop1", "flop2", "flop3") if h[c][1] == h["hole1"][1]) == 2]
    add("Одномастные: флеш-дро на флопе", len(fd), len(su_f), 55 * 39 / 19600)
    add("Одномастные: готовый флеш на флопе", sum(1 for h in su_f if all(h[c][1] == h["hole1"][1] for c in ("flop1", "flop2", "flop3"))), len(su_f), 165 / 19600)
    lines += rows; rows.clear()

    rivered = [h for h in with_hole if h["river"]]
    pp_r = [h for h in rivered if h["hole1"][0] == h["hole2"][0]]
    up_r = [h for h in rivered if h["hole1"][0] != h["hole2"][0]]
    fd_r = [h for h in fd if h["river"]]
    B5 = ("flop1", "flop2", "flop3", "turn", "river")
    P("\n--- До ривера ---")
    add("Пара: сет к риверу", sum(1 for h in pp_r if h["hole1"][0] in [h[c][0] for c in B5]), len(pp_r), 1 - 1712304 / 2118760)
    add("Разные карты: пара к риверу", sum(1 for h in up_r if {h["hole1"][0], h["hole2"][0]} & {h[c][0] for c in B5}), len(up_r), 1 - 1086008 / 2118760)
    add("Флеш-дро на флопе собрался к риверу", sum(1 for h in fd_r if sum(1 for c in B5 if h[c][1] == h["hole1"][1]) >= 3), len(fd_r), 1 - 703 / 1081)
    lines += rows; rows.clear()

    board_cards = [h[c] for h in hands for c in B5 if h[c]]
    P(f"\n--- Карты стола: равномерность ({len(board_cards)} карт) ---")
    rc = Counter(c[0] for c in board_cards)
    sc = Counter(c[1] for c in board_cards)
    v, z = _chi_verdict([rc.get(r, 0) for r in RANKS], len(board_cards))
    P("Номиналы: " + " ".join(f"{RANK_SHOW.get(r, r)}:{rc.get(r, 0)}" for r in RANKS))
    P(f"  ожидается по ~{len(board_cards) / 13:.0f} каждого -> {v}" + (f" (z={z:+.1f})" if z is not None else ""))
    v, z = _chi_verdict([sc.get(s, 0) for s in SUITS], len(board_cards))
    P("Масти:    " + " ".join(f"{SUIT_SYM[s]}:{sc.get(s, 0)}" for s in SUITS))
    P(f"  ожидается по ~{len(board_cards) / 4:.0f} каждой -> {v}" + (f" (z={z:+.1f})" if z is not None else ""))

    hole_cards = [h[c] for h in with_hole for c in ("hole1", "hole2")]
    if hole_cards:
        P(f"\n--- Ваши карты: равномерность ({len(hole_cards)} карт) ---")
        rc = Counter(c[0] for c in hole_cards)
        v, z = _chi_verdict([rc.get(r, 0) for r in RANKS], len(hole_cards))
        P("Номиналы: " + " ".join(f"{RANK_SHOW.get(r, r)}:{rc.get(r, 0)}" for r in RANKS))
        P(f"  -> {v}" + (f" (z={z:+.1f})" if z is not None else ""))

    P("\nКак читать: при 200+ раздачах 'норма' почти везде значит, что раздача")
    P("честная. Одна строка 'немного необычно' бывает и при честной игре")
    P("(из ~15 проверок одна такая выпадает часто). Тревожно, если")
    P("'ПОДОЗРИТЕЛЬНО' держится и после 500+ раздач.")
    text = "\n".join(lines)
    print(text)
    out = out or os.path.join(HERE, "report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nОтчёт сохранён: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--test")
    ap.add_argument("--monitor", type=int, default=1, help="номер монитора (1 = основной)")
    ap.add_argument("--fps", type=float, default=3)
    ap.add_argument("--hints", action="store_true", help="окно с подсказками поверх игры")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.check:
        run_check(a.monitor)
    elif a.test:
        run_test(a.test)
    elif a.hints:
        import overlay
        overlay.run(a.monitor, a.fps)
    else:
        run_live(a.monitor, a.fps)
