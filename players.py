# -*- coding: utf-8 -*-
"""Панели игроков: ID, стек, действие. Профили соперников между сессиями."""
import csv
import json
import os
import time
from collections import Counter

import cv2
import numpy as np

import poker_tracker as pt

_t = pt._t
W_T = {k[2:]: _t[k] for k in _t.files if k.startswith("w_")}
W_KEYS = list(W_T)
W_M = np.stack([W_T[k] for k in W_KEYS])
ST_T = {k[2:]: _t[k] for k in _t.files if k.startswith("t_")}
ST_R = {k[2:]: float(_t[k]) for k in _t.files if k.startswith("q_")}
SHORT = {"Повысил ставку": "raise", "Поставил ставку": "bet", "Уравнял ставку": "call",
         "Пропустил ход": "check", "Сбросил карты": "fold"}
PROFILES = os.path.join(pt.HERE, "players.json")
ACTIONS_LOG = os.path.join(pt.HERE, "actions_log.csv")

# места игроков (левый верхний угол панели) для экрана 2559x1439, дальше масштабируется
SLOTS = [(136, 193), (2096, 193), (136, 793), (2096, 793), (723, 1075), (1509, 1075)]


def _comps(ink, min_h):
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    cs = sorted(tuple(int(a) for a in st[i][:4]) for i in range(1, n) if st[i][3] >= min_h)
    out = []
    for x, y, w, h in cs:
        k = int(round(w / (h * 0.82)))
        if k >= 2 and w > h * 1.5:  # слипшиеся цифры
            out += [(x + round(j * w / k), y, round(w / k), h) for j in range(k)]
        else:
            out.append((x, y, w, h))
    return out


def _read_glyphs(ink, cs):
    s = ""
    for x, y, w, h in cs:
        g = cv2.resize(ink[y:y + h, x:x + w].astype(np.float32), (14, 20), interpolation=cv2.INTER_AREA)
        d = np.abs(W_M - g[None]).mean((1, 2))
        if d.min() > 0.2:
            return None
        s += W_KEYS[int(d.argmin())]
    return s


def read_seat(img, i):
    H, W = img.shape[:2]
    sx, sy = SLOTS[i]
    x0, y0 = int(sx / 2559 * W), int((sy - 70) / 1439 * H)
    x1, y1 = int((sx + 334) / 2559 * W), int((sy + 106) / 1439 * H)
    c = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., k].astype(int) for k in range(3)]
    coin = ((h >= 12) & (h <= 28) & (s > 150) & (v > 140)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(coin)
    cs = [tuple(st[j][:4]) for j in range(1, n)
          if H * .008 <= st[j][3] < H * .03 and st[j][2] < H * .03 and st[j][0] < c.shape[1] * .45]
    if not cs:
        return None
    cx, cy, cw, ch = max(cs, key=lambda a: a[2] * a[3])
    res = {"seat": i, "id": None, "me": False, "status": None, "stack": None}
    # стек
    band = c[max(0, cy - 2):cy + ch + 2, cx + cw + 2:]
    bh = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    bg, top = np.median(bh[..., 2]), np.percentile(bh[..., 2], 99)
    ink = ((bh[..., 2] > bg + max(55, 0.6 * (top - bg))) & (bh[..., 1] < 70)).astype(np.uint8)
    sc = _comps(ink, int(H * .008))
    if 1 <= len(sc) <= 8:
        txt = _read_glyphs(ink, sc)
        if txt and txt.isdigit():
            res["stack"] = int(txt)
    # строка над стеком: ID, "Вы" или действие
    nb = c[max(0, cy - int(H * .036)):cy - int(H * .004), max(0, cx - 2):int(c.shape[1] * .86)]  # без значка D
    nh = cv2.cvtColor(nb, cv2.COLOR_BGR2HSV)
    # жёлтый текст действия; у сбросившего игрока панель затемнена, поэтому порог яркости
    # берём относительно самого яркого жёлтого в строке (в тёмных залах текст тусклее)
    yl = (nh[..., 0] >= 18) & (nh[..., 0] <= 35) & (nh[..., 1] > 90)
    vmax = float(nh[..., 2][yl].max()) if yl.any() else 0
    yellow = (yl & (nh[..., 2] > max(70, 0.6 * vmax))).astype(np.uint8)
    if yellow.sum() > 40:
        ys, xs = np.nonzero(yellow)
        crop = yellow[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        g = cv2.resize(crop.astype(np.float32), (120, 12), interpolation=cv2.INTER_AREA)
        ratio = crop.shape[1] / max(1, crop.shape[0])
        d = {k: np.abs(g - t).mean() + abs(ratio - ST_R[k]) * 0.05 for k, t in ST_T.items()}
        k = min(d, key=d.get)
        if d[k] < 0.25:
            res["status"] = SHORT[k]
    else:
        bg, top = np.median(nh[..., 2]), np.percentile(nh[..., 2], 99)
        wink = ((nh[..., 2] > bg + max(55, 0.6 * (top - bg))) & (nh[..., 1] < 70)).astype(np.uint8)
        wc = _comps(wink, int(H * .008))
        first = _read_glyphs(wink, wc[:1]) if wc else None
        txt = _read_glyphs(wink, wc) if wc else None
        if first == "B" and len(wc) <= 3:
            res["me"] = True
        elif txt and txt.startswith("#") and 3 <= len(txt) <= 8 and txt[1:].isdigit():
            res["id"] = txt
    # в раздаче ли игрок: рубашки над панелью
    bx0, by0 = int((sx + 58) / 2559 * W), int((sy - 62) / 1439 * H)
    bx1, by1 = int((sx + 275) / 2559 * W), int((sy - 2) / 1439 * H)
    bk = cv2.cvtColor(img[max(0, by0):by1, bx0:bx1], cv2.COLOR_BGR2HSV)
    res["in_hand"] = bool(((bk[..., 0] > 110) & (bk[..., 0] < 165) & (bk[..., 1] > 50)).mean() > 0.08)
    return res


def read_seats(img):
    return [r for r in (read_seat(img, i) for i in range(len(SLOTS))) if r]


# ---------------------------------------------------------------- профили
class Profiles:
    FIELDS = ["hands", "vpip", "pfr", "bet", "raise", "call", "check", "fold",
              "faced", "fold_vs_bet", "allin", "shows"]

    def __init__(self, path=PROFILES):
        self.path = path
        try:
            with open(path, encoding="utf-8") as f:
                self.p = json.load(f)
        except Exception:
            self.p = {}
        self.seat_id = {}        # место -> ID (держится, пока не появится другой)
        self.id_votes = {}       # место -> последние прочтения номера
        self.last_status = {}
        self.street = None
        self.street_bet = False
        self.hand_flags = {}     # ID -> set флагов в этой раздаче
        self.log_path = ACTIONS_LOG if path == PROFILES else path + ".log.csv"
        self.dealer = None
        self.last_stack = {}
        self.raw_stack = {}
        self.pending = {}
        self.prev_board = 0

    def get(self, pid):
        d = self.p.setdefault(pid, {})
        for f in self.FIELDS:
            d.setdefault(f, [] if f == "shows" else 0)
        return d

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.p, f, ensure_ascii=False)
        except Exception:
            pass

    def new_hand(self):
        # защита от двойного срабатывания (смена дилера, очистка стола и ваши новые карты - почти одновременно)
        ticks = getattr(self, "_ticks", 0)
        if ticks - getattr(self, "_last_new", -100) < 10 and self.street == 0:
            return  # новая раздача уже засчитана пару секунд назад
        self._last_new = ticks
        self.hand_flags = {}
        self.last_status = {}
        self.street = 0
        self.street_bet = True  # до флопа блайнды = ставка
        self.hand_no = getattr(self, "hand_no", 0) + 1
        self.pending = {}
        self.save()

    def _log(self, row):
        """Журнал действий: по нему потом считаются размеры рейзов и что боты показывают."""
        try:
            new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "hand", "street", "seat", "player", "action", "stack", "put_in", "info"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + row)
        except Exception:
            pass

    def update(self, seats, board_len, dealer=None):
        self._ticks = getattr(self, "_ticks", 0) + 1
        if self.street is None:
            self.new_hand()
        # новая раздача: значок D переехал (видно 2 кадра подряд) или стол очистился после флопа
        cand, self._d_prev = (dealer if dealer is not None and dealer == getattr(self, "_d_prev", None) else None), dealer
        dealer = cand
        if dealer is not None and self.dealer is not None and dealer != self.dealer:
            self.new_hand()
        elif board_len == 0 and self.prev_board >= 3 and self.dealer is None:
            self.new_hand()  # запасной вариант, если значок дилера не виден
        if dealer is not None:
            self.dealer = dealer
        self.prev_board = board_len
        if board_len != self.street and board_len in (3, 4, 5):
            self.street = board_len
            self.street_bet = False
            self.last_status = {}
        for s in seats:
            i = s["seat"]
            if s["me"]:
                self.seat_id[i] = "me"
                continue
            if s["id"]:
                # голосуем только по последним 5 прочтениям: если на место сел другой игрок,
                # его номер быстро вытеснит прежний
                v = self.id_votes.setdefault(i, [])
                v.append(s["id"])
                del v[:-5]
                best, n = Counter(v).most_common(1)[0]
                if n >= 3 and self.seat_id.get(i) != best:
                    self.seat_id[i] = best
            pid = self.seat_id.get(i)
            if not pid or pid == "me":
                continue
            prof = self.get(pid)
            flags = self.hand_flags.setdefault(pid, set())
            if s["in_hand"] and "dealt" not in flags and self.street == 0:
                flags.add("dealt")
                prof["hands"] += 1
            st = s["status"]
            stack = s["stack"]
            street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(self.street, self.street)
            # стек в игре "докручивается" анимацией: верим только значению, прочитанному 2 раза подряд
            stable = None
            if stack is not None and stack == self.raw_stack.get(i):
                stable = stack
            self.raw_stack[i] = stack
            if st and st != self.last_status.get(i):
                self.last_status[i] = st
                self._event(prof, flags, st)
                self.pending[i] = (self.last_stack.get(i), st, street)
                self._log([self.hand_no, street, i, pid, st, self.last_stack.get(i, ""), "", ""])
            if stable is not None:
                if i in self.pending and stable != self.pending[i][0]:
                    before, act, stt = self.pending.pop(i)
                    if before is not None and before > stable:
                        self._log([self.hand_no, stt, i, pid, act + "_amount", stable, before - stable, ""])
                self.last_stack[i] = stable

    def _event(self, prof, flags, st):
        pre = self.street == 0
        if st in ("bet", "raise"):
            if pre and st == "bet":
                return  # до флопа это обычно блайнд
            if self.street_bet and not pre:
                prof["faced"] += 1
            prof[st] += 1
            self.street_bet = True
            if pre:
                if "vpip" not in flags:
                    flags.add("vpip")
                    prof["vpip"] += 1
                if "pfr" not in flags:
                    flags.add("pfr")
                    prof["pfr"] += 1
        elif st == "call":
            prof["call"] += 1
            if not pre:
                prof["faced"] += 1
            if pre and "vpip" not in flags:
                flags.add("vpip")
                prof["vpip"] += 1
        elif st == "check":
            prof["check"] += 1
        elif st == "fold":
            prof["fold"] += 1
            if self.street_bet and not pre:
                prof["faced"] += 1
                prof["fold_vs_bet"] += 1

    def add_show(self, seat, cards, board):
        pid = self.seat_id.get(seat)
        if pid and pid != "me":
            prof = self.get(pid)
            entry = " ".join(cards) + " | " + " ".join(board)
            if entry not in prof["shows"]:
                prof["shows"] = (prof["shows"] + [entry])[-8:]
                self._log([getattr(self, "hand_no", 0), "showdown", seat, pid, "show", "", "", entry])

    def stats(self, pid):
        d = self.get(pid)
        n = d["hands"]
        aggr = d["bet"] + d["raise"]
        return {
            "hands": n,
            "vpip": d["vpip"] / n if n else None,
            "af": aggr / d["call"] if d["call"] else (float(aggr) if aggr else None),
            "fvb": d["fold_vs_bet"] / d["faced"] if d["faced"] >= 5 else None,
            "faced": d["faced"],
        }

    def summary(self, pid):
        s = self.stats(pid)
        if s["hands"] < 10:
            return f"{pid}: мало данных ({s['hands']} р.)"
        tags = []
        if s["vpip"] is not None:
            tags.append("играет много рук" if s["vpip"] > 0.45 else ("тайтовый" if s["vpip"] < 0.2 else f"входит {s['vpip']:.0%}"))
        if s["af"] is not None and d_enough(self.get(pid)):
            tags.append("агрессивный" if s["af"] >= 2.5 else ("пассивный" if s["af"] <= 0.8 else "средняя агрессия"))
        if s["fvb"] is not None:
            tags.append(f"фолд на ставку {s['fvb']:.0%}")
        return f"{pid} ({s['hands']} р.): " + ", ".join(tags)

    def style(self, pid):
        """'aggr' / 'passive' / None - для поправки диапазона."""
        if not pid or pid == "me":
            return None
        d = self.get(pid)
        if not d_enough(d):
            return None
        s = self.stats(pid)
        if s["af"] is None:
            return None
        return "aggr" if s["af"] >= 2.5 else ("passive" if s["af"] <= 0.8 else None)


def d_enough(d):
    return d["bet"] + d["raise"] + d["call"] >= 8


# ---------------------------------------------------------------- дилер и позиция
ORDER = [1, 3, 5, 4, 2, 0]   # места по часовой стрелке (так идёт ход)
D_T = _t["dealer_D"] if "dealer_D" in _t.files else None
POS_NAME = {"btn": "баттон", "sb": "малый блайнд", "bb": "большой блайнд",
            "early": "ранняя позиция", "middle": "средняя позиция", "late": "поздняя позиция (перед баттоном)"}


def find_dealer(img):
    """Место, у панели которого красный значок D."""
    H, W = img.shape[:2]
    best = None
    for i, (sx, sy) in enumerate(SLOTS):
        x0, x1 = int((sx + 250) / 2559 * W), int((sx + 340) / 2559 * W)
        y0, y1 = int((sy - 75) / 1439 * H), int((sy + 80) / 1439 * H)
        hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 140) & (hsv[..., 2] > 130)).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(red)
        for j in range(1, n):
            x, y, w, h, a = st[j]
            if 0.012 * H < h < 0.03 * H and 0.7 < w / h < 1.4 and a > 0.4 * w * h:
                sq = cv2.cvtColor(img[y0 + y:y0 + y + h, x0 + x:x0 + x + w], cv2.COLOR_BGR2HSV)
                g = cv2.resize(((sq[..., 2] > 200) & (sq[..., 1] < 60)).astype(np.float32), (16, 16),
                               interpolation=cv2.INTER_AREA)
                d = np.abs(g - D_T).mean() if D_T is not None else 0
                if d < 0.15 and (best is None or d < best[0]):
                    best = (d, i)
    return best[1] if best else None


def my_position(seats, dealer):
    """Ваша позиция: btn / sb / bb / early / middle / late."""
    me = next((s["seat"] for s in seats if s["me"]), None)
    if me is None or dealer is None:
        return None
    # пропускаем пустые места и тех, у кого 0 фишек и нет карт (выбыл / сидит без игры)
    active = {s["seat"] for s in seats if s["me"] or s["in_hand"] or (s["stack"] or 0) > 0 or s["seat"] == dealer}
    occupied = [i for i in ORDER if i in active]
    if dealer not in occupied or me not in occupied:
        return None
    k = occupied.index(dealer)
    ring = occupied[k:] + occupied[:k]
    off, n = ring.index(me), len(ring)
    if n == 2:
        return "btn" if off == 0 else "bb"
    if off == 0:
        return "btn"
    if off == 1:
        return "sb"
    if off == 2:
        return "bb"
    if off == n - 1:
        return "late"
    return "early" if off == 3 and n >= 6 else "middle"
