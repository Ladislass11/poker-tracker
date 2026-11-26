# -*- coding: utf-8 -*-
"""Чтение сумм с экрана, подсчёт шансов и советы."""
import random
from collections import Counter
from itertools import combinations

import cv2
import numpy as np

import poker_tracker as pt

_t = pt._t
B_T = {k[2:]: _t[k] for k in _t.files if k.startswith("b_")}
P_T = {k[2:]: _t[k] for k in _t.files if k.startswith("p_")}


# ---------------------------------------------------------------- суммы на экране
def _digits(ink, comps, T):
    s = ""
    for x, y, w, h in comps:
        g = cv2.resize(ink[y:y + h, x:x + w].astype(np.float32), (14, 20), interpolation=cv2.INTER_AREA)
        d = {k: np.abs(g - v).mean() for k, v in T.items()}
        k = min(d, key=d.get)
        if d[k] > 0.25:
            return None
        s += k
    return int(s) if s else None


def _components(ink, min_h):
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    return sorted(tuple(int(a) for a in st[i][:4]) for i in range(1, n) if st[i][3] >= min_h)


def read_buttons(img):
    """Возвращает (мой_ход, сумма_колла). сумма 0 = можно чекнуть, None = кнопок нет."""
    H, W = img.shape[:2]
    x0, y0 = int(W * .6), int(H * .8)
    hsv = cv2.cvtColor(img[y0:, x0:], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 225) & (hsv[..., 1] < 25)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 0.10 * W < w < 0.16 * W and 0.04 * H < h < 0.07 * H:
            bs.append((x + x0, y + y0, w, h))
    if len(bs) < 3:
        return False, "none"  # кнопок нет: вы не в раздаче или идёт вскрытие
    x, y, w, h = sorted(bs, key=lambda r: (r[1] // 20, r[0]))[0]
    icon = cv2.cvtColor(img[y + int(h * .3):y + int(h * .7), x + int(w * .03):x + int(w * .11)], cv2.COLOR_BGR2HSV)
    my_turn = (((icon[..., 0] < 10) | (icon[..., 0] > 165)) & (icon[..., 1] > 120)).mean() > 0.05
    b = img[y + 4:y + h - 4, x + int(w * .12):x + w - 6]
    bh = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    ink = ((bh[..., 2] < 120) | ((bh[..., 1] > 120) & (bh[..., 2] > 80))).astype(np.uint8)
    comps = [c for c in _components(ink, int(h * .17)) if c[0] > b.shape[1] * 0.04]
    # CHECK = 5 букв без числа; CALL = 4 буквы + цифры суммы (пробел бывает узким, на него не опираемся)
    if len(comps) < 6:
        return my_turn, 0
    return my_turn, _digits(ink, comps[4:], B_T)  # None = сумму не удалось прочитать


def read_pot(img):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(W * .42), int(H * .12), int(W * .58), int(H * .30)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(int)
    s = hsv[..., 1].astype(int)
    best = None
    for lo, hi in ((21, 32), (18, 38), (0, 55)):
        m = ((v >= lo) & (v <= hi) & (s < 30)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(H * .03), int(H * .03)), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if not (0.05 * W < w < 0.11 * W and 0.035 * H < h < 0.075 * H and 1.8 < w / h < 3.4):
                continue
            if cv2.contourArea(c) / (w * h) < 0.85:
                continue
            d = abs(x + w / 2 - (x1 - x0) / 2)
            if best is None or d < best[0]:
                best = (d, x0 + x, y0 + y, w, h)
        if best:
            break
    if best is None:
        return None
    _, x, y, w, h = best
    box = img[y + int(h * .5):y + h - 2, x + 3:x + w - 3]
    bh = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    ink = ((bh[..., 2] > 165) & (bh[..., 1] < 70)).astype(np.uint8)
    return _digits(ink, _components(ink, box.shape[0] * .4), P_T)


def count_opponents(img):
    """Сколько соперников ещё в раздаче (у кого лежат рубашки карт)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 110) & (hsv[..., 0] < 165) & (hsv[..., 1] > 50)).astype(np.uint8) * 255
    k = max(3, int(H * .01))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        if 0.36 < cx < 0.64 and 0.25 < cy < 0.47:
            continue  # закрытые карты стола
        if 0.03 * H < h < 0.13 * H and 0.03 * W < w < 0.11 * W:
            n += 1
    return max(1, n)


# ---------------------------------------------------------------- оценка рук
RV = {r: i for i, r in enumerate(pt.RANKS, 2)}
CAT = ["Старшая карта", "Пара", "Две пары", "Сет / трипс", "Стрит", "Флеш", "Фулл-хаус", "Каре", "Стрит-флеш"]


def _straight(vals):
    v = sorted(set(vals), reverse=True)
    if 14 in v:
        v.append(1)
    run = 1
    for i in range(1, len(v)):
        run = run + 1 if v[i] == v[i - 1] - 1 else 1
        if run >= 5:
            return v[i] + 4
    return 0


def evaluate(cards):
    """Сила руки из 5-7 карт: кортеж, больше = сильнее."""
    vals = [RV[c[0]] for c in cards]
    suits = Counter(c[1] for c in cards)
    fs = next((s for s, n in suits.items() if n >= 5), None)
    if fs:
        fv = [RV[c[0]] for c in cards if c[1] == fs]
        sf = _straight(fv)
        if sf:
            return (8, sf)
    cnt = Counter(vals)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if groups[0][1] == 4:
        k = max(v for v in vals if v != groups[0][0])
        return (7, groups[0][0], k)
    if groups[0][1] == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, groups[0][0], groups[1][0])
    if fs:
        return (5,) + tuple(sorted(fv, reverse=True)[:5])
    st = _straight(vals)
    if st:
        return (4, st)
    kick = sorted((v for v in vals), reverse=True)
    if groups[0][1] == 3:
        return (3, groups[0][0]) + tuple([v for v in kick if v != groups[0][0]][:2])
    if groups[0][1] == 2 and groups[1][1] == 2:
        a, b = groups[0][0], groups[1][0]
        return (2, a, b, max(v for v in vals if v not in (a, b)))
    if groups[0][1] == 2:
        return (1, groups[0][0]) + tuple([v for v in kick if v != groups[0][0]][:3])
    return (0,) + tuple(kick[:5])


def _opp_ok(hand, board, level):
    """Подходит ли рука соперника под его действия.
    level 0: просто дошёл до флопа (играбельная рука)
    level 1: ставит/коллирует ставку (есть пара с картой на руке или дро)
    level 2: ставит крупно (топ-пара и выше или сильное дро)"""
    if level < 0:
        return True  # соперник играет почти любыми картами
    if not _playable(hand) and random.random() < 0.85:
        return False
    if level == 0 or not board:
        return True
    ev = evaluate(list(hand) + board)
    hv = [RV[c[0]] for c in hand]
    top = max(RV[c[0]] for c in board)
    made = ev[0] >= 2 or (ev[0] == 1 and (ev[1] in hv or hv[0] == hv[1]))
    if len(board) < 5:
        suits = Counter(c[1] for c in list(hand) + board)
        s, cnt = suits.most_common(1)[0]
        fd = cnt == 4 and any(c[1] == s for c in hand)
        vals = {RV[c[0]] for c in list(hand) + board}
        sd = sum(1 for v in range(2, 15) if v not in vals and _straight(list(vals) + [v])) >= 2
    else:
        fd = sd = False
    if level == 1:
        return made or fd or sd
    strong = ev[0] >= 2 or (ev[0] == 1 and ev[1] >= top and ev[1] in hv)
    return strong or fd


def equity(hole, board, n_opp, level=0, iters=1200):
    deck = [r + s for r in pt.RANKS for s in pt.SUITS if r + s not in hole and r + s not in board]
    need = 5 - len(board)
    win = 0.0
    done = 0
    for _ in range(iters):
        d = deck[:]
        random.shuffle(d)
        opps, i = [], 0
        for _o in range(n_opp):
            for _try in range(60):
                hnd = d[i:i + 2]
                i += 2
                if i + 2 > len(d) - need:
                    break
                lv = level if _o == 0 else min(level, 1)
                if _opp_ok(hnd, board, lv):
                    break
            opps.append(hnd)
        b = board + d[i:i + need]
        if len(b) < 5:
            continue
        me = evaluate(list(hole) + b)
        best = max(evaluate(o + b) for o in opps)
        win += 1 if me > best else (0.5 if me == best else 0)
        done += 1
    return win / max(1, done)


def describe(hole, board):
    """Что у вас сейчас: рука + дро, человеческим языком."""
    if not board:
        a, b = hole
        s = "одномастные" if a[1] == b[1] else "разномастные"
        if a[0] == b[0]:
            return f"Карманная пара {pt.RANK_SHOW.get(a[0], a[0])}"
        return f"{pt.card_str(a)} {pt.card_str(b)}, {s}"
    mine = evaluate(list(hole) + board)
    cat = mine[0]
    txt = CAT[cat]
    bvals = sorted({RV[c[0]] for c in board}, reverse=True)
    hv = [RV[c[0]] for c in hole]
    if len(board) == 5 and evaluate(board) == mine:
        txt = f"Играет доска ({txt.lower()})"
    elif cat == 1:
        p = mine[1]
        if p not in hv:
            txt = "Пара на доске (общая), у вас старшая карта"
        elif hv[0] == hv[1]:
            txt = "Оверпара" if p > bvals[0] else "Карманная пара ниже доски"
        elif p == bvals[0]:
            txt = "Топ-пара"
        elif len(bvals) > 1 and p == bvals[1]:
            txt = "Вторая пара"
        else:
            txt = "Слабая пара"
    elif cat == 2 and not (mine[1] in hv or mine[2] in hv):
        txt = "Две пары на доске (общие)"
    extra = []
    if len(board) < 5:
        suits = Counter(c[1] for c in list(hole) + board)
        s, n = suits.most_common(1)[0]
        if n == 4 and cat < 5 and any(c[1] == s for c in hole):
            extra.append("флеш-дро (9 аутов)")
        vals = set(RV[c[0]] for c in list(hole) + board)
        if 14 in vals:
            vals.add(1)
        if cat < 4:
            outs = 0
            for v in range(1, 15):
                if v not in vals and _straight(list(vals) + [v]):
                    outs += 1
            if outs >= 2:
                extra.append("стрит-дро (8 аутов)")
            elif outs == 1:
                extra.append("гатшот (4 аута)")
    return txt + (", " + ", ".join(extra) if extra else "")


# ---------------------------------------------------------------- совет
PREMIUM = {"AA", "KK", "QQ", "AKs", "AKo"}
STRONG = {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}


def _hand_key(hole):
    a, b = sorted(hole, key=lambda c: -RV[c[0]])
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def _playable(hole, pos=None):
    """Можно ли входить в банк с этой рукой (с учётом позиции)."""
    a, b = sorted((RV[c[0]] for c in hole), reverse=True)
    suited = hole[0][1] == hole[1][1]
    gap = a - b
    if a == b:
        return True
    if pos in ("btn", "late"):  # поздняя позиция: играем шире
        if a == 14 or (a == 13 and (b >= 8 or suited)) or (a == 12 and b >= 9) or (a == 11 and b >= 9):
            return True
        return (suited and gap <= 2 and b >= 4) or (not suited and gap == 1 and b >= 7)
    if pos == "early":  # ранняя: только хорошие руки
        if a == 14:
            return b >= 10 or (suited and b >= 9)
        if a >= 11 and b >= 11:
            return True
        return suited and gap == 1 and b >= 7
    if a == 14 and (suited or b >= 10):
        return True
    if a >= 10 and b >= 10:
        return True
    if suited and gap <= 2 and b >= 5:
        return True
    return not suited and gap == 1 and b >= 7


def _push_ok(hole, bbs, facing_short=False):
    """Таблица ол-ин/фолд для короткого стека (bbs = ваш стек в блайндах)."""
    a, b = sorted((RV[c[0]] for c in hole), reverse=True)
    suited = hole[0][1] == hole[1][1]
    wide = bbs <= 8 or facing_short
    if a == b:
        return True
    if a == 14:
        return wide or suited or b >= 7
    if a == 13:
        return b >= (7 if wide else 10) or (suited and b >= (2 if wide else 8))
    if a == 12:
        return b >= (9 if wide else 11) or (suited and b >= 9)
    return wide and suited and a - b == 1 and b >= 8


def _call_allin_ok(hole, bbs, opp_short):
    a, b = sorted((RV[c[0]] for c in hole), reverse=True)
    if a == b:
        return a >= (2 if opp_short else 7)
    if a == 14:
        return b >= (2 if opp_short else (7 if bbs <= 10 else 10))
    if a == 13:
        return b >= (10 if opp_short else 12)
    return opp_short and a == 12 and b >= 11


def advise(hole, board, n_opp, to_call, pot, bb=500, level_shift=0, my_stack=None, opp_short=False, pos=None):
    """Возвращает (шансы 0..1, действие, пояснение)."""
    to_call = to_call or 0
    if my_stack:
        to_call = min(to_call, my_stack)
    level = 0
    if board and to_call:
        level = 2 if (pot and to_call > 0.75 * pot) else 1
    level = max(-1, min(2, level + level_shift))
    eq = equity(list(hole), list(board), n_opp, level)
    bbs = my_stack / bb if my_stack and bb else None
    if not board and bbs is not None and bbs <= 15:
        if to_call <= bb:
            if _push_ok(hole, bbs, facing_short=pos in ("btn", "sb", "late")):
                return eq, "ОЛ-ИН", f"у вас всего ~{bbs:.0f} BB: с такой рукой лучше идти ва-банк"
            return eq, ("ЧЕК" if not to_call else "ФОЛД"), f"короткий стек (~{bbs:.0f} BB), ждём руку для ол-ина"
        if _call_allin_ok(hole, bbs, opp_short):
            return eq, "ОЛ-ИН / КОЛЛ", f"короткий стек (~{bbs:.0f} BB), рука достаточно сильная" + (", соперник короткий и пушит широко" if opp_short else "")
        return eq, "ФОЛД", f"короткий стек (~{bbs:.0f} BB), против рейза рука слабая"
    if not board and opp_short and to_call > 2 * bb and _call_allin_ok(hole, bbs or 50, True):
        return eq, "КОЛЛ", "короткий соперник пушит широко, ваша рука впереди его диапазона"
    if not board:
        k = _hand_key(hole)
        if k in PREMIUM:
            return eq, "РЕЙЗ", f"сильнейшая рука; на рейз соперника можно переповысить или идти до конца"
        if k in STRONG:
            if to_call and to_call > 8 * bb:
                return eq, "ФОЛД", "рейз слишком крупный для этой руки"
            return eq, ("РЕЙЗ ~3 BB" if (to_call or 0) <= bb else "КОЛЛ"), "сильная рука"
        if pos == "bb" and not to_call:
            return eq, "ЧЕК", "вы на большом блайнде, флоп бесплатно"
        if _playable(hole, pos):
            if not to_call:
                return eq, "ЧЕК", "бесплатно смотрим флоп"
            if to_call <= bb and pos in ("btn", "late"):
                return eq, "РЕЙЗ ~3 BB", "поздняя позиция: если до вас все сбросили, забираем блайнды; если были коллы - просто КОЛЛ"
            if to_call <= bb:
                return eq, "КОЛЛ", "дёшево, рука играбельная" + (" (ранняя позиция, на рейз - фолд)" if pos == "early" else "")
            if to_call <= 3 * bb and (hole[0][0] == hole[1][0] or hole[0][1] == hole[1][1]):
                return eq, "КОЛЛ", "небольшой рейз, играем на сет/флеш"
            return eq, "ФОЛД", "дорого для этой руки"
        return eq, ("ЧЕК" if not to_call else "ФОЛД"), "слабая рука" + (" для ранней позиции" if pos == "early" and _playable(hole) else "")
    # после флопа
    adj = eq
    if not to_call:
        p = pot or 0
        if eq >= 0.8:
            size = f" ~{int(p * 0.6):,}".replace(",", " ") if p else ""
            return eq, "СТАВКА" + size, "рука, скорее всего, лучшая: берём деньги"
        if eq >= 0.62 or (eq >= 0.5 and n_opp == 1 and len(board) < 5):
            size = f" ~{int(p * 0.35):,}".replace(",", " ") if p else ""
            return eq, "МАЛЕНЬКАЯ СТАВКА" + size, "вы, вероятно, впереди, но рука не железная"
        return eq, "ЧЕК", "ставить нечем, смотрим карту бесплатно" if len(board) < 5 else "лучше вскрыться бесплатно"
    pot_total = (pot or 0) + 2 * to_call
    need = to_call / pot_total if pot_total else 0.5
    big = pot and to_call > 0.75 * pot
    margin = 0.03
    if eq >= 0.8:
        return eq, "РЕЙЗ", f"очень сильная рука (нужно {need:.0%}, у вас ~{eq:.0%})"
    if adj - margin >= need:
        return eq, "КОЛЛ", f"цена нормальная: нужно {need:.0%}, у вас ~{eq:.0%}"
    return eq, "ФОЛД", f"дорого: нужно {need:.0%}, у вас ~{eq:.0%}" + (" и ставка крупная" if big else "")
