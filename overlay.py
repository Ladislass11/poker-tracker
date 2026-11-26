# -*- coding: utf-8 -*-
"""Окно-подсказка поверх игры + запись раздач."""
import time
from collections import Counter

import poker_tracker as pt
import hints
import players

COLORS = {"ФОЛД": "#ff6b6b", "ЧЕК": "#cfd8dc", "КОЛЛ": "#ffd166", "СТАВКА": "#06d6a0",
          "МАЛЕНЬКАЯ": "#06d6a0", "РЕЙЗ": "#06d6a0", "ОЛ-ИН": "#06d6a0", "СТАВКА-БЛЕФ": "#c792ea"}


class HintEngine:
    def __init__(self):
        self.last_key = None
        self.last = None
        self.bb_seen = Counter()
        self.prof = players.Profiles()
        self.cur_hole = None

    def bb(self):
        # большой блайнд = самая частая сумма колла до флопа (нужно хотя бы 3 раза)
        common = [a for a, n in self.bb_seen.most_common() if n >= 3]
        return common[0] if common else 500

    @staticmethod
    def _seat_xy(i, W, H):
        sx, sy = players.SLOTS[i]
        return (sx + 167) / 2559, (sy + 20) / 1439

    def step(self, img):
        H, W = img.shape[:2]
        board, other = pt.read_screen(img)
        seats = players.read_seats(img)
        me = next((s for s in seats if s["me"]), None)
        # ваши карты лежат прямо над вашей панелью (у соперников на вскрытии - над их панелями)
        mine = []
        if me:
            sx, sy = players.SLOTS[me["seat"]]
            mx, my = (sx + 167) / 2559, (sy - 93) / 1439
            mine = [o for o in other if abs(o[1] - mx) < 0.06 and abs(o[2] - my) < 0.07]
        hole = tuple(sorted(o[0] for o in mine)) if len(mine) == 2 else None
        others = [o for o in other if o not in mine]
        if hole and hole != self.cur_hole and not board:
            self.cur_hole = hole
            self.prof.new_hand()
        self.prof.update(seats, len(board))
        # вскрытие: карты соперников запоминаем в их профиль
        if len(board) == 5 and len(others) >= 2:
            for s in seats:
                if s["me"]:
                    continue
                px, py = players.SLOTS[s["seat"]]
                cx, cy = (px + 167) / 2559, (py - 93) / 1439
                near = [o for o in others if abs(o[1] - cx) < 0.06 and abs(o[2] - cy) < 0.07]
                if len(near) == 2:
                    self.prof.add_show(s["seat"], [o[0] for o in near], board)
        my_turn, to_call = hints.read_buttons(img)
        if not hole or to_call == "none":
            return None  # вы не в раздаче или идёт вскрытие
        unread = to_call is None
        if unread:
            to_call = self.last["to_call"] if self.last and self.last.get("hole") == hole else 0
        if not board and to_call:
            self.bb_seen[to_call] += 1
        pot = hints.read_pot(img) if board else None
        in_hand = [s for s in seats if not s["me"] and s["in_hand"]]
        n_opp = len(in_hand) or hints.count_opponents(img)
        my_stack = me["stack"] if me else None
        # кто поставил: последний с действием "ставка/рейз"
        bettor = next((s for s in in_hand if s["status"] in ("bet", "raise")), None)
        bettor_id = self.prof.seat_id.get(bettor["seat"]) if bettor else None
        style = self.prof.style(bettor_id)
        shift = {"aggr": -1, "passive": 1}.get(style, 0) if to_call else 0
        bb = self.bb()
        opp_short = bool(bettor and bettor["stack"] is not None and to_call
                         and bettor["stack"] < bb * 2 and to_call <= 15 * bb)  # пошёл ол-ин с коротким стеком
        pos = players.my_position(seats, players.find_dealer(img))
        key = (hole, tuple(board), to_call, pot, n_opp, my_stack, shift, opp_short, pos)
        if key != self.last_key:
            eq, act, why = hints.advise(list(hole), list(board), n_opp, to_call, pot, bb,
                                        level_shift=shift, my_stack=my_stack, opp_short=opp_short, pos=pos)
            # блеф против тех, кто часто сбрасывает на ставку
            if board and not to_call and act == "ЧЕК" and in_hand:
                fv = [self.prof.stats(self.prof.seat_id.get(s["seat"]))["fvb"] if self.prof.seat_id.get(s["seat"]) else None
                      for s in in_hand]
                if all(f is not None and f >= 0.6 for f in fv) and len(in_hand) <= 2:
                    size = f" ~{int((pot or 0) * 0.5):,}".replace(",", " ") if pot else ""
                    act, why = "СТАВКА-БЛЕФ" + size, "эти соперники часто сбрасывают на ставку"
            if style == "aggr" and to_call:
                why += " · ставит агрессивный игрок, часто блефует"
            elif style == "passive" and to_call:
                why += " · ставит пассивный игрок - обычно сильная рука"
            risk = ""
            if my_stack and to_call:
                part = min(1.0, to_call / my_stack)
                risk = f"колл = {part:.0%} вашего стека"
            opps = [self.prof.summary(self.prof.seat_id[s["seat"]]) for s in in_hand
                    if self.prof.seat_id.get(s["seat"]) not in (None, "me")][:3]
            self.last = dict(hole=hole, board=board, eq=eq, act=act, why=why, n_opp=n_opp, to_call=to_call,
                             pot=pot, desc=hints.describe(list(hole), list(board)), risk=risk, opps=opps,
                             my_stack=my_stack, bb=bb, pos=pos)
            self.last_key = key
        self.last["my_turn"] = my_turn
        self.last["unread"] = unread
        return self.last


def run(monitor, fps, record=True):
    import mss
    try:
        import tkinter as tk
    except ImportError:
        tk = None
    writer = pt.CsvWriter(pt.CSV_PATH) if record else None
    tracker = pt.Tracker(writer) if record else None
    engine = HintEngine()
    sct = mss.mss()

    def grab():
        return pt.grab_screen(sct, monitor)

    if tk is None:
        print("tkinter не найден - подсказки будут в этом окне.")
        last = None
        try:
            while True:
                img = grab()
                if tracker:
                    tracker.feed(img)
                h = engine.step(img)
                if h and h is not last:
                    print(f"  >>> {h['act']}  ({h['eq']:.0%})  {h['desc']}  - {h['why']}", flush=True)
                    last = h
                time.sleep(1 / fps)
        except KeyboardInterrupt:
            if tracker:
                tracker.finish()
        return

    root = tk.Tk()
    root.title("Подсказка")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.88)
    except Exception:
        pass
    root.configure(bg="#111")
    root.geometry("+20+20")
    big = tk.Label(root, text="Жду раздачу...", font=("Segoe UI", 20, "bold"), fg="#fff", bg="#111", anchor="w")
    big.pack(fill="x", padx=12, pady=(8, 0))
    mid = tk.Label(root, text="", font=("Segoe UI", 12), fg="#ddd", bg="#111", anchor="w", justify="left")
    mid.pack(fill="x", padx=12)
    small = tk.Label(root, text="перетащите мышью · правый клик - закрыть", font=("Segoe UI", 9),
                     fg="#777", bg="#111", anchor="w")
    small.pack(fill="x", padx=12, pady=(0, 8))

    drag = {}

    def start(e):
        drag["x"], drag["y"] = e.x, e.y

    def move(e):
        root.geometry(f"+{e.x_root - drag['x']}+{e.y_root - drag['y']}")

    def close(_e=None):
        if tracker:
            tracker.finish()
        root.destroy()

    for w in (root, big, mid, small):
        w.bind("<Button-1>", start)
        w.bind("<B1-Motion>", move)
        w.bind("<Button-3>", close)

    def tick():
        try:
            img = grab()
            if tracker:
                tracker.feed(img)
            h = engine.step(img)
            if h is None:
                big.config(text="—", fg="#777")
                mid.config(text="нет ваших карт на экране")
            else:
                col = COLORS.get(h["act"].split()[0], "#fff")
                wait = "" if h["my_turn"] else "   (пока не ваш ход)"
                if h.get("unread"):
                    wait += "   (?) сумму на кнопке не прочитал"
                big.config(text=f"{h['act']}{wait}", fg=col)
                cards = " ".join(pt.card_str(c) for c in h["hole"])
                board = " ".join(pt.card_str(c) for c in h["board"]) or "до флопа"
                info = f"{cards}  |  {board}\n{h['desc']}\nШанс выиграть ~{h['eq']:.0%} против {h['n_opp']} соп."
                if h["to_call"]:
                    info += f"  ·  колл {h['to_call']:,}".replace(",", " ")
                if h.get("pos"):
                    info += f"\nПозиция: {players.POS_NAME[h['pos']]}"
                if h["my_stack"]:
                    info += f"\nВаш стек {h['my_stack']:,} (~{h['my_stack'] / h['bb']:.0f} BB)".replace(",", " ")
                    if h["risk"]:
                        info += f"  ·  {h['risk']}"
                info += f"\n{h['why']}"
                if h["opps"]:
                    info += "\n" + "\n".join(h["opps"])
                mid.config(text=info)
        except Exception as e:  # не падаем из-за одного плохого кадра
            mid.config(text=f"ошибка кадра: {e}")
        root.after(int(1000 / fps), tick)

    print("Подсказка запущена: окно в левом верхнем углу экрана.")
    print("Перетащите его мышью куда удобно. Закрыть: правый клик по окну.")
    root.after(300, tick)
    root.mainloop()
    engine.prof.save()
    if tracker:
        print(f"Записано раздач: {tracker.hand_no}")
