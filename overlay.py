# -*- coding: utf-8 -*-
"""Окно-подсказка поверх игры + запись раздач."""
VERSION = "версия 16 (27.11)"
import time
from collections import Counter

import poker_tracker as pt
import hints
import players
import hud


class HintEngine:
    def __init__(self):
        self.last_key = None
        self.last = None
        self.bb_seen = Counter()
        self.prof = players.Profiles()
        self.cur_hole = None
        self.hole_cand = None  # новые карты, прочитанные пока только на одном кадре
        self.fixed_bb = None

    def bb(self):
        if self.fixed_bb:
            return self.fixed_bb
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
        # когда над вашей панелью жёлтая надпись ("Уравнял ставку"), слова "Вы" не видно -
        # берём ваше место из прошлых кадров
        if me:
            self.my_seat = me["seat"]
        elif getattr(self, "my_seat", None) is not None:
            me = next((s for s in seats if s["seat"] == self.my_seat), None)
            if me:
                me["me"] = True
        # ваши карты лежат прямо над вашей панелью (у соперников на вскрытии - над их панелями)
        mine = []
        if me:
            sx, sy = players.SLOTS[me["seat"]]
            mx, my = (sx + 167) / 2559, (sy - 93) / 1439
            mine = [o for o in other if abs(o[1] - mx) < 0.06 and abs(o[2] - my) < 0.07]
        hole = tuple(sorted(o[0] for o in mine)) if len(mine) == 2 else None
        others = [o for o in other if o not in mine]
        if hole and hole != self.cur_hole and not board:
            # новая раздача - только если новые карты прочитаны 2 кадра подряд:
            # один размытый кадр не должен засчитывать всем соперникам лишнюю раздачу
            if hole == self.hole_cand:
                self.cur_hole = hole
                self.hole_cand = None
                self.prof.new_hand()
                self.aggr_streets = set()
            else:
                self.hole_cand = hole
        else:
            self.hole_cand = None
        dealer = players.find_dealer(img)
        self.prof.update(seats, len(board), dealer)
        # вскрытие: карты соперников запоминаем в их профиль
        # на вскрытии часть карт стола затемняется и не читается - берём полный стол, запомненный раньше
        hand_no = getattr(self.prof, "hand_no", 0)
        if getattr(self, "_board_hand", None) != hand_no:
            self._board_hand, self.board5 = hand_no, None
        if len(board) >= len(self.board5 or []):
            self.board5 = list(board)
        show_board = board if len(board) >= len(self.board5 or []) else self.board5
        if show_board and len(show_board) >= 3 and len(others) >= 2:
            for s in seats:
                if s["me"]:
                    continue
                px, py = players.SLOTS[s["seat"]]
                cx, cy = (px + 167) / 2559, (py - 93) / 1439
                near = [o for o in others if abs(o[1] - cx) < 0.06 and abs(o[2] - cy) < 0.07]
                if len(near) == 2:
                    self.prof.add_show(s["seat"], [o[0] for o in near], show_board)
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
        # кто поставил: место, где надпись "ставка/рейз" появилась последней на этой улице;
        # если это не отследили - любой соперник с такой надписью
        agg = self.prof.last_aggressor
        bettor = next((s for s in in_hand if s["seat"] == agg and s["status"] in ("bet", "raise")), None)
        if bettor is None:
            bettor = next((s for s in in_hand if s["status"] in ("bet", "raise")), None)
        # запоминаем, что соперник ставил/повышал после флопа в этой раздаче
        if not hasattr(self, "aggr_streets"):
            self.aggr_streets = set()
        if bettor and board:
            self.aggr_streets.add(len(board))
        prior_aggr = any(st < len(board) for st in self.aggr_streets)
        bettor_id = self.prof.seat_id.get(bettor["seat"]) if bettor else None
        style = self.prof.style(bettor_id)
        shift = {"aggr": -1, "passive": 1}.get(style, 0) if to_call else 0
        # стол агрессивный: обычная ставка неизвестного игрока - ещё не сильная рука (рейз - уже да)
        if to_call and board and style is None and bettor and bettor["status"] == "bet":
            shift = -1
        bb = self.bb()
        # кто повысил до флопа: маньяк (рейзит почти всё) или тайтовый (только сильные руки)
        raiser = None
        if not board and bettor_id and bettor_id != "me":
            d = self.prof.get(bettor_id)
            n = d["hands"]
            if n >= 15:
                if d["pfr"] / n >= 0.35 or d["vpip"] / n >= 0.7:
                    raiser = "maniac"
                elif d["vpip"] / n <= 0.3 and d["pfr"] / n <= 0.2:
                    raiser = "tight"
        opp_short = bool(bettor and bettor["stack"] is not None and to_call
                         and bettor["stack"] < bb * 2 and to_call <= 15 * bb)  # пошёл ол-ин с коротким стеком
        pos = players.my_position(seats, dealer)
        key = (hole, tuple(board), to_call, pot, n_opp, my_stack, shift, opp_short, pos, prior_aggr, raiser)
        if key != self.last_key:
            eq, act, why = hints.advise(list(hole), list(board), n_opp, to_call, pot, bb,
                                        level_shift=shift, my_stack=my_stack, opp_short=opp_short, pos=pos,
                                        prior_aggr=prior_aggr, raiser=raiser)
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
            opps = [self.prof.summary(self.prof.seat_id[s["seat"]]) for s in in_hand
                    if self.prof.seat_id.get(s["seat"]) not in (None, "me")][:3]
            self.last = dict(hole=hole, board=board, eq=eq, act=act, why=why, n_opp=n_opp, to_call=to_call,
                             pot=pot, desc=hints.describe(list(hole), list(board)), opps=opps,
                             my_stack=my_stack, bb=bb, pos=pos)
            self.last_key = key
        self.last["my_turn"] = my_turn
        self.last["unread"] = unread
        return self.last


def run(monitor, fps, record=True, bb=None):
    import mss
    try:
        import tkinter as tk
    except ImportError:
        tk = None
    writer = pt.CsvWriter(pt.CSV_PATH) if record else None
    tracker = pt.Tracker(writer) if record else None
    engine = HintEngine()
    engine.fixed_bb = bb
    sct = mss.MSS() if hasattr(mss, 'MSS') else mss.mss()

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
            engine.prof.close()
        return

    root = tk.Tk()
    root.title("Подсказка")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.93)
    except Exception:
        pass
    root.configure(bg=hud.PALETTE["back"])
    root.geometry("+20+20")
    view = hud.Hud(root, VERSION)

    drag = {}

    def start(e):
        drag["x"], drag["y"] = e.x, e.y

    def move(e):
        root.geometry(f"+{e.x_root - drag['x']}+{e.y_root - drag['y']}")

    def close(_e=None):
        if tracker:
            tracker.finish()
        root.destroy()

    # привязка к окну срабатывает и для холста внутри него (один раз на клик)
    root.bind("<Button-1>", start)
    root.bind("<B1-Motion>", move)
    root.bind("<Button-3>", close)

    def tick():
        try:
            img = grab()
            if tracker:
                tracker.feed(img)
            h = engine.step(img)
            view.show(hud.build_view(h, hands_seen=getattr(engine.prof, "hand_no", 0)))
        except Exception as e:  # не падаем из-за одного плохого кадра
            try:
                view.show_error(f"ошибка кадра: {e}")
            except Exception:
                print(f"ошибка кадра: {e}", flush=True)
        finally:
            root.after(int(1000 / fps), tick)  # следующий кадр - даже если этот не нарисовался

    print(f"Подсказка запущена ({VERSION}): окно в левом верхнем углу экрана.")
    print("Перетащите его мышью куда удобно. Закрыть: правый клик по окну.")
    root.after(300, tick)
    root.mainloop()
    engine.prof.close()
    if tracker:
        print(f"Записано раздач: {tracker.hand_no}")
