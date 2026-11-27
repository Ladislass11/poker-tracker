# -*- coding: utf-8 -*-
"""Окно подсказки: что показать (build_view) и как это нарисовать (Hud)."""
import hints
import players
import poker_tracker as pt

WIDTH = 360
PAD = 16
HOLE_CARD = (40, 56)   # ваши карты крупнее
BOARD_CARD = (34, 46)  # карты стола
FACT_KEY_W = 76        # ширина колонки «Колл / Позиция / Стек»

PALETTE = {
    "back": "#161D33",      # фон: тёмно-синий, как рубашка карты
    "rule": "#2E3960",      # линии и пустые места под карты
    "text": "#E3E7F1",
    "muted": "#8A94B2",
    "paper": "#ECE7DB",     # лицевая сторона карты
    "card_ink": "#1E2030",
    "card_red": "#C8312D",
    "track": "#262F4E",
    "fill": "#8D9AC4",
    "tick": "#F4F1EA",
    "error": "#FF6F6A",
}

# цвет действия: (текст, заливка верхней полосы, когда ваш ход)
TONES = {
    "fold": ("#FF6F6A", "#4C1F2B"),
    "call": ("#F4B942", "#57400E"),
    "bet": ("#4FD495", "#163F33"),
    "check": ("#B7C0D6", "#2A3352"),
    "bluff": ("#BC98F5", "#372A58"),
}

ACTION_TONE = {"ФОЛД": "fold", "ЧЕК": "check", "КОЛЛ": "call", "СТАВКА": "bet", "МАЛЕНЬКАЯ": "bet",
               "РЕЙЗ": "bet", "ОЛ-ИН": "bet", "СТАВКА-БЛЕФ": "bluff"}


# ---------------------------------------------------------------- что показать
def _num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _card(c: str) -> tuple:
    return pt.RANK_SHOW.get(c[0], c[0]), pt.SUIT_SYM[c[1]], c[1] in "hd"


def _status(h: dict) -> str:
    if h.get("unread"):
        return "сумму колла не прочитал"
    return "ваш ход" if h["my_turn"] else "ещё не ваш ход"


def _facts(h: dict, to_call: int) -> list:
    facts = []
    stack = h.get("my_stack")
    if to_call:
        share = f", {min(1.0, to_call / stack):.0%} стека" if stack and to_call / stack >= 0.01 else ""
        facts.append(("Колл", _num(to_call) + share))
    if h.get("pos"):
        facts.append(("Позиция", players.POS_NAME.get(h["pos"], h["pos"]).replace(" позиция", "")))
    if stack:
        in_bb = f" ({stack / h['bb']:.0f} BB)" if h.get("bb") else ""
        facts.append(("Стек", _num(stack) + in_bb))
    return facts


def _describe(h: dict) -> str:
    """До флопа карты уже нарисованы - пишем только масть; после флопа - что собралось."""
    (r1, s1), (r2, s2) = h["hole"]
    if h["board"] or r1 == r2:
        return h["desc"]
    return "Одномастные" if s1 == s2 else "Разномастные"


def build_view(h: dict | None, hands_seen: int = 0) -> dict:
    """Готовые для окна строки и значения. h - словарь из HintEngine.step (None - вас нет в раздаче)."""
    if h is None:
        return {"idle": True, "active": False, "tone": "check", "verdict": "Жду ваши карты", "status": "",
                "why": f"Наблюдаю за столом. Записано раздач: {hands_seen}",
                "hole": [None, None], "board": [None] * 5, "desc": "", "eq": None, "n_opp": 0,
                "need": None, "facts": [], "opps": []}
    board = list(h["board"])
    to_call = h["to_call"] or 0
    return {
        "idle": False,
        "active": bool(h["my_turn"]),
        "tone": ACTION_TONE.get(h["act"].split()[0], "check"),
        "verdict": h["act"],
        "status": _status(h),
        "why": h["why"],
        "hole": [_card(c) for c in h["hole"]],
        "board": [_card(c) for c in board] + [None] * (5 - len(board)),
        "desc": _describe(h),
        "eq": h["eq"],
        "n_opp": h["n_opp"],
        "need": hints.price(to_call, h["pot"]) if board and to_call and h["pot"] else None,
        "facts": _facts(h, to_call),
        "opps": list(h["opps"]),
    }


# ---------------------------------------------------------------- как нарисовать
def _mix(c1: str, c2: str, t: float) -> str:
    """Цвет между c1 и c2 (t=0 - c1, t=1 - c2)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _pick_fonts(root) -> dict:
    import tkinter.font as tkfont
    have = set(tkfont.families(root))

    def font(name, size, weight="normal"):
        return (name, size) if name in have else ("Segoe UI", size, weight)

    return {
        "verdict": font("Bahnschrift SemiBold Condensed", 24, "bold"),
        "strong": font("Bahnschrift SemiBold", 12, "bold"),
        "pct": font("Bahnschrift SemiBold", 15, "bold"),
        "body": font("Bahnschrift", 11),
        "small": font("Bahnschrift", 9),
        "rank_big": font("Bahnschrift SemiBold", 17, "bold"),
        "rank": font("Bahnschrift SemiBold", 14, "bold"),
        "suit_big": ("Segoe UI Symbol", 15),
        "suit": ("Segoe UI Symbol", 12),
    }


class Hud:
    def __init__(self, root, version: str):
        import tkinter as tk
        self.version = version[:1].upper() + version[1:]
        self.fonts = _pick_fonts(root)
        self.canvas = tk.Canvas(root, width=WIDTH, height=200, bg=PALETTE["back"], highlightthickness=0, bd=0)
        self.canvas.pack()
        self._view = None
        self._error = ""
        self.show(build_view(None))

    def show(self, view: dict):
        if view == self._view and not self._error:
            return
        self._view, self._error = view, ""
        self._draw()

    def show_error(self, message: str):
        """Ошибку видно всегда: если не рисуется сама подсказка, показываем её поверх пустого стола."""
        self._error = message
        try:
            self._draw()
        except Exception:
            self._view = build_view(None)
            self._draw()

    # ------------------------------------------------------------ части окна
    def _draw(self):
        v = self._view
        self.canvas.delete("all")
        y = self._draw_verdict(v)
        y = self._draw_cards(v, y + 14)
        if v["desc"]:
            y = self._text(PAD, y + 8, v["desc"], "strong", PALETTE["text"])
        if v["eq"] is not None:
            y = self._draw_odds(v, y + 14)
        if v["facts"]:
            y = self._draw_facts(v["facts"], y + 12)
        if v["opps"]:
            y = self._draw_opps(v["opps"], y + 12)
        y = self._draw_footer(y + 12)
        self.canvas.config(height=y + 10)

    def _text(self, x, y, text, font, color, width=WIDTH - 2 * PAD, anchor="nw"):
        item = self.canvas.create_text(x, y, text=text, font=self.fonts[font], fill=color,
                                       anchor=anchor, width=width)
        box = self.canvas.bbox(item)
        return box[3] if box else y

    def _draw_verdict(self, v):
        """Верхняя полоса: действие + почему. Залита цветом действия, только когда ваш ход."""
        bright, deep = TONES[v["tone"]]
        if v["idle"]:
            color = PALETTE["muted"]
        elif v["active"]:
            color = bright
        else:
            color = _mix(bright, PALETTE["back"], 0.45)
        band = self.canvas.create_rectangle(0, 0, WIDTH, 1, width=0,
                                            fill=deep if v["active"] else PALETTE["back"])
        status_w = 0
        if v["status"]:
            st_color = (TONES["call"][0] if v["status"].startswith("сумму")
                        else PALETTE["text"] if v["active"] else PALETTE["muted"])
            item = self.canvas.create_text(WIDTH - PAD, 22, text=v["status"], font=self.fonts["small"],
                                           fill=st_color, anchor="ne")
            x0, _, x1, _ = self.canvas.bbox(item)
            status_w = x1 - x0 + 12
        y = self._text(PAD, 10, v["verdict"], "verdict", color, width=WIDTH - 2 * PAD - status_w)
        y = self._text(PAD, y + 1, v["why"], "body", PALETTE["text"] if not v["idle"] else PALETTE["muted"])
        bottom = y + 12
        self.canvas.coords(band, 0, 0, WIDTH, bottom)
        if not v["active"]:
            self.canvas.create_line(PAD, bottom, WIDTH - PAD, bottom, fill=PALETTE["rule"])
        return bottom

    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
               x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _draw_card(self, x, y, size, card, big):
        w, h = size
        if card is None:  # карта ещё не открыта
            self._round_rect(x, y, x + w, y + h, 6, fill="", outline=PALETTE["rule"], width=1)
            return
        rank, suit, red = card
        ink = PALETTE["card_red"] if red else PALETTE["card_ink"]
        self._round_rect(x, y, x + w, y + h, 6, fill=PALETTE["paper"], outline="")
        cx = x + w / 2
        self.canvas.create_text(cx, y + h * 0.36, text=rank, fill=ink,
                                font=self.fonts["rank_big" if big else "rank"])
        self.canvas.create_text(cx, y + h * 0.74, text=suit, fill=ink,
                                font=self.fonts["suit_big" if big else "suit"])

    def _draw_cards(self, v, y):
        """Ваши две карты и пять мест стола: пустое место - карта ещё не открыта."""
        x = PAD
        for card in v["hole"]:
            self._draw_card(x, y, HOLE_CARD, card, big=True)
            x += HOLE_CARD[0] + 6
        x += 14
        board_y = y + HOLE_CARD[1] - BOARD_CARD[1]
        for card in v["board"]:
            self._draw_card(x, board_y, BOARD_CARD, card, big=False)
            x += BOARD_CARD[0] + 5
        return y + HOLE_CARD[1]

    def _draw_odds(self, v, y):
        """Шанс выиграть полоской; отметка - сколько нужно, чтобы колл окупился."""
        n = v["n_opp"]
        who = f"против {n} соперника" if n == 1 else f"против {n} соперников"
        self._text(PAD, y + 4, f"Шанс выиграть {who}", "body", PALETTE["muted"])
        bottom = self._text(WIDTH - PAD, y, f"{v['eq']:.0%}", "pct", PALETTE["text"], anchor="ne")
        top = bottom + 6
        left, right = PAD, WIDTH - PAD
        self.canvas.create_rectangle(left, top, right, top + 6, fill=PALETTE["track"], width=0)
        self.canvas.create_rectangle(left, top, left + (right - left) * v["eq"], top + 6,
                                     fill=PALETTE["fill"], width=0)
        end = top + 6
        if v["need"] is not None:
            tx = left + (right - left) * v["need"]
            self.canvas.create_line(tx, top - 4, tx, top + 10, fill=PALETTE["tick"], width=2)
            label = f"нужно {v['need']:.0%}"
            anchor = "nw" if tx < left + 40 else ("ne" if tx > right - 40 else "n")
            end = self._text(tx, top + 12, label, "small", PALETTE["tick"], anchor=anchor)
        return end

    def _draw_facts(self, facts, y):
        for key, value in facts:
            self._text(PAD, y, key, "body", PALETTE["muted"])
            y = self._text(PAD + FACT_KEY_W, y, value, "body", PALETTE["text"],
                           width=WIDTH - 2 * PAD - FACT_KEY_W) + 3
        return y

    def _draw_opps(self, opps, y):
        self.canvas.create_line(PAD, y, WIDTH - PAD, y, fill=PALETTE["rule"])
        y += 8
        for line in opps:
            y = self._text(PAD, y, line, "small", PALETTE["text"]) + 4
        return y

    def _draw_footer(self, y):
        if self._error:
            return self._text(PAD, y, self._error, "small", PALETTE["error"])
        return self._text(PAD, y, f"{self.version}. Левая кнопка двигает, правая закрывает.",
                          "small", PALETTE["muted"])
