import numpy as np
import pytest

import hints
import overlay
import players
import poker_tracker as pt
from conftest import seat

MY_SEAT = 4
IMG = np.zeros((1439, 2559, 3), np.uint8)


def my_cards(c1, c2):
    """Две открытые карты прямо над вашей панелью (как их отдаёт read_screen)."""
    sx, sy = players.SLOTS[MY_SEAT]
    x, y = (sx + 167) / 2559, (sy - 93) / 1439
    return [(c1, x, y), (c2, x + 0.01, y)]


class FakeTable:
    def __init__(self):
        self.board = []
        self.mine = []
        self.seats = []
        self.to_call = 0
        self.advise_calls = []


@pytest.fixture
def table(monkeypatch):
    t = FakeTable()

    def fake_advise(*args, **kwargs):
        t.advise_calls.append(kwargs)
        return 0.5, "ЧЕК", "ok"

    monkeypatch.setattr(pt, "read_screen", lambda img: (list(t.board), list(t.mine)))
    monkeypatch.setattr(players, "read_seats", lambda img: [dict(s) for s in t.seats])
    monkeypatch.setattr(players, "find_dealer", lambda img: None)
    monkeypatch.setattr(hints, "read_buttons", lambda img: (True, t.to_call))
    monkeypatch.setattr(hints, "read_pot", lambda img: None)
    monkeypatch.setattr(hints, "describe", lambda hole, board: "")
    monkeypatch.setattr(hints, "advise", fake_advise)
    return t


@pytest.fixture
def engine(tmp_path, table):
    e = overlay.HintEngine()
    e.prof = players.Profiles(str(tmp_path / "players.json"))
    return e


def test_single_misread_frame_does_not_start_new_hand(engine, table):
    table.seats = [seat(MY_SEAT, me=True), seat(0, "#A")]
    table.mine = my_cards("Ah", "Kd")
    for _ in range(12):
        engine.step(IMG)
    assert engine.prof.hand_no == 1

    table.mine = my_cards("Qh", "Kd")  # один размытый кадр
    engine.step(IMG)
    table.mine = my_cards("Ah", "Kd")
    engine.step(IMG)
    assert engine.prof.hand_no == 1

    table.mine = my_cards("9s", "9h")  # настоящая новая раздача
    engine.step(IMG)
    engine.step(IMG)
    assert engine.prof.hand_no == 2


def test_bettor_is_the_latest_raiser_not_the_first_seat(engine, table):
    table.mine = my_cards("Ah", "Kd")
    table.to_call = 500
    table.seats = [seat(MY_SEAT, me=True), seat(0, "#A"), seat(1, "#B")]
    for _ in range(3):
        engine.step(IMG)

    table.board = ["2c", "7d", "9s"]
    # каждая надпись держится несколько кадров (новая засчитывается со 2-го кадра)
    table.seats = [seat(MY_SEAT, me=True), seat(0, "#A", "bet"), seat(1, "#B")]
    engine.step(IMG)
    engine.step(IMG)
    table.seats = [seat(MY_SEAT, me=True), seat(0, "#A", "bet"), seat(1, "#B", "raise")]
    engine.step(IMG)
    engine.step(IMG)

    # простая ставка незнакомого игрока ослабляет его диапазон (shift -1), рейз - нет (shift 0):
    # совет должен опираться на рейз #B, а не на ставку #A, которая сидит на месте с меньшим номером
    assert table.advise_calls[-1]["level_shift"] == 0


def test_new_cards_must_be_read_on_consecutive_frames(engine, table):
    table.seats = [seat(MY_SEAT, me=True), seat(0, "#A")]
    table.mine = my_cards("Ah", "Kd")
    for _ in range(12):
        engine.step(IMG)

    for mine in (my_cards("Qh", "Kd"), [], [], my_cards("Qh", "Kd")):  # между ними кадры без карт
        table.mine = mine
        engine.step(IMG)

    assert engine.prof.hand_no == 1
