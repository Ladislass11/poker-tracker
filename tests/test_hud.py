import pytest

import hints
import hud


def hint(**over):
    h = dict(hole=("Ah", "Kd"), board=["2c", "7d", "9s"], eq=0.38, act="КОЛЛ",
             why="цена нормальная: нужно 25%, у вас ~38%", n_opp=2, to_call=500, pot=1000,
             desc="Старшая карта, дро на стрит", opps=["#180104 (~25 р.): тайтовый"],
             my_stack=10000, bb=50, pos="middle", my_turn=True, unread=False)
    h.update(over)
    return h


def test_price_is_call_share_of_final_pot():
    assert hints.price(500, 1000) == pytest.approx(0.25)
    assert hints.price(500, None) == pytest.approx(0.5)


@pytest.mark.parametrize("act, tone", [
    ("ФОЛД", "fold"), ("ЧЕК", "check"), ("КОЛЛ", "call"),
    ("РЕЙЗ ~4 BB", "bet"), ("РЕЙЗ (×3) или ОЛ-ИН", "bet"), ("ОЛ-ИН / КОЛЛ", "bet"),
    ("СТАВКА ~750", "bet"), ("МАЛЕНЬКАЯ СТАВКА ~350", "bet"), ("СТАВКА-БЛЕФ ~500", "bluff"),
    ("что-то новое", "check"),
])
def test_action_tone_comes_from_first_word(act, tone):
    assert hud.build_view(hint(act=act))["tone"] == tone


def test_idle_view_shows_empty_table_and_hand_count():
    v = hud.build_view(None, hands_seen=12)
    assert v["idle"] and not v["active"]
    assert v["hole"] == [None, None] and v["board"] == [None] * 5
    assert "12" in v["why"]


def test_cards_are_split_into_rank_suit_and_colour():
    v = hud.build_view(hint(hole=("Th", "9s")))
    assert v["hole"] == [("10", "♥", True), ("9", "♠", False)]
    assert v["board"][:3] == [("2", "♣", False), ("7", "♦", True), ("9", "♠", False)]
    assert v["board"][3:] == [None, None]


def test_need_is_shown_only_after_flop_with_known_pot_and_a_bet():
    assert hud.build_view(hint())["need"] == pytest.approx(0.25)
    assert hud.build_view(hint(board=[], pot=None))["need"] is None
    assert hud.build_view(hint(pot=None))["need"] is None
    assert hud.build_view(hint(to_call=0))["need"] is None


def test_facts_show_call_share_position_and_stack_in_blinds():
    facts = dict(hud.build_view(hint(to_call=1200))["facts"])
    assert facts == {"Колл": "1 200, 12% стека", "Позиция": "средняя",
                     "Стек": "10 000 (200 BB)"}


def test_tiny_call_share_is_not_shown():
    assert dict(hud.build_view(hint(to_call=50))["facts"])["Колл"] == "50"


@pytest.mark.parametrize("hole, desc, shown", [
    (("Ah", "Kd"), "A♥ K♦, разномастные", "Разномастные"),
    (("Ah", "Kh"), "A♥ K♥, одномастные", "Одномастные"),
    (("Qh", "Qd"), "Карманная пара Q", "Карманная пара Q"),
])
def test_preflop_description_does_not_repeat_the_drawn_cards(hole, desc, shown):
    assert hud.build_view(hint(hole=hole, board=[], pot=None, desc=desc))["desc"] == shown


def test_stack_without_known_blind_and_unknown_position_do_not_crash():
    facts = dict(hud.build_view(hint(bb=0, pos="utg"))["facts"])
    assert facts["Стек"] == "10 000" and facts["Позиция"] == "utg"


def test_facts_skip_unknown_values():
    facts = dict(hud.build_view(hint(to_call=0, pos=None, my_stack=None))["facts"])
    assert facts == {}


def test_status_tells_whose_turn_and_unread_amount():
    assert hud.build_view(hint())["status"] == "ваш ход"
    assert hud.build_view(hint(my_turn=False))["status"] == "ещё не ваш ход"
    assert hud.build_view(hint(unread=True))["status"] == "сумму колла не прочитал"
    assert not hud.build_view(hint(my_turn=False))["active"]


@pytest.fixture
def root():
    tk = pytest.importorskip("tkinter")
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("нет экрана")
    yield r
    r.destroy()


def test_hud_draws_every_state_without_errors(root):
    h = hud.Hud(root, "версия тест")
    for view in (hud.build_view(None), hud.build_view(hint()),
                 hud.build_view(hint(board=[], pot=None, my_turn=False)),
                 hud.build_view(hint(act="СТАВКА-БЛЕФ ~500", to_call=0, why=""))):
        h.show(view)
        root.update_idletasks()
        assert h.canvas.winfo_reqheight() > 100
    h.show_error("ошибка кадра: тест")
    root.update_idletasks()


def test_error_is_shown_even_if_the_hint_itself_cannot_be_drawn(root):
    h = hud.Hud(root, "версия тест")
    broken = dict(hud.build_view(hint()), eq="не число")
    with pytest.raises(ValueError):
        h.show(broken)
    h.show_error("ошибка кадра: тест")  # окно не должно зависнуть на сломанной подсказке
    texts = [h.canvas.itemcget(i, "text") for i in h.canvas.find_all() if h.canvas.type(i) == "text"]
    assert "ошибка кадра: тест" in texts
