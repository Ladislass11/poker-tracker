import csv

from conftest import next_hand, seat

FLOP = ["2c", "7d", "9s"]


def warm_up(prof, ids):
    """Номер игрока засчитывается месту после 3 прочтений из последних 5."""
    for _ in range(3):
        prof.update([seat(i, pid, in_hand=False) for i, pid in ids], 0)


def show_rows(tmp_path):
    path = tmp_path / "players.json.log.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["action"] == "show"]


def play_tight_hand(prof, k):
    """#A входит в 1 раздачу из 8 коллом, на флопе против ставки #B чаще сбрасывает."""
    next_hand(prof)
    a_plays = k % 8 == 0
    prof.update([seat(0, "#A", "call" if a_plays else None), seat(1, "#B", "raise")], 0)
    if not a_plays:
        return
    prof.update([seat(0, "#A"), seat(1, "#B", "bet")], 3)
    prof.update([seat(0, "#A", "call" if k % 16 == 0 else "fold"), seat(1, "#B", "bet")], 3)


def test_tight_passive_bot_gets_style_and_fold_to_bet_despite_decay(prof):
    warm_up(prof, [(0, "#A"), (1, "#B")])
    for k in range(320):
        play_tight_hand(prof, k)

    assert prof.style("#A") == "passive"
    fvb = prof.stats("#A")["fvb"]
    assert fvb is not None and 0.3 <= fvb <= 0.7
    assert "пассивный" in prof.summary("#A")


def test_old_action_is_forgotten_over_time(prof):
    warm_up(prof, [(0, "#A")])
    next_hand(prof)
    prof.update([seat(0, "#A", "call")], 0)
    for _ in range(200):
        next_hand(prof)
        prof.update([seat(0, "#A")], 0)

    assert prof.get("#A")["call"] < 0.01


def test_summary_does_not_round_count_up_to_threshold(prof):
    prof.get("#A")["hands"] = 9.7

    assert "(9 р.)" in prof.summary("#A")


def test_raise_counts_hand_even_if_card_backs_were_not_seen(prof):
    warm_up(prof, [(0, "#A")])
    for _ in range(30):
        next_hand(prof)
        prof.update([seat(0, "#A", "raise", in_hand=False)], 0)

    d = prof.get("#A")
    assert d["hands"] > 0
    assert d["vpip"] <= d["hands"] + 1e-9
    assert d["pfr"] <= d["hands"] + 1e-9


def test_allin_showdown_is_saved_once_with_full_board(prof, tmp_path):
    prof.seat_id[2] = "#S"
    next_hand(prof)
    for board in (FLOP, FLOP + ["Jh"], FLOP + ["Jh", "3s"]):
        prof.add_show(2, ["Kh", "Ad"], board)

    assert prof.get("#S")["shows"] == ["Ad Kh | 2c 7d 9s Jh 3s"]
    assert [r["info"] for r in show_rows(tmp_path)] == ["Ad Kh | 2c 7d 9s Jh 3s"]


def test_showdown_without_river_is_logged_once_at_next_hand(prof, tmp_path):
    prof.seat_id[2] = "#S"
    next_hand(prof)
    hand = prof.hand_no
    prof.add_show(2, ["Kh", "Ad"], FLOP)
    prof.add_show(2, ["Kh", "Ad"], FLOP + ["Jh"])
    assert show_rows(tmp_path) == []

    next_hand(prof)

    rows = show_rows(tmp_path)
    assert [(r["hand"], r["info"]) for r in rows] == [(str(hand), "Ad Kh | 2c 7d 9s Jh")]


def test_showdowns_from_different_hands_are_kept(prof):
    prof.seat_id[2] = "#S"
    next_hand(prof)
    prof.add_show(2, ["Kh", "Ad"], FLOP + ["Jh", "3s"])
    next_hand(prof)
    prof.add_show(2, ["7h", "7c"], FLOP + ["Jh", "3s"])

    assert prof.get("#S")["shows"] == ["Ad Kh | 2c 7d 9s Jh 3s", "7c 7h | 2c 7d 9s Jh 3s"]


def test_profiles_saved_before_lifetime_counters_still_load(tmp_path):
    import json

    import players

    path = tmp_path / "players.json"
    path.write_text(json.dumps({"#A": {"hands": 30, "bet": 5, "raise": 4, "call": 1, "faced": 6,
                                       "fold_vs_bet": 3}}), encoding="utf-8")
    prof = players.Profiles(str(path))

    assert prof.style("#A") == "aggr"
    assert prof.stats("#A")["fvb"] == 0.5


def test_pure_aggressor_is_not_called_passive_after_quiet_stretch(prof):
    d = prof.get("#A")
    d.update({"hands": 20.0, "raise": 0.5, "seen_acts": 12})  # рейзил давно, коллов не было

    assert prof.style("#A") is None  # свежих действий почти нет - судить рано
    d["raise"] = 1.5
    assert prof.style("#A") == "aggr"  # только рейзы и ни одного колла


def test_one_misread_label_does_not_steal_the_aggressor(prof):
    next_hand(prof)
    frames = [("raise", None), ("raise", None), ("raise", "raise"), ("raise", "raise"),
              ("call", "raise"),  # надпись #A на одном кадре прочиталась неверно
              ("raise", "raise")]
    for a, b in frames:
        prof.update([seat(0, "#A", a), seat(1, "#B", b)], 3)

    assert prof.last_aggressor == 1


def test_saving_mid_hand_does_not_log_showdown_twice(prof, tmp_path):
    prof.seat_id[2] = "#S"
    next_hand(prof)
    prof.add_show(2, ["Kh", "Ad"], FLOP)
    prof.save()
    prof.add_show(2, ["Kh", "Ad"], FLOP + ["Jh", "3s"])

    assert [r["info"] for r in show_rows(tmp_path)] == ["Ad Kh | 2c 7d 9s Jh 3s"]


def test_closing_logs_pending_showdown(prof, tmp_path):
    prof.seat_id[2] = "#S"
    next_hand(prof)
    prof.add_show(2, ["Kh", "Ad"], FLOP + ["Jh"])

    prof.close()

    assert [r["info"] for r in show_rows(tmp_path)] == ["Ad Kh | 2c 7d 9s Jh"]
