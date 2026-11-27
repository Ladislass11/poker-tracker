import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import players  # noqa: E402


def seat(i, pid=None, status=None, in_hand=True, stack=10000, me=False):
    """Панель игрока в том виде, в каком её возвращает players.read_seat."""
    return {"seat": i, "id": pid, "me": me, "status": status, "stack": stack, "in_hand": in_hand}


def next_hand(prof):
    """Новая раздача в обход защиты от двойного срабатывания (она смотрит на число кадров)."""
    prof._ticks = getattr(prof, "_ticks", 0) + 20
    prof.new_hand()


@pytest.fixture
def prof(tmp_path):
    return players.Profiles(str(tmp_path / "players.json"))
