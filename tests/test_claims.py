from dataclasses import replace

from fakes import GPUS

from gpu_orc.claims import CardRoom, Claim, Reservations, plan_eviction, side_model_claim, with_reservations
from gpu_orc.model import Holder

A, B, C = (gpu.uuid for gpu in GPUS)
_, CARD_B, CARD_C = GPUS


def idle(consumer: str, item: str, mib: int) -> Claim:
    return Claim(consumer, item, mib, mib, False)


def test_a_side_model_claims_the_larger_of_memory_and_need() -> None:
    starting = Holder("tts", "qwen", frozenset({C}), memory_mib={C: 800}, need_mib=6024)
    grown = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 4100}, need_mib=3712)
    blind = Holder("tts", "xtts", frozenset({C}), need_mib=3680)
    assert side_model_claim(starting, CARD_C) == Claim("tts", "qwen", 6024, 800, None)
    assert side_model_claim(grown, CARD_C) == Claim("stt", "gpu-worker", 4100, 4100, None)
    assert side_model_claim(blind, CARD_C) == Claim("tts", "xtts", 3680, 3680, None)


def test_a_reservation_raises_the_claim_of_its_holder_or_counts_alone() -> None:
    reservations = Reservations(lambda: 0.0)
    reservations.put("tts", "qwen", {C: 6024}, 60)
    reservations.put("stt", "gpu-worker", {C: 3712}, 60)
    merged = with_reservations([Claim("tts", "qwen", 900, 900, None)], reservations.active())
    assert merged == (Claim("tts", "qwen", 6024, 900, None), Claim("stt", "gpu-worker", 3712, 0, None))


def test_reservations_run_out_and_a_new_promise_replaces_the_old() -> None:
    now = [0.0]
    reservations = Reservations(lambda: now[0])
    reservations.put("ls", "big", {A: 40000, B: 30000}, 60)
    reservations.put("ls", "big", {A: 41000}, 30)
    assert [(r.card, r.need_mib) for r in reservations.active()] == [(A, 41000)]
    now[0] = 30.0
    assert reservations.active() == []


def test_eviction_takes_the_card_with_the_fewest_and_the_lowest_priority_first() -> None:
    rank = {"stt": 2, "tts": 1}.get
    on_c = (idle("stt", "w1", 15000), idle("stt", "w2", 15000), idle("ls", "big", 15000))
    on_b = (idle("tts", "qwen", 6000), idle("stt", "gpu-worker", 25000))
    two_small = CardRoom(replace(CARD_C, memory_used_mib=45000), on_c)        # both workers must go
    one_big = CardRoom(replace(CARD_B, memory_used_mib=31000), on_b)          # the worker alone makes room
    eviction = plan_eviction([two_small, one_big], 20000, 512, lambda claim: rank(claim.consumer))
    assert eviction is not None
    assert (eviction.room.gpu.index, [c.item for c in eviction.claims]) == (1, ["gpu-worker"])


def test_nothing_gives_way_without_a_rank_or_while_busy() -> None:
    room = CardRoom(CARD_C, (Claim("stt", "w", 40000, 40000, True), idle("ls", "big", 40000)))
    assert plan_eviction([room], 20000, 512, lambda claim: 2 if claim.consumer == "stt" else None) is None
