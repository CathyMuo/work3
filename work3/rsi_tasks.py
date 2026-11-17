"""Deterministic read-only tasks for the paired RSI feasibility probe."""

from __future__ import annotations

import random
from dataclasses import dataclass

FAMILIES = ("single", "order", "bundle", "reserved")
SPLITS = {"development": ("single", "order"), "heldout": ("bundle", "reserved")}


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    control: bool
    prompt: str
    worlds: tuple[dict, dict]
    oracle: tuple[str, ...]
    changed_query: str
    answers: tuple[str, str]

    def sufficient(self, world_index: int, queries: tuple[str, ...]) -> bool:
        if set(self.oracle) <= set(queries):
            return True
        if self.family != "bundle" or self.oracle[0] not in queries:
            return False
        world = self.worlds[world_index]
        items = world[self.oracle[0]]["items"]
        return any(f"stock {item['sku']}" in queries
                   and world[f"stock {item['sku']}"]["on_hand"] < item["quantity"]
                   for item in items)


def make_case(family: str, seed: int, index: int, control: bool = False) -> Case:
    if family not in FAMILIES:
        raise ValueError(f"Unknown task family: {family}")
    rng = random.Random(f"{seed}:{family}:{index}:{control}")
    order, sku, other, distractor = rng.sample(range(100000, 999999), 4)
    order = f"ORD-{order}"
    sku, other, distractor = (f"SKU-{i}" for i in (sku, other, distractor))
    stock, extra = f"stock {sku}", f"stock {other}"
    order_query = f"order {order}"
    quantity = rng.randint(3, 15)
    second = rng.randint(3, 15)
    reserve = rng.randint(1, 5) if family == "reserved" else 0
    high = quantity + reserve + rng.randrange(3)
    low = quantity + reserve - rng.randint(1, 2)
    stocks = (high, low) if rng.randrange(2) else (low, high)
    world = {f"stock {distractor}": {"on_hand": rng.randint(1, 25), "reserved": 0}}
    if family == "single":
        question = (f"The order needs {quantity} units of {sku}. Route to warehouse A if its "
                    "on-hand stock covers the order, otherwise warehouse B.")
        oracle = (stock,)
    elif family == "bundle":
        world[order_query] = {"items": [{"sku": sku, "quantity": quantity},
                                        {"sku": other, "quantity": second}]}
        world[extra] = {"on_hand": second + rng.randint(0, 4), "reserved": 0}
        question = (f"Route order {order} to warehouse A if on-hand stock covers every item "
                    "in its bundle. Otherwise route the entire order to warehouse B.")
        oracle = (order_query, stock, extra)
    else:
        world[order_query] = {"sku": sku, "quantity": quantity}
        rule = ("available stock (on-hand minus reserved)" if family == "reserved"
                else "on-hand stock")
        question = (f"Route order {order} to warehouse A if {rule} covers its required quantity. "
                    "Otherwise route it to warehouse B.")
        oracle = (order_query, stock)
    worlds = tuple({**world, stock: {"on_hand": n, "reserved": reserve}} for n in stocks)
    def answer(records: dict) -> str:
        if family == "bundle":
            enough = all(records[f"stock {item['sku']}"]["on_hand"] >= item["quantity"]
                         for item in records[order_query]["items"])
        else:
            needed = quantity if family == "single" else records[order_query]["quantity"]
            available = records[stock]["on_hand"]
            if family == "reserved":
                available -= records[stock]["reserved"]
            enough = available >= needed
        return "A" if enough else "B"

    answers = tuple(answer(records) for records in worlds)
    if control:
        override = ("A", "B")[rng.randrange(2)]
        question += (f" A public dispatch override applies to this order: route to warehouse "
                     f"{override} regardless of stock or item availability.")
        answers = (override, override)
        oracle = ()
    return Case(f"{family}-{index}-{int(control)}", family, control, question, worlds,
                oracle, stock, answers)


def make_cases(split: str, count: int, controls: int, seed: int) -> list[Case]:
    return [make_case(family, seed, i, control)
            for family in SPLITS[split] for control, n in ((False, count), (True, controls))
            for i in range(n)]
