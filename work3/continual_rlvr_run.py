"""Distributed worker for the frozen necessity and matched method experiments."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from time import monotonic

import torch
import torch.distributed as dist

from work3.continual_rlvr_data import load_bundle, sha256_file
from work3.continual_rlvr_engine import Engine, fixed_order, rollout_tokens
from work3.continual_rlvr_protocol import (
    ARMS, A_STEPS, BASE_EVAL, B_STEPS, GROUP, MAX_PREFIXES, METHOD_EVAL, PREFIX, PREFIX_SAMPLES,
    PRETRAIN_BANK, SEEDS, check_scores, coverage, method_verdict, necessity_verdict, pass_one,
)
from work3.wiki_rsi_progress import Progress, atomic_json

BANK_ARMS = {"trace_replay", "uniform_prefix", "guard"}


class Run:
    def __init__(self, engine, directory, progress, splits):
        self.engine = engine
        self.directory = directory
        self.progress = progress
        self.splits = splits
        self.inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
        self.context = {}

    def report(self, detail, done=0, total=1, **metrics):
        if self.progress:
            self.progress.update("running", done, total, detail=detail,
                                 metrics={"active_gpus": 4, **self.context, **metrics})

    def save(self, filename, row):
        if self.engine.rank == 0:
            atomic_json(self.directory / filename, row)

    def finish(self, phase, verdict):
        self.save(f"wiki_rsi_continual_{phase}_summary.json", {"provenance": self.inputs, **verdict})
        if self.progress:
            self.progress.update("done" if verdict["verdict"] == "GO" else "blocked", 1, 1,
                                 detail="continual_" + verdict["verdict"].lower())

    def evaluation(self, task, size, seed, label, count=None):
        cases = self.splits[f"{task}_dev"][:count]
        rows, tokens, truncations = self.engine.evaluate(cases, size, seed, lambda done, total: self.report(
            label, done, total, validation_cases=len(cases)))
        check_scores(rows, cases, size)
        return rows, {"tokens": tokens, "truncations": truncations, "sequences": len(cases) * size}

    def train(self, task, steps, seed, arm, label, *, bank=None, midpoint=None, evaluate=None):
        engine = self.engine
        started = monotonic()
        primary = fixed_order(self.splits[f"{task}_train"], seed)
        anchors = fixed_order(self.splits["b_anchor"], seed)
        packs = bank or []
        good_prompts = [case for case, _ in packs]
        rollout_count, update_count, mixed_reward_groups, eval_seconds = 0, 0, 0, 0.0
        for step in range(steps):
            local = step * 8 + engine.rank * 2
            first, second = primary[local % len(primary)], primary[(local + 1) % len(primary)]
            randomness = seed + 1031 * step + 127 * engine.rank
            rows = engine.generate([first] * GROUP, randomness)
            rollout_count += rollout_tokens(rows)
            mixed_reward_groups += 0 < sum(row.reward for row in rows) < GROUP
            if arm in BANK_ARMS:
                chosen = [packs[(step * 4 + i) % len(packs)][1] for i in range(min(4, len(packs)))]
                update_count += engine.update(rows, chosen, arm)
            else:
                if task == "a" and arm in {"static_mix", "prompt_replay"} and step % 2 == 0:
                    pool = good_prompts if arm == "prompt_replay" else anchors
                    second = pool[(step * 2 + engine.rank) % len(pool)]
                auxiliary = engine.generate([second] * GROUP, randomness + 1)
                rollout_count += rollout_tokens(auxiliary)
                update_count += engine.update(rows, auxiliary)
            self.report(label, step + 1, steps, train_steps=step + 1)
            if midpoint and step + 1 == steps // 2:
                rollout_count += midpoint()
            if evaluate and step + 1 == steps // 2:
                eval_start = monotonic()
                evaluate()
                eval_seconds += monotonic() - eval_start
        parts = engine.gather((rollout_count, update_count, mixed_reward_groups, monotonic() - started - eval_seconds))
        return {"rollout_tokens": sum(row[0] for row in parts), "update_tokens": sum(row[1] for row in parts),
                "mixed_reward_groups": sum(row[2] for row in parts), "seconds": max(row[3] for row in parts)}

    def smoke(self):
        self.engine.reset()
        cost = self.train("a", 1, SEEDS[0], "sequential", "continual_train_a")
        _, readout = self.evaluation("b", 2, SEEDS[0], "continual_eval_b", count=8)
        self.finish("smoke", {"verdict": "GO", "train": cost, "eval": readout})

    def necessity(self):
        self.engine.reset()
        rows, cost = {}, {}
        for snapshot in ("before", "after"):
            if snapshot == "after":
                cost["train"] = self.train("a", A_STEPS, SEEDS[0], "sequential", "continual_train_a")
                self.engine.save(self.directory / "stage_a_adapter")
            a, a_cost = self.evaluation("a", BASE_EVAL["a"], 12911, "continual_eval_a")
            b, b_cost = self.evaluation("b", 2 * BASE_EVAL["b"], 17021, "continual_eval_b")
            rows[snapshot] = {"a": a, "b": b}
            cost[snapshot] = {"a": a_cost, "b": b_cost}
        raw = "necessity_rollouts.json"
        self.save(raw, rows)
        if self.engine.rank == 0:
            self.finish("necessity", {**necessity_verdict(rows["before"], rows["after"]), "cost": cost,
                                      "rollouts_sha256": sha256_file(self.directory / raw)})

    def bank(self, seed):
        engine = self.engine
        started = monotonic()
        anchors = fixed_order(self.splits["b_anchor"], seed + 991)[engine.rank::engine.world]
        train, calibration, tokens = [], [], 0
        for index, case in enumerate(anchors):
            randomness = seed + 7211 * engine.rank + 97 * index
            rows = engine.generate([case] * PRETRAIN_BANK, randomness)
            tokens += rollout_tokens(rows)
            candidates = {}
            for row in rows:
                if row.reward and len(row.response) > PREFIX:
                    candidates.setdefault(row.response[:PREFIX], row)
            packs = []
            for slot, (prefix, row) in enumerate(sorted(candidates.items())[:MAX_PREFIXES]):
                continuations = engine.generate([case] * PREFIX_SAMPLES, randomness + 1 + slot,
                                                [prefix] * PREFIX_SAMPLES)
                tokens += rollout_tokens(continuations)
                hits = sum(trial.reward for trial in continuations)
                if hits:
                    packs.append((row, hits / PREFIX_SAMPLES))
            if packs:
                log_probs = engine.prefix_log_probs([item[0] for item in packs])
                entries = [(row, q, base, q) for (row, q), base in zip(packs, log_probs)]
                (calibration if index % 4 == 0 else train).append((case, entries))
            self.report("continual_anchor_bank", index + 1, len(anchors))
        parts = engine.gather((len(train), len(calibration), tokens, monotonic() - started))
        if min(row[0] for row in parts) < 4 or min(row[1] for row in parts) < 2:
            return None, None, {"bank_train_counts": [row[0] for row in parts],
                                "bank_calibration_counts": [row[1] for row in parts]}
        return train, calibration, {"bank_tokens": sum(row[2] for row in parts),
                                    "bank_seconds": max(row[3] for row in parts),
                                    "bank_train_counts": [row[0] for row in parts],
                                    "bank_calibration_counts": [row[1] for row in parts]}

    def refresh(self, bank, seed):
        engine = self.engine
        tokens = 0
        for index, (case, pack) in enumerate(bank[:8]):
            for slot, (row, _, base, base_q) in enumerate(pack):
                n = 4
                refreshed = engine.generate([case] * n, seed + 67 * engine.rank + 13 * index + slot,
                                            [row.response[:PREFIX]] * n)
                tokens += rollout_tokens(refreshed)
                refreshed_q = 0.5 * base_q + 0.5 * (sum(trial.reward for trial in refreshed) / n)
                pack[slot] = (row, refreshed_q, base, base_q)
        return tokens

    def calibration(self, bank):
        engine = self.engine
        differences = []
        for _, pack in bank:
            current = engine.prefix_log_probs([row for row, _, _, _ in pack])
            q = torch.tensor([p[3] for p in pack], dtype=torch.float64).log()
            before = torch.tensor([p[2] for p in pack], dtype=torch.float64)
            after = torch.tensor(current, dtype=torch.float64)
            differences.append(float(torch.logsumexp(after + q, 0) - torch.logsumexp(before + q, 0)))
        return mean(value for rank in engine.gather(differences) for value in rank)

    def arm(self, seed, arm, bank, calibration, bank_cost):
        engine = self.engine
        engine.reset()
        packs = [(case, list(pack)) for case, pack in bank]
        a = self.train("a", A_STEPS, seed, arm, "continual_train_a", bank=packs,
                       midpoint=lambda: self.refresh(packs, seed + 54021))
        proxy = self.calibration(calibration)
        readouts, costs = {}, {}

        def measure_b(step):
            readouts[step], costs[step] = self.evaluation("b", METHOD_EVAL, seed + 50021, "continual_eval_b")

        a_rows, costs["a_after_a"] = self.evaluation("a", METHOD_EVAL, seed + 30029, "continual_eval_a")
        measure_b(0)
        engine.new_optimizer()
        b = self.train("b", B_STEPS, seed + 29031, "sequential", "continual_train_b",
                       evaluate=lambda: measure_b(B_STEPS // 2))
        measure_b(B_STEPS)
        a_final, costs["a_final"] = self.evaluation("a", METHOD_EVAL, seed + 30029, "continual_eval_a")
        engine.save(self.directory / str(seed) / arm / "stage_b_adapter")
        return {"b_curve": {step: pass_one(rows) for step, rows in readouts.items()},
                "b_pass1": pass_one(readouts[B_STEPS]), "b_after_a_pass4": coverage(readouts[0]),
                "a_after_a_pass1": pass_one(a_rows), "a_final_pass1": pass_one(a_final),
                "b_final": readouts[B_STEPS], "calibration_ratio": proxy,
                "a_scores": a_rows, "a_final_scores": a_final, "b_scores": readouts, "eval_cost": costs,
                "cost": {"a_rollout_tokens": a["rollout_tokens"] + bank_cost["bank_tokens"],
                         "a_update_tokens": a["update_tokens"], "b_rollout_tokens": b["rollout_tokens"],
                         "b_update_tokens": b["update_tokens"],
                         "train_seconds": a["seconds"] + b["seconds"] + bank_cost["bank_seconds"],
                         "a_mixed_reward_groups": a["mixed_reward_groups"],
                         "b_mixed_reward_groups": b["mixed_reward_groups"]}}

    def method(self):
        records, bank_costs = {}, {}
        for seed_index, seed in enumerate(SEEDS):
            self.context = {"seed_index": seed_index + 1, "arm_index": 0}
            self.engine.reset()
            bank, calibration, cost = self.bank(seed)
            bank_costs[seed] = cost
            if bank is None:
                self.finish("method", {"verdict": "NO_GO", "reason": "anchor_bank_coverage", "bank_costs": bank_costs})
                return
            records[seed] = {}
            sequence = ARMS if seed_index == 0 else ARMS[::-1]
            for arm_index, arm in enumerate(sequence, start=1):
                self.context["arm_index"] = arm_index
                result = self.arm(seed, arm, bank, calibration, cost)
                records[seed][arm] = result
                self.save(f"{seed}-{arm}.json", result)
        small = {seed: {arm: {key: value for key, value in result.items()
                              if key not in {"b_final", "b_scores", "a_scores", "a_final_scores"}}
                        for arm, result in arms.items()} for seed, arms in records.items()}
        self.finish("method", {**method_verdict(records), "arms": small, "bank_costs": bank_costs})


def main():
    phase, run_dir, model_path, data_path, duration = sys.argv[1:]
    directory = Path(run_dir)
    lead = int(os.environ["LOCAL_RANK"]) == 0
    progress = Progress(directory, "continual_" + phase) if lead else None
    if progress:
        progress.update("running", detail="continual_loading")
    splits = load_bundle(data_path)["splits"]
    try:
        with progress.heartbeat() if progress else nullcontext():
            engine = Engine(model_path, splits, monotonic() + int(duration))
            getattr(Run(engine, directory, progress, splits), phase)()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
