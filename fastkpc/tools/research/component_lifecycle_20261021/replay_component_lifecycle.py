#!/usr/bin/env python3
"""Offline per-level component lifecycle replay for the dcc.gamma route.

Input: CSV exports of a canonical one-call receipt (tasks, levels, summary,
wave groups).  The script rebuilds every layer plan with the native
``make_layer_plan`` enumeration (edge-major, x->y orientation first, then
y->x, conditioning sets in lexicographic combination order), replays the
recorded executed tests and deletions, checks the plan against the receipt,
and then simulates component (target, S) production / consumption / release
under several consumption orders and production policies.

This is a structural replay only.  It does not predict wall time and does not
touch the GPU.  Run with ``python3 -I``.
"""
import csv
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


class LevelPlan:
    """Native-order layer plan plus the recorded execution outcome."""

    def __init__(self, level, p, adjacency, rows):
        self.level = level
        self.p = p
        snapshot = [list(row) for row in adjacency]
        self.tasks = []  # (edge_id, a, b, S)
        for x in range(p - 1):
            for y in range(x + 1, p):
                if not snapshot[x][y]:
                    continue
                edge_id = x * p + y
                nx = [v for v in range(p) if v != y and snapshot[x][v]]
                for cond in itertools.combinations(nx, level):
                    self.tasks.append((edge_id, x, y, cond))
                ny = [v for v in range(p) if v != x and snapshot[y][v]]
                for cond in itertools.combinations(ny, level):
                    self.tasks.append((edge_id, y, x, cond))
        count = len(self.tasks)
        self.executed = [False] * count
        self.deleted = [False] * count
        alive = set()
        for x in range(p - 1):
            for y in range(x + 1, p):
                if snapshot[x][y]:
                    alive.add(x * p + y)
        pointer = 0
        ignored = 0
        deletions = 0
        for position, (edge_id, a, b, cond) in enumerate(self.tasks):
            if edge_id not in alive:
                ignored += 1
                continue
            if pointer >= len(rows):
                raise RuntimeError(
                    "level %d plan has more live tasks than recorded tests" % level)
            row = rows[pointer]
            s_key = "|".join(str(v + 1) for v in cond)
            if (int(row["level"]) != level or int(row["task_index"]) != position + 1 or
                    int(row["x"]) != a + 1 or int(row["y"]) != b + 1 or
                    row["S_key"] != s_key or
                    int(row["edge_x"]) != min(a, b) + 1 or
                    int(row["edge_y"]) != max(a, b) + 1):
                raise RuntimeError(
                    "level %d task %d does not match recorded test %s" % (
                        level, position + 1, row))
            pointer += 1
            self.executed[position] = True
            if row["native_edge_deleted"] == "TRUE":
                self.deleted[position] = True
                deletions += 1
                alive.remove(edge_id)
                x, y = edge_id // p, edge_id % p
                adjacency[x][y] = False
                adjacency[y][x] = False
        if pointer != len(rows):
            raise RuntimeError(
                "level %d replayed %d of %d recorded tests" % (
                    level, pointer, len(rows)))
        self.tests_replayed = pointer
        self.ignored = ignored
        self.deletions = deletions

        # Component identities: (target, S).
        self.component_index = {}
        self.component_key = []
        self.task_components = []
        for edge_id, a, b, cond in self.tasks:
            pair = []
            for target in (a, b):
                key = (target, cond)
                index = self.component_index.get(key)
                if index is None:
                    index = len(self.component_key)
                    self.component_index[key] = index
                    self.component_key.append(key)
                pair.append(index)
            self.task_components.append(pair)
        self.component_count = len(self.component_key)
        self.refs = [0] * self.component_count
        for pair in self.task_components:
            for c in pair:
                self.refs[c] += 1
        consumed = set()
        for position, pair in enumerate(self.task_components):
            if self.executed[position]:
                consumed.update(pair)
        self.consumed = consumed
        self.edge_tasks = defaultdict(list)
        for position, task in enumerate(self.tasks):
            self.edge_tasks[task[0]].append(position)
        self.distinct_s = len({task[3] for task in self.tasks})
        self.executed_distinct_s = len(
            {task[3] for position, task in enumerate(self.tasks)
             if self.executed[position]})

    def static_production_order(self, planned_order=None):
        """Components sorted by first appearance in a static planned order."""
        if planned_order is None:
            planned_order = range(len(self.tasks))
        first = {}
        for rank, position in enumerate(planned_order):
            for c in self.task_components[position]:
                if c not in first:
                    first[c] = rank
        return sorted(first, key=lambda c: (first[c], c))

    def order_round_robin(self):
        """Static position-major merge: round r takes task r of every edge.

        With precomputed components this is the natural batched consumer:
        every active edge advances one task per round, so a round can
        evaluate one pair per active edge without speculation.  Deleted
        edges simply drop out; the relative order of the remaining tasks is
        unchanged, so the planned order is static.
        """
        edges = sorted(self.edge_tasks)
        longest = max(len(self.edge_tasks[e]) for e in edges) if edges else 0
        order = []
        rounds = []
        self.round_of_position = [None] * len(self.tasks)
        for r in range(longest):
            executed_in_round = 0
            for edge in edges:
                positions = self.edge_tasks[edge]
                if r < len(positions):
                    order.append(positions[r])
                    self.round_of_position[positions[r]] = r
                    if self.executed[positions[r]]:
                        executed_in_round += 1
            if executed_in_round:
                rounds.append(executed_in_round)
        self.round_robin_rounds = rounds
        return order

    def speculative_s_major(self, batch_size):
        """Structure of a fully speculative S-major physical schedule.

        Every planned task is evaluated physically, grouped by conditioning
        set; p-values are consumed logically in canonical order afterwards.
        """
        comps_by_s = defaultdict(set)
        pairs_by_s = defaultdict(int)
        for (edge_id, a, b, cond), pair in zip(self.tasks, self.task_components):
            comps_by_s[cond].update(pair)
            pairs_by_s[cond] += 1
        comps = sorted(len(v) for v in comps_by_s.values())
        pairs = sorted(pairs_by_s.values())
        ordered = sorted(comps_by_s)
        # Residency if components are produced in S order in batches and a
        # whole S group must be resident while its pairs are evaluated: one
        # batch in flight plus the groups the batch straddles.
        batches = 0
        current = 0
        for cond in ordered:
            current += len(comps_by_s[cond])
            if current >= batch_size:
                batches += 1
                current = 0
        if current:
            batches += 1
        return {
            "distinct_s": len(comps_by_s),
            "components_per_s_mean": sum(comps) / len(comps) if comps else 0.0,
            "components_per_s_p95": percentile(comps, 0.95),
            "components_per_s_max": comps[-1] if comps else 0,
            "pairs_per_s_mean": sum(pairs) / len(pairs) if pairs else 0.0,
            "pairs_per_s_max": pairs[-1] if pairs else 0,
            "physical_tests": len(self.tasks),
            "logical_tests": self.tests_replayed,
            "physical_waste_tests": len(self.tasks) - self.tests_replayed,
            "physical_waste_fraction": (len(self.tasks) - self.tests_replayed) / len(self.tasks) if self.tasks else 0.0,
            "components_produced": self.component_count,
            "components_wasted": self.component_count - len(self.consumed),
            "batch_size": batch_size,
            "batch_count": batches,
            "residency_bound_double_buffer": 2 * batch_size + (comps[-1] if comps else 0),
        }

    # ---- consumption orders -------------------------------------------
    def order_canonical(self):
        return list(range(len(self.tasks)))

    def order_bucket(self, rule):
        """Semantics-preserving merge of per-edge sequences.

        rule == "largest": take the ready conditioning-set bucket with the
        most edges (ties: lexicographically smallest S).  This approximates
        the production frontier scheduler without its prepared-context
        preference.
        rule == "smallest_s": take the lexicographically smallest ready S.
        """
        pointer = {edge: 0 for edge in self.edge_tasks}
        alive = set(self.edge_tasks)
        order = []
        buckets = defaultdict(list)
        for edge in sorted(alive):
            buckets[self.tasks[self.edge_tasks[edge][0]][3]].append(edge)
        while buckets:
            if rule == "largest":
                best = None
                for cond, edges in buckets.items():
                    if best is None or len(edges) > len(buckets[best]) or (
                            len(edges) == len(buckets[best]) and cond < best):
                        best = cond
            elif rule == "smallest_s":
                best = min(buckets)
            else:
                raise ValueError(rule)
            edges = buckets.pop(best)
            for edge in edges:
                position = self.edge_tasks[edge][pointer[edge]]
                order.append(position)
                if self.deleted[position]:
                    alive.discard(edge)
                    continue
                pointer[edge] += 1
                if pointer[edge] < len(self.edge_tasks[edge]):
                    buckets[self.tasks[self.edge_tasks[edge][
                        pointer[edge]]][3]].append(edge)
        return order

    # ---- lifecycle simulation ----------------------------------------
    def simulate(self, order, policy, batch_size=None, samples=20,
                 production_order=None, feedback="task"):
        """Simulate production/consumption/release along a consumption order.

        policy: "demand" (produce a component when the first executed task
        needs it), "batch" (produce the next ``batch_size`` still-live,
        unproduced components in first-use order of this consumption order,
        triggered on demand), "level" (produce every planned component before
        the first test).
        Release: a component is released as soon as every planned task that
        references it has been executed or has become unreachable because its
        edge was deleted.
        feedback: "task" applies each test's outcome (reference resolution and
        edge deletion) immediately after the test; "round" models a batched
        round-robin consumer that prepares a whole round before any of its
        results are known: every component a round needs is produced before
        the round starts, and all resolutions, deletions and releases of the
        round are applied only after the round completes.  "round" requires
        the round_robin order.
        """
        if feedback not in ("task", "round"):
            raise ValueError(feedback)
        if feedback == "round" and getattr(self, "round_of_position", None) is None:
            raise RuntimeError("round feedback requires the round_robin order")
        count = self.component_count
        pending = list(self.refs)
        produced = [False] * count
        released = [False] * count
        used = [False] * count
        live = 0
        peak = 0
        peak_step = -1
        produced_total = 0
        batches = []
        consumed_steps = 0
        executed_order = [t for t in order if self.executed[t]]
        step_total = len(executed_order)
        first_use = [None] * count
        for step, position in enumerate(executed_order):
            for c in self.task_components[position]:
                if first_use[c] is None:
                    first_use[c] = step
        if production_order is None:
            # Oracle production order: first use by an executed task.  This
            # is clairvoyant about which tasks will run and is a lower bound.
            candidates = sorted(
                (c for c in range(count) if first_use[c] is not None),
                key=lambda c: (first_use[c], c))
        else:
            # Static production order decided from the plan alone.
            candidates = list(production_order)
        # Components never touched by an executed task are only produced by
        # the "level" policy; append them for completeness of that policy.
        cursor = 0
        produce_time = [None] * count
        release_time = [None] * count
        curve = []
        sample_points = {int(round(k * (step_total - 1) / max(1, samples - 1)))
                         for k in range(samples)} if step_total > 0 else set()

        def produce(c, step):
            nonlocal live, produced_total
            produced[c] = True
            produce_time[c] = step
            produced_total += 1
            live += 1

        def release(c, step):
            nonlocal live
            if produced[c] and not released[c]:
                released[c] = True
                release_time[c] = step
                live -= 1

        def resolve(c, step):
            pending[c] -= 1
            if pending[c] == 0:
                release(c, step)

        if policy == "level":
            for c in range(count):
                produce(c, -1)
            peak = live
            peak_step = -1

        def ensure_produced(c, step):
            if produced[c]:
                return
            if policy == "demand":
                produce(c, step)
            elif policy == "batch":
                while not produced[c]:
                    taken = 0
                    while cursor_box[0] < len(candidates) and taken < batch_size:
                        candidate = candidates[cursor_box[0]]
                        cursor_box[0] += 1
                        if produced[candidate] or pending[candidate] == 0:
                            continue
                        produce(candidate, step)
                        taken += 1
                    if taken == 0:
                        raise RuntimeError("batch production stalled")
                    batches.append(taken)
            else:
                raise RuntimeError("component unproduced under level policy")

        def apply_outcome(position, step):
            for c in self.task_components[position]:
                resolve(c, step)
            if self.deleted[position]:
                edge = self.tasks[position][0]
                for later in self.edge_tasks[edge]:
                    if later > position and not self.executed[later]:
                        for c in self.task_components[later]:
                            resolve(c, step)

        cursor_box = [cursor]
        if feedback == "task":
            groups = [[position] for position in executed_order]
        else:
            groups = []
            current_round = None
            for position in executed_order:
                r = self.round_of_position[position]
                if r != current_round:
                    groups.append([])
                    current_round = r
                groups[-1].append(position)
        step = -1
        for group in groups:
            group_start = step + 1
            # Production boundary: everything the group needs exists before
            # the group's first test runs.
            for position in group:
                for c in self.task_components[position]:
                    ensure_produced(c, group_start)
                    used[c] = True
            if live > peak:
                peak = live
                peak_step = group_start
            step += len(group)
            # Feedback boundary: outcomes are applied after the group.
            for position in group:
                apply_outcome(position, step)
            if live > peak:
                peak = live
                peak_step = step
            if any(s in sample_points for s in range(group_start, step + 1)):
                curve.append((step, live))
        if live != 0:
            raise RuntimeError("live components remain after level end: %d" % live)
        waste = sum(1 for c in range(count) if produced[c] and not used[c])
        lifetimes = [release_time[c] - max(0, produce_time[c])
                     for c in range(count) if produced[c] and used[c]]
        return {
            "policy": policy,
            "feedback": feedback,
            "production": ("static" if production_order is not None else "oracle") if policy == "batch" else None,
            "batch_size": batch_size,
            "executed_steps": step_total,
            "produced": produced_total,
            "used": sum(used),
            "wasted": waste,
            "peak_live": peak,
            "peak_step": peak_step,
            "peak_step_fraction": (peak_step / step_total) if step_total > 0 and peak_step >= 0 else None,
            "batch_count": len(batches),
            "batch_mean": (sum(batches) / len(batches)) if batches else None,
            "batch_min": min(batches) if batches else None,
            "batch_full_fraction": (sum(1 for b in batches if b == batch_size) / len(batches)) if batches else None,
            "lifetime_p50": percentile(lifetimes, 0.5),
            "lifetime_p95": percentile(lifetimes, 0.95),
            "lifetime_max": max(lifetimes) if lifetimes else None,
            "lifetime_p50_fraction": (percentile(lifetimes, 0.5) / step_total) if lifetimes and step_total else None,
            "curve": curve,
        }


def main(argv):
    if len(argv) < 2:
        print("usage: replay_component_lifecycle.py ARTIFACT_DIR [OUTPUT_JSON]")
        return 2
    art = argv[1]
    output = argv[2] if len(argv) > 2 else os.path.join(art, "component_lifecycle_replay.json")
    paths = {name: os.path.join(art, "canonical_queues32_%s.csv" % name)
             for name in ("tasks", "levels", "summary", "wave_groups")}
    tasks = read_csv(paths["tasks"])
    levels = {int(r["level"]): r for r in read_csv(paths["levels"])}
    summary = {r["key"]: r["value"] for r in read_csv(paths["summary"])}
    waves = read_csv(paths["wave_groups"])
    p = int(summary["p"])
    n = int(summary["n"])
    num_col = int(summary["numCol"])
    bytes_per_entry = (n * num_col + num_col + 2) * 8
    capacity = int(summary["legacy_component_cache_capacity_entries"])
    residual_bytes = n * 8
    batch_sizes = [256, 640, 1280, 2560]

    ids = [int(r["canonical_test_order_id"]) for r in tasks]
    if ids != list(range(1, len(tasks) + 1)):
        raise RuntimeError("canonical test order ids are not contiguous")
    rows_by_level = defaultdict(list)
    for r in tasks:
        rows_by_level[int(r["level"])].append(r)
    wave_by_level = defaultdict(lambda: defaultdict(float))
    for w in waves:
        level = int(w["level"])
        wave_by_level[level]["groups"] += 1
        wave_by_level[level]["miss"] += int(w["component_miss_count"])
        wave_by_level[level]["hit"] += int(w["component_hit_count"])
        wave_by_level[level]["eig_s"] += float(w["full_eig_cuda_ms"]) / 1000.0
        wave_by_level[level]["build_s"] += float(w["component_build_cuda_ms"]) / 1000.0

    adjacency = [[i != j for j in range(p)] for i in range(p)]
    max_level = max(levels)
    report_levels = []
    totals = defaultdict(int)
    for level in range(max_level + 1):
        plan = LevelPlan(level, p, adjacency, rows_by_level.get(level, []))
        expect = levels[level]
        checks = {
            "tasks_planned": (len(plan.tasks), int(expect["tasks_planned"])),
            "tests_replayed": (plan.tests_replayed, int(expect["tests_replayed"])),
            "tasks_ignored_after_delete": (plan.ignored, int(expect["tasks_ignored_after_delete"])),
            "deletions": (plan.deletions, int(expect["deletions"])),
            "unique_component_misses": (len(plan.consumed), int(wave_by_level[level]["miss"])),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise RuntimeError("level %d %s: replay %d != receipt %d" % (level, name, got, want))
        orders = {
            "canonical": plan.order_canonical(),
            "round_robin": plan.order_round_robin(),
            "frontier_largest_bucket": plan.order_bucket("largest"),
            "smallest_ready_s": plan.order_bucket("smallest_s"),
        }
        policies = {}
        for order_name, order in orders.items():
            executed_positions = [t for t in range(len(plan.tasks)) if plan.executed[t]]
            if sorted(t for t in order if plan.executed[t]) != executed_positions or \
                    len(order) != len(set(order)):
                raise RuntimeError("order %s does not cover every executed task once" % order_name)
            entry = {"demand": plan.simulate(order, "demand")}
            for b in batch_sizes:
                entry["batch_oracle_%d" % b] = plan.simulate(order, "batch", b)
            if order_name in ("canonical", "round_robin"):
                static_order = plan.static_production_order(order)
                for b in batch_sizes:
                    entry["batch_static_%d" % b] = plan.simulate(
                        order, "batch", b, production_order=static_order)
                entry["level"] = plan.simulate(order, "level")
            if order_name == "round_robin":
                entry["demand_round_feedback"] = plan.simulate(
                    order, "demand", feedback="round")
                for b in batch_sizes:
                    entry["batch_static_%d_round_feedback" % b] = plan.simulate(
                        order, "batch", b, production_order=static_order,
                        feedback="round")
                rounds = plan.round_robin_rounds
                entry["rounds"] = {
                    "count": len(rounds),
                    "pairs_per_round_mean": sum(rounds) / len(rounds) if rounds else None,
                    "pairs_per_round_p50": percentile(rounds, 0.5),
                    "pairs_per_round_p10": percentile(rounds, 0.1),
                    "pairs_per_round_max": max(rounds) if rounds else None,
                    "rounds_with_ge_64_pairs": sum(1 for r in rounds if r >= 64),
                }
            policies[order_name] = entry
        speculative = plan.speculative_s_major(640)
        unconsumed = plan.component_count - len(plan.consumed)
        row = {
            "level": level,
            "tasks_planned": len(plan.tasks),
            "tests_replayed": plan.tests_replayed,
            "tasks_ignored_after_delete": plan.ignored,
            "deletions": plan.deletions,
            "edges_at_level_start": len(plan.edge_tasks),
            "distinct_s_planned": plan.distinct_s,
            "distinct_s_executed": plan.executed_distinct_s,
            "components_planned": plan.component_count,
            "components_consumed": len(plan.consumed),
            "components_unconsumed": unconsumed,
            "unconsumed_fraction": unconsumed / plan.component_count if plan.component_count else 0.0,
            "mean_refs_per_component": (2.0 * len(plan.tasks) / plan.component_count) if plan.component_count else 0.0,
            "components_per_s_mean": plan.component_count / plan.distinct_s if plan.distinct_s else 0.0,
            "planned_component_bytes": plan.component_count * bytes_per_entry,
            "planned_residual_bytes": plan.component_count * residual_bytes,
            "receipt_eig_cuda_s": wave_by_level[level]["eig_s"],
            "receipt_component_build_cuda_s": wave_by_level[level]["build_s"],
            "receipt_groups": int(wave_by_level[level]["groups"]),
            "receipt_miss_per_group": (wave_by_level[level]["miss"] / wave_by_level[level]["groups"]) if wave_by_level[level]["groups"] else None,
            "policies": policies,
            "speculative_s_major_640": speculative,
        }
        report_levels.append(row)
        totals["components_planned"] += plan.component_count
        totals["components_consumed"] += len(plan.consumed)
        totals["tasks_planned"] += len(plan.tasks)
        totals["tests_replayed"] += plan.tests_replayed
        print("level %d: planned %d tests %d comps %d consumed %d unconsumed %.1f%% distinctS %d "
              "demand-peak canon %d / frontier %d / sweep %d  batch640-peak canon %d waste %d" % (
                  level, len(plan.tasks), plan.tests_replayed, plan.component_count, len(plan.consumed),
                  100.0 * row["unconsumed_fraction"], plan.distinct_s,
                  policies["canonical"]["demand"]["peak_live"],
                  policies["frontier_largest_bucket"]["demand"]["peak_live"],
                  policies["smallest_ready_s"]["demand"]["peak_live"],
                  policies["canonical"]["batch_static_640"]["peak_live"],
                  policies["canonical"]["batch_static_640"]["wasted"]))
        rr = policies["round_robin"]
        print("         round-robin: rounds %d pairs/round mean %.1f p50 %s | demand-peak %d | static640 peak %d waste %d batches %d (full %.0f%%)" % (
            rr["rounds"]["count"], rr["rounds"]["pairs_per_round_mean"] or 0.0, rr["rounds"]["pairs_per_round_p50"],
            rr["demand"]["peak_live"], rr["batch_static_640"]["peak_live"], rr["batch_static_640"]["wasted"],
            rr["batch_static_640"]["batch_count"], 100.0 * (rr["batch_static_640"]["batch_full_fraction"] or 0.0)))
        rf = rr["batch_static_640_round_feedback"]
        print("         round feedback: demand peak %d | static640 produced %d waste %d peak %d" % (
            rr["demand_round_feedback"]["peak_live"], rf["produced"], rf["wasted"], rf["peak_live"]))
        print("         static640 batches %d (mean %.1f, full %.0f%%) lifetime p50 %.3f of level | spec S-major: comps/S mean %.1f max %d, pairs/S mean %.1f, phys waste %.1f%%" % (
            policies["canonical"]["batch_static_640"]["batch_count"],
            policies["canonical"]["batch_static_640"]["batch_mean"] or 0.0,
            100.0 * (policies["canonical"]["batch_static_640"]["batch_full_fraction"] or 0.0),
            policies["canonical"]["demand"]["lifetime_p50_fraction"] or 0.0,
            speculative["components_per_s_mean"], speculative["components_per_s_max"],
            speculative["pairs_per_s_mean"], 100.0 * speculative["physical_waste_fraction"]))
        sys.stdout.flush()

    native_sha256 = summary.get("receipt_native_sha256", "")
    if len(native_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in native_sha256):
        raise RuntimeError("summary export lacks the receipt's top-level receipt_native_sha256")
    receipt_path = summary.get("receipt_path")
    receipt_input = None
    if receipt_path and os.path.exists(receipt_path):
        receipt_input = {"path": receipt_path, "sha256": sha256_file(receipt_path)}
    prefill_unique = int(summary["prefill_unique_target_key_count"])
    prefill_consumed = int(summary["prefill_consumed_unique_target_key_count"])
    level0 = report_levels[0]
    cross = {
        "planned_components_levels_ge1": totals["components_planned"] - level0["components_planned"],
        "prefill_unique_target_key_count": prefill_unique,
        "consumed_components_levels_ge1": totals["components_consumed"] - level0["components_consumed"],
        "prefill_consumed_unique_target_key_count": prefill_consumed,
        "consumed_components_all_levels": totals["components_consumed"],
        "legacy_component_cache_insert_count": int(summary["legacy_component_cache_insert_count"]),
    }
    if cross["planned_components_levels_ge1"] != prefill_unique or \
            cross["consumed_components_levels_ge1"] != prefill_consumed or \
            totals["components_consumed"] != cross["legacy_component_cache_insert_count"]:
        raise RuntimeError("component totals do not match receipt counters: %s" % cross)
    peak_planned = max(r["components_planned"] for r in report_levels)
    result = {
        "schema": "fastkpc-component-lifecycle-replay-v1",
        "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in paths.items()},
        "receipt": {
            "native_sha256": native_sha256,
            "elapsed_sec": float(summary["receipt_elapsed_sec"]) if "receipt_elapsed_sec" in summary else None,
            "summary_elapsed_sec": float(summary["elapsed_sec"]) if "elapsed_sec" in summary else None,
            "schema": summary.get("receipt_schema"),
            "workload": summary.get("receipt_workload"),
            "rds": receipt_input,
            "scheduler": summary.get("scheduler"),
            "p": p, "n": n, "numCol": num_col,
            "bytes_per_entry": bytes_per_entry,
            "cache_capacity_entries": capacity,
            "cache_capacity_bytes": capacity * bytes_per_entry,
        },
        "cross_checks": cross,
        "totals": dict(totals),
        "max_level_components_planned": peak_planned,
        "max_level_components_planned_bytes": peak_planned * bytes_per_entry,
        # Capacity requirements.  "reset_only" assumes the cache is cleared
        # between levels but never reclaims a slot inside a level, so it must
        # hold a level's cumulative production.  "in_level_reclaim" assumes
        # slots are reused as soon as a component loses its last reference.
        "capacity_requirements": {
            name: {
                "reset_only_entries": max(r["policies"][order][key]["produced"] for r in report_levels),
                "reset_only_bytes": max(r["policies"][order][key]["produced"] for r in report_levels) * bytes_per_entry,
                "in_level_reclaim_entries": max(r["policies"][order][key]["peak_live"] for r in report_levels),
                "in_level_reclaim_bytes": max(r["policies"][order][key]["peak_live"] for r in report_levels) * bytes_per_entry,
                "total_produced": sum(r["policies"][order][key]["produced"] for r in report_levels),
                "total_wasted": sum(r["policies"][order][key]["wasted"] for r in report_levels),
            }
            for name, (order, key) in {
                "whole_level": ("canonical", "level"),
                "round_robin_static_640_task_feedback": ("round_robin", "batch_static_640"),
                "round_robin_static_640_round_feedback": ("round_robin", "batch_static_640_round_feedback"),
                "round_robin_demand_round_feedback": ("round_robin", "demand_round_feedback"),
            }.items()
        },
        "levels": report_levels,
    }
    print("capacity requirements:")
    for name, value in result["capacity_requirements"].items():
        print("  %-42s reset-only %6d (%.2f GiB)  in-level reclaim %6d (%.2f GiB)  produced %d wasted %d" % (
            name, value["reset_only_entries"], value["reset_only_bytes"] / 2**30,
            value["in_level_reclaim_entries"], value["in_level_reclaim_bytes"] / 2**30,
            value["total_produced"], value["total_wasted"]))
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)
    print("cross checks", cross)
    print("max planned components in one level", peak_planned,
          "bytes %.3f GiB" % (peak_planned * bytes_per_entry / 2**30),
          "capacity", capacity)
    print("wrote", output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
