"""Decode fixed token batches with independently placed vLLM replicas."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from work3.wiki_rsi_progress import atomic_json


def native_gpu_ids(legacy_device, devices):
    if devices is None:
        return []
    if not re.fullmatch(r"[0-9]+", legacy_device or ""):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the original single GPU for the legacy repeat")
    selected = [value.strip() for value in devices.split(",")]
    if not 1 <= len(selected) <= 3 or any(not re.fullmatch(r"[0-9]+", value) for value in selected):
        raise ValueError("--native-gpus expects up to three comma-separated GPU indices")
    gpu_ids = [int(value) for value in selected]
    if len(set(gpu_ids)) != len(gpu_ids) or int(legacy_device) in gpu_ids:
        raise ValueError("--native-gpus must be distinct from one another and the legacy GPU")
    return [str(value) for value in gpu_ids]


class DecodeEngine:
    def __init__(self, model_dir, profile):
        from vllm import LLM, SamplingParams

        self.max_tokens = profile["max_new_tokens"]
        self.engine = LLM(
            model=str(model_dir), tensor_parallel_size=profile["tensor_parallel_size"],
            max_model_len=profile["max_model_len"], dtype=profile["dtype"], seed=profile["seed"],
            gpu_memory_utilization=profile["gpu_memory_utilization"],
            enforce_eager=profile["enforce_eager"], generation_config=profile["generation_config"],
        )
        self.params = SamplingParams(temperature=profile["temperature"], max_tokens=self.max_tokens,
                                     skip_special_tokens=True)

    def generate(self, token_batches):
        prompts = [{"prompt_token_ids": ids} for ids in token_batches]
        responses = self.engine.generate(prompts, sampling_params=self.params, use_tqdm=False)
        records = []
        for prompt, response in zip(prompts, responses, strict=True):
            if list(response.prompt_token_ids) != prompt["prompt_token_ids"] or len(response.outputs) != 1:
                raise RuntimeError("Prompt or completion count changed in vLLM")
            completion = response.outputs[0]
            length = len(completion.token_ids)
            if length > self.max_tokens or completion.finish_reason not in {"stop", "length"}:
                raise RuntimeError("Query generation did not respect the fixed token budget")
            records.append({"text": completion.text, "finish": completion.finish_reason, "tokens": length})
        return records


def worker(job_path, output_path):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    start = time.monotonic()
    engine = DecodeEngine(job["model_dir"], job["profile"])
    loaded = time.monotonic()
    batches = {index: engine.generate(prompts) for index, prompts in job["batches"]}
    atomic_json(output_path, {
        "batches": batches, "load_seconds": round(loaded - start, 3),
        "generation_seconds": round(time.monotonic() - loaded, 3),
    })


class NativeReplicas:
    def __init__(self):
        self.jobs = []

    def require_healthy(self):
        for _, process, log, _, _ in self.jobs:
            status = process.poll()
            if status is not None and status != 0:
                raise RuntimeError(f"Native GPU replica exited with code {status}. See {log.name}")

    def collect(self):
        results, timings = {}, {}
        for device, process, _, output, assigned in self.jobs:
            process.wait()
            self.require_healthy()
            data = json.loads(output.read_text(encoding="utf-8"))
            decoded = {int(index): records for index, records in data["batches"].items()}
            if decoded.keys() != assigned or results.keys() & decoded.keys():
                raise RuntimeError("Native replica results do not match assigned batches")
            results.update(decoded)
            timings[device] = {
                "batches": len(assigned), "load_seconds": data["load_seconds"],
                "generation_seconds": data["generation_seconds"],
            }
        return results, timings

    def stop(self):
        active = [proc for _, proc, _, _, _ in self.jobs if proc.poll() is None or proc.returncode != 0]
        for proc in active:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in active:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


@contextmanager
def native_replicas(run_dir, devices, batches, model_dir, profile):
    replicas = NativeReplicas()
    with TemporaryDirectory(prefix=".decode-", dir=run_dir) as temporary:
        scratch = Path(temporary)
        try:
            for number, device in enumerate(devices):
                assigned = list(range(number, len(batches), len(devices)))
                if not assigned:
                    continue
                job = scratch / f"gpu{device}.json"
                output = scratch / f"gpu{device}_out.json"
                log = run_dir / f"w2_contract_native_gpu{device}.log"
                atomic_json(job, {
                    "model_dir": str(model_dir), "profile": {**profile, "cuda_visible_devices": device},
                    "batches": [(index, batches[index]) for index in assigned],
                })
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "work3.wiki_rsi_decode", str(job), str(output)],
                        stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                    )
                replicas.jobs.append((device, process, log, output, set(assigned)))
            yield replicas
        finally:
            replicas.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    worker(options.job, options.output)
