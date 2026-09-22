"""Two-class cap/ground segmentation export; ground can never become a cup."""

import ast
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys
import time
import numpy as np
from cup_grasp_demo.flow.core import ROOT
from cup_grasp_demo.flow.cup_perception import decode
from dice_cup_localization.yolo_seg import preprocess


def cap_outputs(outputs):
    if (
        len(outputs) != 2
        or outputs[0].shape != (1, 38, 8400)
        or outputs[1].shape != (1, 32, 160, 160)
    ):
        raise ValueError("Green model requires [1,38,8400] and [1,32,160,160]")
    if not all(np.isfinite(x).all() for x in outputs):
        raise ValueError("Nonfinite YOLO outputs")
    scores = outputs[0][:, 4:6, :]
    if scores.min() < -1e-6 or scores.max() > 1 + 1e-6:
        raise ValueError("Invalid class scores")
    detection = np.concatenate([outputs[0][:, :5, :], outputs[0][:, 6:, :]], axis=1)
    detection[:, 4, :] = np.where(
        scores[:, 0, :] >= scores[:, 1, :], scores[:, 0, :], 0
    )
    return [detection, outputs[1]]


def runtime_settings(opts):
    threads = opts.get("inference_threads", 1)
    provider = opts.get("inference_provider", "cpu")
    cpus = opts.get("inference_cpu_ids", [])
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("YOLO inference_threads must be 1..4")
    if provider not in ("cpu", "spacemit"):
        raise ValueError("YOLO inference_provider must be cpu or spacemit")
    if not isinstance(cpus, (list, tuple)) or any(type(x) is not int for x in cpus):
        raise ValueError("YOLO inference_cpu_ids must be integer CPU IDs")
    if provider == "spacemit":
        if len(cpus) != threads or len(set(cpus)) != len(cpus) or any(x < 8 or x > 15 for x in cpus):
            raise ValueError("K3 AI backend needs one distinct CPU ID in 8..15 per inference thread")
    elif cpus:
        raise ValueError("inference_cpu_ids is for the spacemit backend; use [] for cpu")
    return threads, provider, tuple(cpus)


def configured_session(opts):
    path = (ROOT / opts["model"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Model must be inside dice_demo")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return session(str(path), sha, opts["ort_package_dir"], *runtime_settings(opts))


@lru_cache(maxsize=2)
def session(path, sha, package, threads=1, provider="cpu", cpu_ids=()):
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("YOLO inference_threads must be 1..4")
    try:
        ort = importlib.import_module("onnxruntime")
    except ModuleNotFoundError:
        if package not in sys.path:
            sys.path.append(package)
        ort = importlib.import_module("onnxruntime")
    runtime_settings(dict(inference_threads=threads, inference_provider=provider, inference_cpu_ids=cpu_ids))
    providers = ["CPUExecutionProvider"]
    if provider == "spacemit":
        # The vendor EP creates AI threads with the K3 kernel-specific affinity setup.
        # Do not move the Python/camera/CAN thread to the A100 domain with taskset.
        importlib.import_module("spacemit_ort")
        providers = [("SpaceMITExecutionProvider", {
            "SPACEMIT_EP_INTRA_THREAD_NUM": str(threads),
            "SPACEMIT_EP_INTRA_THREAD_AFFINITY": ";".join(map(str, cpu_ids)),
            "SPACEMIT_EP_INTER_THREAD_NUM": "1",
        }), "CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1 if provider == "spacemit" else threads
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(
        path, sess_options=options, providers=providers
    )
    if provider == "spacemit" and "SpaceMITExecutionProvider" not in model.get_providers():
        raise RuntimeError("SpaceMIT EP unavailable; select cpu explicitly instead of silent fallback")
    if (
        model.get_inputs()[0].shape != [1, 3, 640, 640]
        or model.get_inputs()[0].type != "tensor(float)"
    ):
        raise ValueError("Invalid green YOLO input")
    metadata = model.get_modelmeta().custom_metadata_map
    if metadata.get("task") != "segment" or ast.literal_eval(
        metadata.get("names", "{}")
    ) != {0: "cap", 1: "ground"}:
        raise ValueError("Green model must have cap=0, ground=1")
    return model


def infer(image, opts):
    path = (ROOT / opts["model"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Model must be inside dice_demo")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = configured_session(opts)
    tensor, transform = preprocess(image)
    start = time.perf_counter()
    outputs = model.run(None, {model.get_inputs()[0].name: tensor})
    items = decode(cap_outputs(outputs), image.shape, transform, opts)
    return items, dict(
        model=str(path.relative_to(ROOT)),
        sha256=sha,
        classes={0: "cap", 1: "ground"},
        selected_class=0,
        providers=model.get_providers(),
        inference_threads=opts.get("inference_threads", 1),
        inference_cpu_ids=opts.get("inference_cpu_ids", []),
        inference_s=time.perf_counter() - start,
    )
