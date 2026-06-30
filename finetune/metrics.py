from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from pesq import pesq
from pystoi import stoi


def si_snr(reference, estimate, eps=1e-8):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projection = np.dot(estimate, reference) * reference / (np.dot(reference, reference) + eps)
    noise = estimate - projection
    return float(10 * np.log10((np.dot(projection, projection) + eps) / (np.dot(noise, noise) + eps)))


def utterance_metrics(args):
    reference, estimate, sample_rate = args
    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    return {
        "pesq": float(pesq(sample_rate, reference, estimate, "wb")),
        "stoi": float(stoi(reference, estimate, sample_rate, extended=False)),
        "si_snr": si_snr(reference, estimate),
    }


class MetricAccumulator:
    def __init__(self, workers=1):
        self.pool = (
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            if workers > 1
            else None
        )
        self.max_pending = max(1, workers * 4)
        self.pending = []
        self.sums = {"pesq": 0.0, "stoi": 0.0, "si_snr": 0.0}
        self.count = 0

    def _add_value(self, value):
        for key in self.sums:
            self.sums[key] += value[key]
        self.count += 1

    def submit(self, item):
        if self.pool is None:
            self._add_value(utterance_metrics(item))
            return
        self.pending.append(self.pool.submit(utterance_metrics, item))
        if len(self.pending) >= self.max_pending:
            self._add_value(self.pending.pop(0).result())

    def finish(self):
        for future in self.pending:
            self._add_value(future.result())
        self.pending.clear()
        if self.pool is not None:
            self.pool.shutdown()
        if not self.count:
            raise ValueError("no metric samples")
        return {key: value / self.count for key, value in self.sums.items()}
