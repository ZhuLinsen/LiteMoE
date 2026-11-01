"""CUDA Graph and race-free device/host snapshot ring."""

import torch


class OutputRing:
    def __init__(self, emit, consume, asynchronous=False):
        self.emit, self.consume = emit, consume
        self.asynchronous = asynchronous
        self.pending = [False, False]
        self.used = [False, False]
        if asynchronous:
            self.device = [torch.empty_like(emit) for _ in range(2)]
            self.host = [torch.empty(emit.shape, dtype=emit.dtype, pin_memory=True) for _ in range(2)]
            self.ready = [torch.cuda.Event() for _ in range(2)]
            self.copied = [torch.cuda.Event() for _ in range(2)]
            self.stream = torch.cuda.Stream(device=emit.device)

    def take(self, s):
        if self.pending[s]:
            self.copied[s].synchronize()
            self.consume(self.host[s].tolist())
            self.pending[s] = False

    def publish(self, n):
        if not self.asynchronous:
            self.consume(self.emit.cpu().tolist())
            return
        s = n % 2
        self.take(s)
        compute = torch.cuda.current_stream(self.emit.device)
        if self.used[s]:
            compute.wait_event(self.copied[s])
        self.device[s].copy_(self.emit)
        self.ready[s].record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ready[s])
            self.host[s].copy_(self.device[s], non_blocking=True)
            self.copied[s].record()
        self.pending[s] = self.used[s] = True
        self.take((n - 1) % 2)

    def drain(self):
        for s in range(2):
            self.take(s)


def capture_decode(step, reset):
    """Warmup/capture mutate state. Caller restores it before real prefill."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reset()
            step()
        reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.synchronize()
    reset()
    return graph
