"""Conflict-free KV stores: one writer per request, head and channel."""
import triton
import triton.language as tl


@triton.jit
def _store(K, V, KC, VC, POS, ACTIVE, SLOTS, H: tl.constexpr, D: tl.constexpr,
           C: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr, KS3: tl.constexpr,
           VS0: tl.constexpr, VS1: tl.constexpr, VS3: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (x < H * D) & tl.load(ACTIVE + row)
    head, channel = x // D, x % D
    position = tl.load(POS + row)
    slot = tl.load(SLOTS + row)
    target = ((slot * H + head) * C + position) * D + channel
    k = tl.load(K + row * KS0 + head * KS1 + channel * KS3, valid, other=0)
    v = tl.load(V + row * VS0 + head * VS1 + channel * VS3, valid, other=0)
    tl.store(KC + target, k, valid)
    tl.store(VC + target, v, valid)


def write(cache, layer, k, v, positions, active, slots):
    """Inputs may have strided heads/channels; cache storage is contiguous BHCD."""
    _, heads, _, dim = k.shape
    _store[(k.shape[0], triton.cdiv(heads * dim, 256))](
        k, v, cache.k[layer], cache.v[layer], positions, active, slots, heads, dim,
        cache.capacity, k.stride(0), k.stride(1), k.stride(3),
        v.stride(0), v.stride(1), v.stride(3), BLOCK=256)
