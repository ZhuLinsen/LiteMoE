"""Frozen benchmark text used to build the published inputs."""
from litemoe.engine import Request

TEXTS = [
    "Explain how a computer stores information in memory. Use a concrete example and describe why locality matters. ",
    "A city plans a new public park with trees, footpaths, a pond and a small library. Discuss the design tradeoffs. ",
    "Describe the water cycle, including evaporation, condensation and precipitation, for a curious student. ",
    "Write a short Python function that computes a running average. Explain its time and space complexity. ",
    "Compare expert parallelism and data parallelism in neural network inference. Discuss communication and memory. ",
    "How can a scientific experiment distinguish a useful optimization from random variation? Explain controls and measurement. ",
]


def workload(tokenizer, batch, prompt_length, output_length, split="eval"):
    texts = TEXTS[:2] if split == "calibration" else TEXTS[2:]
    requests = []
    for i in range(batch):
        text = texts[i % len(texts)]
        tokens = tokenizer.encode(text * (prompt_length // 8 + 2), add_special_tokens=False)[:prompt_length]
        assert len(tokens) == prompt_length
        requests.append(Request(f"{split}-{i}", tokens, output_length))
    return requests
