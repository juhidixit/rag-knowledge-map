import pytest

from ragmap.embeddings import HashingEmbedder
from ragmap.engine import Engine, EngineConfig
from ragmap.ingest import parse_text

SAMPLE = """# Battery Thermal Management for Electric Vehicles

## Introduction

Lithium-ion battery packs in electric vehicles degrade quickly when cell temperature leaves the 15 to 35 degree window.
A battery thermal management system keeps the pack inside that window during fast charging and aggressive driving.

## Liquid Cooling

Liquid cooling plates circulate a glycol coolant under the cell modules. Liquid cooling removes heat four times faster
than air cooling and keeps the temperature spread between cells below five degrees during fast charging.

The coolant pump and the chiller consume energy, so the controller trades cooling power against driving range.

## Air Cooling

Air cooling uses cabin air and fans. Air cooling is cheap and light but cannot keep up with fast charging above 100 kW,
which is why most long-range electric vehicles use liquid cooling.

## Model Predictive Control

A model predictive control strategy forecasts cell temperature from the route and the charging plan and pre-cools the
battery pack before a fast charging stop. In simulation, model predictive control reduced peak cell temperature by
4 degrees and saved 3 percent of driving range compared with a thermostat controller.
"""


class FakeReranker:
    """Scores by word overlap, deterministic and dependency-free."""

    name = "fake"

    def score(self, query, passages):
        import numpy as np

        q = set(query.lower().split())
        return np.array([len(q & set(p.lower().split())) for p in passages], dtype=float)


@pytest.fixture
def doc():
    return parse_text(SAMPLE)


@pytest.fixture
def engine(doc):
    eng = Engine(EngineConfig(chunk_words=60, rerank_model=None), embedder=HashingEmbedder(), reranker=FakeReranker())
    eng.add_document(doc)
    return eng
