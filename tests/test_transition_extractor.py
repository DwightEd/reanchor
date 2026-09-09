from types import SimpleNamespace

import numpy as np

from reanchor.capture.protocol import AuditSample
from reanchor.discovery.extractor import TransitionConfig, TransitionExtractor


class MetadataDataset:
    def load_metadata(self, sample, *fields):
        values = {
            "token_ids": np.arange(8),
            "token_text": np.array(list("abcdefgh")),
            "row_position": np.arange(2, 8),
            "response_start": np.array(3),
            "special_mask": np.zeros(8, dtype=bool),
            "evidence_mask": np.array([False, False, True, False, False, False, False, False]),
        }
        return {field: values[field] for field in fields}


class FixedAttentionReader:
    def __init__(self, dataset):
        self.dataset = dataset

    def iter_layers(self, sample):
        rows = np.arange(2, 8)
        attention = np.zeros((2, len(rows), 8), dtype=np.float32)
        for index, query in enumerate(rows):
            attention[:, index, query] = 1
        attention[:, 2] = 0
        attention[:, 2, 4] = 1
        attention[:, 3] = 0
        attention[:, 3, 2] = 0.8
        attention[:, 3, 5] = 0.2
        yield 0, attention
        yield 1, attention * 1


def test_extractor_builds_layer_head_features_and_token_scores():
    sample = AuditSample("train", "QA", "one", "source", SimpleNamespace(), 5, 3)
    reader = FixedAttentionReader(MetadataDataset())
    config = TransitionConfig(
        window=2,
        local_floor=0.5,
        site_gain_floor=0.1,
        broad_head_fraction=0.5,
        position_bins=2,
    )

    features = TransitionExtractor(reader, config).run(sample)
    scores = features.token_scores(config)

    assert features.remote_gain.shape == (2, 2, 6)
    assert features.eligible.tolist() == [False, True, True, True, True, False]
    assert scores.position_bin.tolist() == [0, 0, 0, 0, 1, 1]
    assert scores.sparse_score[3] == np.float32(0.8)
    assert scores.broad_score[3] == np.float32(0.8)
    assert scores.supporting_sites[3] == 4
