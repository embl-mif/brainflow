"""Candidate detection with cellfinder on the synthetic brain (runs on the CPU)."""

import numpy as np

import brainflow as bf
from conftest import BLOBS


def test_find_candidates_finds_the_blobs(project):
    brain = bf.open_brain("B1", project)
    cfos = bf.load_into_memory(bf.get_channel(brain, "cFos"))
    candidates = bf.find_candidates(cfos, bf.voxel_size(brain), project)
    found = candidates[["z", "y", "x"]].to_numpy(float)
    for blob in BLOBS:
        assert np.min(np.linalg.norm(found - blob, axis=1)) < 4, f"blob {blob} not found"


def test_prepare_region_makes_files_once(project):
    assert not bf.region_ready("B1", "regionA", project)
    signal, background, size, candidates = bf.prepare_region("B1", "regionA", project)
    assert bf.region_ready("B1", "regionA", project)
    assert signal.shape == background.shape and len(candidates) >= len(BLOBS)
    again = bf.prepare_region("B1", "regionA", project)[3]
    assert again.equals(candidates)
