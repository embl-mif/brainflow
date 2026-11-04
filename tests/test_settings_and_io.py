import json

import numpy as np
import pandas as pd
import pytest

import brainflow as bf
from conftest import SHAPE, VOXEL


def test_load_settings_fills_in_folders(project):
    results = project["results_folder"]
    assert project["networks_folder"].endswith("networks")
    assert project["regions_file"].endswith("regions.json")
    assert project["atlas"]["name"] == "allen_mouse_25um"
    assert project["atlas"]["folder"].startswith(results.rsplit("results", 1)[0])


def test_settings_found_from_environment(project, monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINFLOW_SETTINGS", project["settings_file"])
    monkeypatch.chdir(tmp_path.parent)
    assert bf.load_settings()["settings_file"] == project["settings_file"]


def test_missing_settings_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINFLOW_SETTINGS", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="brainflow init"):
        bf.load_settings()


def test_regions_saved_with_comments_kept(project):
    regions = bf.load_regions(project)
    assert list(regions) == ["B1"]
    regions["B1"]["regionB"] = {"z": [0, 10], "y": [0, 10], "x": [0, 10]}
    bf.save_regions(regions, project)
    raw = json.load(open(project["regions_file"]))
    assert raw["_comment"] == "test" and "regionB" in raw["B1"]


def test_write_templates(tmp_path):
    written = bf.write_templates(tmp_path / "new")
    assert sorted(p.name for p in written) == ["regions.json", "settings.json"]
    assert bf.write_templates(tmp_path / "new") == []  # existing files are kept


def test_open_brain_and_channels(project):
    assert bf.brain_names(project) == ["B1"]
    brain = bf.open_brain("B1", project)
    assert bf.voxel_size(brain) == VOXEL
    cfos = bf.load_into_memory(bf.get_channel(brain, project["signal_channel"]))
    assert cfos.shape == SHAPE and cfos.dtype == np.uint16
    with pytest.raises(ValueError):
        bf.get_channel(brain, "missing-channel")


def test_save_region(project, tmp_path):
    brain = bf.open_brain("B1", project)
    region = bf.save_region(brain, (5, 15), (10, 30), (20, 60), tmp_path / "r.ome.zarr")
    cfos = bf.load_into_memory(bf.get_channel(region, "cFos"))
    full = bf.load_into_memory(bf.get_channel(brain, "cFos"))
    np.testing.assert_array_equal(cfos, full[5:15, 10:30, 20:60])


def test_density_map(project, tmp_path):
    brain = bf.open_brain("B1", project)
    cells = pd.DataFrame({"z": [1, 2, 30], "y": [1, 2, 100], "x": [1, 2, 100]})
    density = bf.save_density_map(cells, brain, tmp_path / "d.ome.zarr", bin_um=30)
    assert density.sum() == 3 and density.max() == 2


def test_outputs_can_be_written_again(project, tmp_path):
    """Running a step twice replaces its output (zarrsmith refuses to, unless asked)."""
    brain = bf.open_brain("B1", project)
    cells = pd.DataFrame({"z": [1, 2], "y": [1, 2], "x": [1, 2]})
    assert bf.save_density_map(cells, brain, tmp_path / "d.ome.zarr", bin_um=30).sum() == 2
    assert bf.save_density_map(cells.iloc[:1], brain, tmp_path / "d.ome.zarr", bin_um=30).sum() == 1

    bf.save_region(brain, (0, 10), (0, 10), (0, 10), tmp_path / "r.ome.zarr")
    region = bf.save_region(brain, (0, 5), (0, 5), (0, 5), tmp_path / "r.ome.zarr")
    assert bf.get_channel(region, "cFos").shape == (5, 5, 5)

    bf.save_image(np.zeros((4, 4, 4), np.uint16), (1, 1, 1), tmp_path / "i.ome.zarr", ["a"])
    image = bf.save_image(np.ones((6, 6, 6), np.uint16), (1, 1, 1), tmp_path / "i.ome.zarr", ["a"])
    assert image.data[0].shape[-3:] == (6, 6, 6)
