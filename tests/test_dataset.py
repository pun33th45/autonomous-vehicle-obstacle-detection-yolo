"""
test_dataset.py
---------------
Unit tests for dataset preprocessing utilities.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.dataset.preprocess_dataset import (
    CLASS_NAMES,
    COCO_TO_PROJECT,
    split_dataset,
    write_dataset_yaml,
)


# ─── COCO Mapping ─────────────────────────────────────────────────────────────

class TestCocoMapping:
    def test_person_maps_to_pedestrian(self):
        assert COCO_TO_PROJECT[1] == 0  # person → pedestrian

    def test_car_maps_correctly(self):
        assert COCO_TO_PROJECT[3] == 2  # car → car

    def test_class_names_length(self):
        assert len(CLASS_NAMES) == 8


# ─── Dataset Split ────────────────────────────────────────────────────────────

class TestSplitDataset:
    @pytest.fixture
    def fake_dataset(self, tmp_path):
        """Create a synthetic dataset with 100 images and labels."""
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        paths = []
        for i in range(100):
            img_path = img_dir / f"img_{i:04d}.jpg"
            img_path.write_bytes(b"fake_img")
            lbl_path = lbl_dir / f"img_{i:04d}.txt"
            lbl_path.write_text("0 0.5 0.5 0.3 0.3")
            paths.append(img_path)

        return tmp_path, paths

    def test_split_counts(self, fake_dataset):
        root, paths = fake_dataset
        out_dir = root / "split_output"
        splits = split_dataset(paths, out_dir, train_ratio=0.8, val_ratio=0.1, seed=42)

        assert len(splits["train"]) == 80
        assert len(splits["val"])   == 10
        assert len(splits["test"])  == 10
        assert sum(len(v) for v in splits.values()) == 100

    def test_split_creates_dirs(self, fake_dataset):
        root, paths = fake_dataset
        out_dir = root / "split_dirs"
        split_dataset(paths, out_dir, train_ratio=0.8, val_ratio=0.1)

        for split in ("train", "val", "test"):
            assert (out_dir / "images" / split).exists()
            assert (out_dir / "labels" / split).exists()

    def test_reproducibility(self, fake_dataset):
        root, paths = fake_dataset
        out1 = root / "split1"
        out2 = root / "split2"
        s1 = split_dataset(paths, out1, seed=42)
        s2 = split_dataset(paths, out2, seed=42)
        assert [p.name for p in s1["train"]] == [p.name for p in s2["train"]]


# ─── Dataset YAML ─────────────────────────────────────────────────────────────

class TestWriteDatasetYaml:
    def test_yaml_content(self, tmp_path):
        yaml_path = write_dataset_yaml(tmp_path, num_classes=8)
        content = yaml_path.read_text()
        assert "nc: 8" in content
        assert "pedestrian" in content
        assert "car" in content
        assert "train:" in content
        assert "val:" in content

    def test_yaml_exists(self, tmp_path):
        yaml_path = write_dataset_yaml(tmp_path)
        assert yaml_path.exists()
