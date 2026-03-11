"""
test_utils.py
-------------
Unit tests for src/utils modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import ConfigDict, load_config, merge_configs, save_config
from src.utils.helper_functions import (
    compute_iou,
    draw_detections,
    generate_color_palette,
    normalize_box,
    resize_image,
    set_seed,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


# ─── ConfigDict ──────────────────────────────────────────────────────────────

class TestConfigDict:
    def test_attribute_access(self):
        cfg = ConfigDict({"training": {"epochs": 50}})
        assert cfg.training.epochs == 50

    def test_missing_key_raises(self):
        cfg = ConfigDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = cfg.missing_key

    def test_set_attribute(self):
        cfg = ConfigDict({})
        cfg.lr = 0.001
        assert cfg["lr"] == 0.001

    def test_get_nested(self):
        cfg = ConfigDict({"a": {"b": {"c": 42}}})
        assert cfg.get_nested("a", "b", "c") == 42
        assert cfg.get_nested("a", "b", "x", default="none") == "none"

    def test_merge_configs(self):
        base     = {"lr": 0.01, "epochs": 10}
        override = {"epochs": 50, "batch": 16}
        merged   = merge_configs(base, override)
        assert merged["epochs"] == 50
        assert merged["lr"] == 0.01
        assert merged["batch"] == 16


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        yaml_content = "training:\n  epochs: 100\n  lr: 0.001\n"
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        assert cfg.training.epochs == 100

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_overrides(self, tmp_path):
        yaml_content = "training:\n  epochs: 100\n"
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file, overrides={"training.epochs": 200})
        assert cfg.training.epochs == 200

    def test_save_and_reload(self, tmp_path):
        cfg = ConfigDict({"model": {"arch": "yolov8m"}, "lr": 0.001})
        out = tmp_path / "saved.yaml"
        save_config(cfg, out)
        loaded = load_config(out)
        assert loaded.model.arch == "yolov8m"
        assert loaded["lr"] == 0.001


# ─── Bounding Box Helpers ─────────────────────────────────────────────────────

class TestBBoxHelpers:
    def test_xyxy_to_xywh(self):
        box = [10.0, 20.0, 110.0, 170.0]
        xywh = xyxy_to_xywh(box)
        assert xywh == [10.0, 20.0, 100.0, 150.0]

    def test_xywh_to_xyxy(self):
        box = [10.0, 20.0, 100.0, 150.0]
        xyxy = xywh_to_xyxy(box)
        assert xyxy == [10.0, 20.0, 110.0, 170.0]

    def test_roundtrip(self):
        original = [50.0, 30.0, 200.0, 180.0]
        assert xywh_to_xyxy(xyxy_to_xywh(original)) == pytest.approx(original)

    def test_normalize_box(self):
        box = [100.0, 50.0, 300.0, 200.0]
        norm = normalize_box(box, img_w=640, img_h=480)
        assert norm == pytest.approx([100/640, 50/480, 300/640, 200/480])

    def test_compute_iou_perfect(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 5.0, 15.0, 15.0]
        iou = compute_iou(box1, box2)
        assert 0.0 < iou < 1.0


# ─── Image Helpers ────────────────────────────────────────────────────────────

class TestImageHelpers:
    def test_resize_image_no_aspect(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_image(img, (320, 320), keep_aspect=False)
        assert resized.shape == (320, 320, 3)

    def test_resize_image_keep_aspect(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_image(img, (320, 320), keep_aspect=True)
        assert resized.shape == (320, 320, 3)

    def test_generate_color_palette(self):
        colors = generate_color_palette(8)
        assert len(colors) == 8
        for c in colors:
            assert len(c) == 3

    def test_draw_detections_no_crash(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes  = [[10.0, 10.0, 100.0, 100.0]]
        labels = ["car"]
        confs  = [0.9]
        result = draw_detections(img, boxes, labels, confs)
        assert result.shape == img.shape

    def test_set_seed_reproducible(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)
