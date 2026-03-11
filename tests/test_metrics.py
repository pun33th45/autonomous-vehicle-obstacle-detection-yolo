"""
test_metrics.py
---------------
Unit tests for src/evaluation/metrics.py
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_ap,
    compute_iou_matrix,
    compute_map,
    compute_precision_recall_f1,
    match_detections_to_gt,
)


class TestIoUMatrix:
    def test_perfect_overlap(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        mat   = compute_iou_matrix(boxes, boxes)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_no_overlap(self):
        b1 = np.array([[0, 0, 10, 10]], dtype=float)
        b2 = np.array([[20, 20, 30, 30]], dtype=float)
        mat = compute_iou_matrix(b1, b2)
        assert mat[0, 0] == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = np.array([[0, 0, 10, 10]], dtype=float)
        b2 = np.array([[5, 5, 15, 15]], dtype=float)
        mat = compute_iou_matrix(b1, b2)
        # Intersection: 5x5=25, union: 100+100-25=175
        assert mat[0, 0] == pytest.approx(25 / 175, abs=1e-4)

    def test_empty_input(self):
        b1 = np.empty((0, 4), dtype=float)
        b2 = np.array([[0, 0, 10, 10]], dtype=float)
        mat = compute_iou_matrix(b1, b2)
        assert mat.shape == (0, 1)


class TestAP:
    def test_perfect_ap(self):
        recall    = np.linspace(0, 1, 10)
        precision = np.ones(10)
        ap = compute_ap(recall, precision)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_ap(self):
        recall    = np.linspace(0, 1, 10)
        precision = np.zeros(10)
        ap = compute_ap(recall, precision)
        assert ap == pytest.approx(0.0, abs=0.01)


class TestMatchDetections:
    def test_perfect_match(self):
        pred = np.array([[0, 0, 10, 10]], dtype=float)
        conf = np.array([0.9])
        gt   = np.array([[0, 0, 10, 10]], dtype=float)
        tp, fp = match_detections_to_gt(pred, conf, gt, iou_threshold=0.5)
        assert tp[0] == True
        assert fp[0] == False

    def test_no_match_low_iou(self):
        pred = np.array([[100, 100, 110, 110]], dtype=float)
        conf = np.array([0.9])
        gt   = np.array([[0, 0, 10, 10]], dtype=float)
        tp, fp = match_detections_to_gt(pred, conf, gt, iou_threshold=0.5)
        assert fp[0] == True

    def test_empty_gt(self):
        pred = np.array([[0, 0, 10, 10]], dtype=float)
        conf = np.array([0.9])
        gt   = np.empty((0, 4), dtype=float)
        tp, fp = match_detections_to_gt(pred, conf, gt, iou_threshold=0.5)
        assert fp[0] == True


class TestPRF1:
    def test_perfect(self):
        p, r, f = compute_precision_recall_f1(tp=10, fp=0, fn=0)
        assert p == pytest.approx(1.0, abs=1e-3)
        assert r == pytest.approx(1.0, abs=1e-3)
        assert f == pytest.approx(1.0, abs=1e-3)

    def test_all_fp(self):
        p, r, f = compute_precision_recall_f1(tp=0, fp=10, fn=5)
        assert p == pytest.approx(0.0, abs=1e-3)
        assert r == pytest.approx(0.0, abs=1e-3)

    def test_balanced(self):
        # tp=5, fp=5, fn=5
        p, r, f = compute_precision_recall_f1(tp=5, fp=5, fn=5)
        assert p == pytest.approx(0.5, abs=0.01)
        assert r == pytest.approx(0.5, abs=0.01)
        assert f == pytest.approx(0.5, abs=0.01)


class TestComputeMap:
    def _make_perfect_dataset(self):
        """Identical predictions and ground truths → mAP ≈ 1.0."""
        box = np.array([0, 0, 10, 10], dtype=float)
        dets = {0: [(box.copy(), 0.99)]}
        gts  = {0: [box.copy()]}
        return dets, gts

    def test_perfect_map50(self):
        dets, gts = self._make_perfect_dataset()
        results = compute_map(dets, gts, iou_thresholds=[0.5])
        assert results["mAP50"] == pytest.approx(1.0, abs=0.01)

    def test_empty_detections(self):
        gts = {0: [np.array([0, 0, 10, 10], dtype=float)]}
        dets = {}
        results = compute_map(dets, gts)
        assert results["mAP50"] == pytest.approx(0.0, abs=0.01)
