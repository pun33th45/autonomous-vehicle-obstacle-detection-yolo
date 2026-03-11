"""
conftest.py
-----------
Shared pytest fixtures.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Return a 640×640 BGR NumPy image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_boxes():
    """Return a list of sample bounding boxes in [x1, y1, x2, y2] format."""
    return [
        [10.0, 20.0, 100.0, 150.0],
        [200.0, 50.0, 400.0, 300.0],
        [500.0, 100.0, 620.0, 580.0],
    ]


@pytest.fixture
def sample_labels():
    return ["car", "pedestrian", "truck"]


@pytest.fixture
def sample_confidences():
    return [0.92, 0.87, 0.75]
