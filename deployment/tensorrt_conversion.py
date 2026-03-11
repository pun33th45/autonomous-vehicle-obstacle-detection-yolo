"""
tensorrt_conversion.py
----------------------
Convert a trained YOLOv8 model to TensorRT engine format for
maximum GPU inference performance.

Supports:
  - FP32 / FP16 / INT8 precision
  - Dynamic batch size
  - Calibration for INT8 quantisation

Prerequisites:
  - NVIDIA GPU with TensorRT installed
  - tensorrt Python bindings (pip install tensorrt)

Usage:
    python deployment/tensorrt_conversion.py \
        --weights models/weights/best.pt \
        --output models/weights/best.engine \
        --fp16
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── TensorRT Converter ───────────────────────────────────────────────────────

class TensorRTConverter:
    """
    Converts YOLOv8 PyTorch weights to a TensorRT engine via Ultralytics.

    Args:
        weights:     Path to ``.pt`` weights file.
        output_path: Path for the ``.engine`` output.
        img_size:    Input image size.
        workspace:   TensorRT builder workspace in GB.
        fp16:        FP16 precision (recommended).
        int8:        INT8 precision (requires calibration data).
        dynamic:     Dynamic batch size.
        device:      GPU device index.
    """

    def __init__(
        self,
        weights: str,
        output_path: Optional[str] = None,
        img_size: int = 640,
        workspace: int = 4,
        fp16: bool = True,
        int8: bool = False,
        dynamic: bool = False,
        device: str = "0",
    ):
        self.weights     = Path(weights)
        self.output_path = Path(output_path) if output_path else self.weights.with_suffix(".engine")
        self.img_size    = img_size
        self.workspace   = workspace
        self.fp16        = fp16
        self.int8        = int8
        self.dynamic     = dynamic
        self.device      = device

    def convert(self) -> Path:
        """
        Export and build the TensorRT engine.

        Returns:
            Path to the ``.engine`` file.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required: pip install ultralytics")

        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        precision = "int8" if self.int8 else ("fp16" if self.fp16 else "fp32")
        logger.info(
            f"Building TensorRT engine | precision={precision} | "
            f"workspace={self.workspace}GB | dynamic={self.dynamic}"
        )

        model = YOLO(str(self.weights))
        exported = model.export(
            format="engine",
            imgsz=self.img_size,
            half=self.fp16,
            int8=self.int8,
            dynamic=self.dynamic,
            workspace=self.workspace,
            device=self.device,
        )

        # Move to desired output path if needed
        exported_path = Path(exported) if isinstance(exported, str) else exported
        if exported_path != self.output_path:
            import shutil
            shutil.move(str(exported_path), str(self.output_path))

        logger.info(f"TensorRT engine saved: {self.output_path}")
        return self.output_path

    def benchmark(
        self, n_runs: int = 200, warmup: int = 20
    ) -> dict:
        """
        Benchmark TensorRT engine inference latency.

        Args:
            n_runs:  Number of inference runs after warmup.
            warmup:  Number of warmup runs.

        Returns:
            Dict with latency statistics.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed.")
            return {}

        if not self.output_path.exists():
            logger.error(f"Engine not found: {self.output_path}")
            return {}

        dummy_img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        model = YOLO(str(self.output_path))

        logger.info(f"Benchmarking {self.output_path.name} ({warmup} warmup + {n_runs} runs)...")

        times = []
        for i in range(warmup + n_runs):
            t0 = time.perf_counter()
            model.predict(dummy_img, verbose=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                times.append(elapsed_ms)

        avg_ms  = float(np.mean(times))
        min_ms  = float(np.min(times))
        max_ms  = float(np.max(times))
        p95_ms  = float(np.percentile(times, 95))
        fps     = 1000 / avg_ms

        result = {
            "engine":   str(self.output_path),
            "avg_ms":   round(avg_ms, 2),
            "min_ms":   round(min_ms, 2),
            "max_ms":   round(max_ms, 2),
            "p95_ms":   round(p95_ms, 2),
            "fps":      round(fps, 1),
        }

        logger.info("=" * 50)
        logger.info("TensorRT Benchmark Results")
        logger.info("=" * 50)
        for k, v in result.items():
            logger.info(f"  {k:<20} {v}")

        return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 model to TensorRT engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights",   default="models/weights/best.pt")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--config",    default="configs/training_config.yaml")
    parser.add_argument("--img-size",  type=int, default=None)
    parser.add_argument("--workspace", type=int, default=None,
                        help="Builder workspace in GB.")
    parser.add_argument("--fp16",      action="store_true", default=True)
    parser.add_argument("--no-fp16",   dest="fp16", action="store_false")
    parser.add_argument("--int8",      action="store_true")
    parser.add_argument("--dynamic",   action="store_true")
    parser.add_argument("--device",    default="0")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-runs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trt_cfg = cfg.deployment.get("tensorrt", {})

    converter = TensorRTConverter(
        weights=args.weights,
        output_path=args.output or trt_cfg.get("output_path"),
        img_size=args.img_size or cfg.model.get("input_size", 640),
        workspace=args.workspace or trt_cfg.get("workspace", 4),
        fp16=args.fp16,
        int8=args.int8,
        dynamic=args.dynamic,
        device=args.device,
    )

    converter.convert()

    if args.benchmark:
        converter.benchmark(n_runs=args.bench_runs)


if __name__ == "__main__":
    main()
