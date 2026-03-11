"""
onnx_export.py
--------------
Export a trained YOLOv8 model to ONNX format with optional simplification
and dynamic batch-size support.

Benchmark comparison (FP32 vs FP16 ONNX Runtime vs TorchScript) is also
provided.

Usage:
    python deployment/onnx_export.py \
        --weights models/weights/best.pt \
        --output models/weights/best.onnx \
        --opset 17 \
        --simplify
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Exporter ────────────────────────────────────────────────────────────────

class ONNXExporter:
    """
    Handles ONNX export and inference benchmarking.

    Args:
        weights:     Path to PyTorch ``.pt`` weights.
        output_path: Destination for the ``.onnx`` file.
        img_size:    Input image size (square).
        opset:       ONNX opset version.
        simplify:    Run onnx-simplifier after export.
        dynamic:     Export with dynamic batch axis.
        half:        Export in FP16 mode.
    """

    def __init__(
        self,
        weights: str,
        output_path: Optional[str] = None,
        img_size: int = 640,
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
        half: bool = False,
    ):
        self.weights     = Path(weights)
        self.output_path = Path(output_path) if output_path else self.weights.with_suffix(".onnx")
        self.img_size    = img_size
        self.opset       = opset
        self.simplify    = simplify
        self.dynamic     = dynamic
        self.half        = half

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self) -> Path:
        """
        Export to ONNX and optionally simplify.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        from ultralytics import YOLO

        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {self.weights.name} → ONNX")
        logger.info(
            f"  opset={self.opset} | simplify={self.simplify} | "
            f"dynamic={self.dynamic} | half={self.half}"
        )

        model = YOLO(str(self.weights))
        exported = model.export(
            format="onnx",
            imgsz=self.img_size,
            opset=self.opset,
            simplify=self.simplify,
            dynamic=self.dynamic,
            half=self.half,
        )

        # Ultralytics saves beside the .pt; move to desired output path
        exported_path = Path(exported) if isinstance(exported, str) else exported
        if exported_path != self.output_path:
            import shutil
            shutil.move(str(exported_path), str(self.output_path))

        logger.info(f"ONNX model saved: {self.output_path}")
        self._validate_onnx()
        return self.output_path

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_onnx(self) -> None:
        """Check ONNX model validity using onnx.checker."""
        try:
            import onnx
            model_proto = onnx.load(str(self.output_path))
            onnx.checker.check_model(model_proto)
            logger.info("ONNX model validation: PASSED")

            # Print I/O info
            for inp in model_proto.graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                logger.info(f"  Input  — name={inp.name}  shape={shape}")
            for out in model_proto.graph.output:
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                logger.info(f"  Output — name={out.name}  shape={shape}")

        except ImportError:
            logger.warning("onnx package not installed — skipping validation.")
        except Exception as exc:
            logger.error(f"ONNX validation failed: {exc}")

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, n_runs: int = 100) -> dict:
        """
        Benchmark ONNX Runtime inference speed vs PyTorch.

        Args:
            n_runs: Number of inference passes.

        Returns:
            Dict with ``ort_avg_ms`` and ``torch_avg_ms``.
        """
        import torch
        from ultralytics import YOLO

        dummy = np.random.rand(1, 3, self.img_size, self.img_size).astype(np.float32)

        # PyTorch benchmark
        pt_model = YOLO(str(self.weights))
        pt_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            pt_model.predict(dummy[0].transpose(1, 2, 0), verbose=False)
            pt_times.append((time.perf_counter() - t0) * 1000)
        torch_avg = float(np.mean(pt_times[10:]))  # skip warmup

        # ONNX Runtime benchmark
        ort_avg = self._benchmark_ort(dummy, n_runs)

        result = {
            "pytorch_avg_ms":  round(torch_avg, 2),
            "ort_avg_ms":      round(ort_avg, 2) if ort_avg else None,
            "pytorch_fps":     round(1000 / torch_avg, 1),
            "ort_fps":         round(1000 / ort_avg, 1) if ort_avg else None,
            "speedup_x":       round(torch_avg / ort_avg, 2) if ort_avg else None,
        }

        logger.info("=" * 50)
        logger.info("Benchmark Results")
        logger.info("=" * 50)
        for k, v in result.items():
            logger.info(f"  {k:<25} {v}")

        return result

    def _benchmark_ort(self, dummy: np.ndarray, n_runs: int) -> Optional[float]:
        """Run ONNX Runtime inference and return average ms."""
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(self.output_path), providers=providers)
            input_name = sess.get_inputs()[0].name

            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                sess.run(None, {input_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)

            return float(np.mean(times[10:]))

        except ImportError:
            logger.warning("onnxruntime not installed — skipping ORT benchmark.")
            return None
        except Exception as exc:
            logger.warning(f"ORT benchmark failed: {exc}")
            return None


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights",  default="models/weights/best.pt")
    parser.add_argument("--output",   default=None,
                        help="Output .onnx path (default: alongside weights).")
    parser.add_argument("--config",   default="configs/training_config.yaml")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--opset",    type=int, default=None)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--dynamic",  action="store_true")
    parser.add_argument("--half",     action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark after export.")
    parser.add_argument("--bench-runs", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    deploy_cfg = cfg.deployment.get("onnx", {})

    exporter = ONNXExporter(
        weights=args.weights,
        output_path=args.output or deploy_cfg.get("output_path"),
        img_size=args.img_size or cfg.model.get("input_size", 640),
        opset=args.opset or deploy_cfg.get("opset", 17),
        simplify=args.simplify,
        dynamic=args.dynamic or deploy_cfg.get("dynamic", False),
        half=args.half,
    )

    exporter.export()

    if args.benchmark:
        exporter.benchmark(n_runs=args.bench_runs)


if __name__ == "__main__":
    main()
