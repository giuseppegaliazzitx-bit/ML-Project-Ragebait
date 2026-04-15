import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.config import load_settings
from ragebait_detector.labeling.vllm_labeler import label_csv_with_vllm
from ragebait_detector.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label compiled tweets with vLLM using guided JSON decoding"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--enable-random",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--model")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--temperature", type=float)
    return parser


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    if args.model:
        settings.vllm.model = args.model
    if args.quantization:
        settings.vllm.quantization = args.quantization
    if args.gpu_memory_utilization is not None:
        settings.vllm.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        settings.vllm.max_model_len = args.max_model_len
    if args.temperature is not None:
        settings.vllm.temperature = args.temperature
    summary = label_csv_with_vllm(
        input_path=args.input or settings.paths.unlabeled_posts_path,
        output_path=args.output or settings.paths.vllm_labels_path,
        summary_path=args.summary
        or Path(settings.paths.output_dir) / "vllm_labeling_summary.json",
        settings=settings,
        limit=args.limit,
        random_seed=args.random_seed,
        enable_random=args.enable_random,
    )
    print(summary)
