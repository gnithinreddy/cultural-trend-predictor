"""
Cultural Trend Predictor - main entry point.
Manages all pipelines via check → collect → process → dashboard.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_run(skip_trend: bool = False) -> int:
    """Launch app: run check pipeline (raw → process), then Streamlit dashboard."""
    from pipeline.check import ensure_raw_data_ready, ensure_processed_data_ready

    logger.info("Running check pipeline...")
    ensure_raw_data_ready()
    if not ensure_processed_data_ready(skip_trend=skip_trend):
        logger.error("Processed data not ready. Fix errors above before launching dashboard.")
        return 1

    logger.info("Launching dashboard...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "views/dashboard.py", "--server.headless", "true"],
            check=False,
        )
        return result.returncode
    except FileNotFoundError:
        logger.error("Streamlit not found. Install: pip install streamlit")
        return 1


def cmd_collect() -> int:
    """Run data collection only (skip check)."""
    from controllers.collect_controller import run_collect
    return run_collect()


def cmd_process(skip_trend: bool = False) -> int:
    """Run process only (clean + classify + trend)."""
    from controllers.process_controller import run_process
    return run_process(skip_trend=skip_trend)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cultural Trend Predictor")
    parser.add_argument("--skip-trend", action="store_true", help="Skip pytrends (with run)")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    p_run = subparsers.add_parser("run", help="Check pipeline + launch dashboard (default)")
    p_run.add_argument("--skip-trend", action="store_true", help="Skip pytrends (faster)")
    p_run.set_defaults(func=lambda a: cmd_run(skip_trend=a.skip_trend))

    p_collect = subparsers.add_parser("collect", help="Collect raw data only")
    p_collect.set_defaults(func=lambda _: cmd_collect())

    p_process = subparsers.add_parser("process", help="Process raw data (clean + classify + trend)")
    p_process.add_argument("--skip-trend", action="store_true", help="Skip trend scoring (faster, no pytrends)")
    p_process.set_defaults(func=lambda a: cmd_process(skip_trend=a.skip_trend))

    args = parser.parse_args()
    if args.command is None:
        return cmd_run(skip_trend=args.skip_trend)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
