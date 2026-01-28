import argparse
import multiprocessing as mp


def main() -> int:
    # Must run before creating any Pool / ProcessPoolExecutor
    # and ideally before importing modules that load native libs.
    mp.set_start_method("spawn", force=True)

    from experiments.config import load_config
    from experiments.runner import run_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    return run_all(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
