import argparse
from experiments.config import load_config
from experiments.runner import run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raise SystemExit(run_all(cfg))


if __name__ == "__main__":
    main()
