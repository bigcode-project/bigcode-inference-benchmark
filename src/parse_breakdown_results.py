import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--title")
    return parser


def read_data(input_file: Path):
    try:
        with input_file.open("r") as f:
            data = json.load(f)
            data = {**data["config"], **data["results"]}
    except (ValueError, OSError) as e:
        raise ValueError(f"Cannot parse file {input_file} ({e})")
    data["Setting"] = input_file.stem
    return data


def plot(data, title=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()

    for dat in data:
        latency_data = dat["Latency (generate breakdown)"]
        ax.plot(
            [int(k) for k in latency_data.keys()],
            [v * 1000 for v in latency_data.values()],
            label=dat["Setting"],
            linewidth=1,
        )  # , linestyle=":")#, markersize=1, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    fig.show()
    input("Press enter to continue")


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    data = [read_data(input_file) for input_file in args.input_dir.iterdir()]

    if len(data) == 0:
        raise RuntimeError(f"No data to show.")

    plot(data, args.title)


if __name__ == "__main__":
    main()
