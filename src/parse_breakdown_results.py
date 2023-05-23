import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--size", nargs=2, type=float)
    parser.add_argument("--save_dir", "--save", type=Path)
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


def plot(data, title=None, size=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot()

    cmap = plt.get_cmap("tab20").colors
    cmap = cmap[::2] + cmap[1::2]

    for i, dat in enumerate(data):
        latency_data = dat["Latency (generate breakdown)"]
        ax.plot(
            [int(k) for k in latency_data.keys()],
            [v * 1000 for v in latency_data.values()],
            label=dat["Setting"],
            linewidth=1,
            color=cmap[i],
        )  # , linestyle=":")#, markersize=1, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    return fig


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    data = [read_data(input_file) for input_file in args.input_dir.iterdir()]

    if len(data) == 0:
        raise RuntimeError(f"No data to show.")

    title = args.title
    dirname = args.input_dir.stem
    if title is None:
        try:
            name, _, bs, _, _, _, _, step, cycles = dirname.rsplit("_", 8)
            title = f"{name}, bs = {bs} (s={step}, c={cycles})"
        except ValueError:
            title = dirname

    fig = plot(data, title, args.size)
    fig.show()
    if args.save_dir:
        save_path = (args.save_dir / dirname).with_suffix(".jpg")
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

    input("Press enter to continue")


if __name__ == "__main__":
    main()
