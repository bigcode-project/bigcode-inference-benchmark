import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from src.metrics import Metrics
from src.utils import parse_config_args


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--filter", action="append")
    parser.add_argument("--column", "--col", action="append")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-x", "--x_axis", default=Metrics.BATCH_SIZE)
    parser.add_argument("-y", "--y_axis", default=Metrics.THROUGHPUT_E2E)
    parser.add_argument("-z", "--z_axis")
    return parser


DEFAULT_COLUMNS = (
    "Setting",
    Metrics.BATCH_SIZE,
    Metrics.INPUT_LENGTH,
    Metrics.TOKENS_SAMPLE,
    Metrics.THROUGHPUT_E2E,
    Metrics.LATENCY_E2E,
)


def read_data(input_file: Path):
    try:
        with input_file.open("r") as f:
            data = json.load(f)
            data = {**data["config"], **data["results"]}
    except (ValueError, OSError) as e:
        raise ValueError(f"Cannot parse file {input_file} ({e})")
    try:
        setting, bs_, bs, seq_, seq, tok_, tok = input_file.stem.rsplit("_", 6)
        assert bs_ == "bs"
        assert data[Metrics.BATCH_SIZE] == int(bs)
        assert seq_ == "seq"
        assert data[Metrics.INPUT_LENGTH] == int(seq) or int(seq) < 0
        assert tok_ == "tok"
        assert data[Metrics.TOKENS_SAMPLE] == int(tok)
    except (ValueError, AssertionError) as e:
        raise ValueError(f"Cannot parse filename {input_file} ({e})")
    data["Setting"] = setting
    return data


def make_table(data, cols):
    from markdownTable import markdownTable

    data = [Metrics.format_metrics({col: x[col] for col in cols}) for x in data]
    return markdownTable(data).getMarkdown()


def parse_key(key: Optional[str]) -> Optional[str]:
    if key is None:
        return key
    return getattr(Metrics, key.upper(), key)


def filter_data(data, filters):
    if filters is None:
        return data
    filters = parse_config_args(filters)
    filters = {parse_key(key): value for key, value in filters.items()}
    filtered_data = []
    for x in data:
        filter = True
        for key, value in filters.items():
            filter = filter and x[key] == value
        if filter:
            filtered_data.append(x)
    return filtered_data


def plot(data, x_axis, y_axis, z_axis):
    import matplotlib.pyplot as plt

    x_axis = parse_key(x_axis)
    y_axis = parse_key(y_axis)
    z_axis = parse_key(z_axis)
    x = [d[x_axis] for d in data]
    y = [d[y_axis] for d in data]
    z = None if z_axis is None else [d[z_axis] for d in data]

    fig = plt.figure()
    ax = fig.add_subplot()

    scatter = ax.scatter(x, y, c=z)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    if z_axis is not None:
        handles, labels = scatter.legend_elements()
        ax.legend(handles=handles, labels=labels, title=z_axis)
    fig.show()
    input("Press enter to continue")


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    data = [read_data(input_file) for input_file in args.input_dir.iterdir()]

    data = filter_data(data, args.filter)

    if len(data) == 0:
        raise RuntimeError(f"No data to show.")

    cols = DEFAULT_COLUMNS if args.column is None else [parse_key(col) for col in args.column]

    if args.table:
        print(make_table(data, cols))

    if args.plot:
        plot(data, args.x_axis, args.y_axis, args.z_axis)


if __name__ == "__main__":
    main()
