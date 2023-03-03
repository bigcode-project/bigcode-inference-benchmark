import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from src.metrics import Metrics
from src.utils import parse_config_args, parse_config_arg


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--filter", action="append")
    parser.add_argument("--column", "--col", action="append")
    parser.add_argument("--compare_value")
    parser.add_argument("--compare_col", default="Setting")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-x", "--x_axis", default=Metrics.BATCH_SIZE)
    parser.add_argument("-y", "--y_axis", default=Metrics.THROUGHPUT_E2E)
    parser.add_argument("-z", "--z_axis")
    parser.add_argument("--title")
    return parser


DEFAULT_COLUMNS = (
    "Setting",
    Metrics.INPUT_LENGTH,
    Metrics.TOKENS_SAMPLE,
    Metrics.BATCH_SIZE,
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


def parse_key(key: Optional[str]) -> Optional[str]:
    if key is None:
        return key
    return getattr(Metrics, key.upper(), key)


def make_table(data, cols):
    from markdownTable import markdownTable

    data = [Metrics.format_metrics({col: x[col] for col in cols}) for x in data]
    return markdownTable(data).getMarkdown()


def make_compare_table(data, cols, compare_value, compare_col):
    from markdownTable import markdownTable

    compare_value = parse_key(compare_value)
    compare_col = parse_key(compare_col)
    compare_data = {}
    all_compare_index = set()
    # Aggregate by the cols entries, then map compare_key to compare
    for x in data:
        index = tuple(x[col] for col in cols)
        if index not in compare_data:
            compare_data[index] = {}
        compare_index = x[compare_col]
        all_compare_index.add(compare_index)
        if compare_index in compare_data[index]:
            print(f"Duplicate entry {compare_index} for index {index}")
        compare_data[index][compare_index] = Metrics.format_metric(compare_value, x[compare_value])

    table_data = []
    for index in sorted(compare_data):
        # Merge the index and values
        table_data.append(
            {
                **Metrics.format_metrics({col: v for col, v in zip(cols, index)}),
                **{
                    compare_index: compare_data[index].get(compare_index, "N.A.")
                    for compare_index in sorted(all_compare_index)
                },
            }
        )

    return markdownTable(table_data).getMarkdown()


def filter_data(data, filters):
    if filters is None:
        return data

    parsed_filters = {}
    for filter in filters:
        key, value = parse_config_arg(filter)
        key = parse_key(key)
        if key not in parsed_filters:
            parsed_filters[key] = []
        parsed_filters[key].append(value)

    filtered_data = []
    for x in data:
        filter = True
        for key, value in parsed_filters.items():
            filter = filter and x[key] in value
        if filter:
            filtered_data.append(x)
    return filtered_data


def plot(data, x_axis, y_axis, z_axis, title=None):
    import matplotlib.pyplot as plt

    x_axis = parse_key(x_axis)
    y_axis = parse_key(y_axis)
    z_axis = parse_key(z_axis)
    x = [d[x_axis] for d in data]
    y = [d[y_axis] for d in data]

    fig = plt.figure()
    ax = fig.add_subplot()

    # z = None if z_axis is None else [d[z_axis] for d in data]
    if z_axis is None:
        ax.scatter(x, y)
    else:
        z = [d[z_axis] for d in data]
        for z_value in set(z):
            xx, yy = tuple(zip(*sorted((x_, y_) for x_, y_, z_ in zip(x, y, z) if z_ == z_value)))
            ax.plot(xx, yy, label=z_value, linewidth=1, linestyle=":", markersize=4, marker="o")
            # ax.scatter(x,y, label=z_value)
        # handles, labels = scatter.legend_elements()
        ax.legend(loc="upper left")  # handles=handles, labels=labels, title=z_axis)

    ax.set_title(y_axis if title is None else title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
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
        if args.compare_value:
            print(make_compare_table(data, cols, args.compare_value, args.compare_col))
        else:
            print(make_table(data, cols))

    if args.plot:
        plot(data, args.x_axis, args.y_axis, args.z_axis, args.title)


if __name__ == "__main__":
    main()
