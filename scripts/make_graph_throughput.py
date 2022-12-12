import argparse

import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str)
    parser.add_argument("--plot", choices=["throughput", "inverse_throughput", "latency"])

    args = parser.parse_args()

    return args


def parse_line(line: str, plot: str = "throughput") -> str:
    line = line.strip()
    line = line[1:-1]
    line = line.split(" | ")

    line = [i.strip() for i in line]

    try:
        line[0] = int(line[0])
    except:
        pass

    if plot == "throughput":
        for i in range(1, len(line)):
            if "\|" in line[i]:
                line[i] = float(line[i].split("\|")[0])
    elif plot == "inverse_throughput":
        for i in range(1, len(line)):
            if "\|" in line[i]:
                line[i] = float(line[i].split("\|")[1])
    elif plot == "latency":
        for i in range(1, len(line)):
            try:
                line[i] = float(line[i])
            except:
                pass

    return line


def parse_data(data: list):
    x = []
    y = [[], [], [], []]
    for dp in data:
        x.append(dp[0])
        for i in range(1, len(dp)):
            if dp[i] != "oom":
                y[i - 1].append(dp[i])

    for i in y:
        yield x[: len(i)], i


def main() -> None:
    args = get_args()

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    column_names = parse_line(lines[0])
    data = [parse_line(line, args.plot) for line in lines[2:]]

    data = parse_data(data)

    c = 1
    for i in data:
        plt.plot(i[0], i[1], marker=".", label=column_names[c])
        c += 1

    plt.legend()
    plt.xlabel(column_names[0])

    if args.plot == "throughput":
        plt.ylabel(args.plot + " (tokens/sec)")
    elif args.plot == "inverse_throughput":
        plt.ylabel(args.plot + " (msecs/token)")
    if args.plot == "latency":
        plt.ylabel(args.plot + " (sec)")

    plt.savefig(args.plot, dpi=1200)


if __name__ == "__main__":
    main()
