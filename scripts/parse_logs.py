import argparse
import copy
import os
from typing import Tuple

from markdownTable import markdownTable
from pandas import DataFrame


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def parse_line(line: str) -> Tuple[str, str]:
    line = line.strip()

    if line.endswith("tokens/sec"):
        line = line.split("Throughput (including tokenization) = ")[1]
        line = line.split(" tokens/sec")[0]

        return line, "throughput"
    elif line.endswith("msecs/token"):
        line = line.split("Throughput (including tokenization) = ")[1]
        line = line.split(" msecs/token")[0]

        return line, "inverse_throughput"
    elif line.startswith("Latency = ") and line.endswith("secs"):
        line = line.split("Latency = ")[1]
        line = line.split(" secs")[0]

        return line, "latency"
    elif "with batch size = " in line:
        line = line.split("with batch size = ")[1]

        return line, "batch_size"

    return None, None


def get_throughput_dataframe(results: dict, order: list) -> DataFrame:
    throughput = copy.deepcopy(results["throughput"])
    for key in results["inverse_throughput"]:
        for index, value in enumerate(results["inverse_throughput"][key]):
            throughput[key][index] = throughput[key][index] + " \| " + value

    max_rows = -1
    batch_size_column = None
    for key in results["batch_size"]:
        bs = len(results["batch_size"][key])

        if bs > max_rows:
            max_rows = bs
            batch_size_column = results["batch_size"][key]

    for key in throughput:
        while len(throughput[key]) < max_rows:
            throughput[key].append("oom")
    throughput["batch_size"] = batch_size_column

    df = DataFrame(throughput)
    df = df.loc[:, order]

    return df


def get_latency_dataframe(results: dict, order: list) -> DataFrame:
    latency = copy.deepcopy(results["latency"])

    max_rows = -1
    batch_size_column = None
    for key in results["batch_size"]:
        bs = len(results["batch_size"][key])

        if bs > max_rows:
            max_rows = bs
            batch_size_column = results["batch_size"][key]

    for key in latency:
        while len(latency[key]) < max_rows:
            latency[key].append("oom")
    latency["batch_size"] = batch_size_column

    df = DataFrame(latency)
    df = df.loc[:, order]

    return df


def make_table(results: dict):
    order = ["batch_size", "HF (fp32)", "HF (bf16)", "HF (int8)"]

    kwargs = dict(
        row_sep="markdown",
        padding_width=1,
    )

    throughput = get_throughput_dataframe(results, order)
    throughput = throughput.to_dict(orient="records")
    throughput = markdownTable(throughput).setParams(**kwargs).getMarkdown().split("```")[1]

    latency = get_latency_dataframe(results, order)
    latency = latency.to_dict(orient="records")
    latency = markdownTable(latency).setParams(**kwargs).getMarkdown().split("```")[1]

    return throughput, latency


def main() -> None:
    args = get_args()

    input_files = os.listdir(args.input_dir)
    results = {"throughput": {}, "inverse_throughput": {}, "latency": {}, "batch_size": {}}
    filename_column = {
        "fp32.log": "HF (fp32)",
        "bf16.log": "HF (bf16)",
        "int8.log": "HF (int8)",
        "fp16.log": "DS-inference (fp16)",
    }

    for filename in input_files:
        with open(os.path.join(args.input_dir, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            value, key = parse_line(line)

            if key is not None:
                column_name = filename_column[filename]
                if column_name not in results[key]:
                    results[key][column_name] = []
                results[key][column_name].append(value)

    throughput, latency = make_table(results)

    print("Throughput (tokens/sec | msec/token)")
    print(throughput)
    print()
    print("Latency (sec)")
    print(latency)


if __name__ == "__main__":
    main()
