import argparse
import json
import math
from collections import Counter
from typing import List

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ResultsAnalyzer",
        description="Analyze results by loading json file and plotting results",
    )
    parser.add_argument("--in_json_file", help="Json file to analyze results")
    parser.add_argument(
        "--MA_window_size",
        type=int,
        default=500,
        help="Moving average window size",
    )

    return parser.parse_args()


def moving_avg(values: List[float], window_size: int) -> List[float]:
    avgs: List[float] = []
    for beg in range(0, len(values), window_size):
        avgs.append(sum(values[beg : beg + window_size]) / window_size)

    return avgs


def plot_max_grids_statistics(max_grids: List[float]):
    counter = Counter(max_grids)
    log2_keys = [int(math.log2(k)) for k in counter]
    barchart = plt.bar(log2_keys, counter.values())
    plt.title("max_grids statistics (%)")
    ax = plt.gca()
    ax.set_xticks(log2_keys, (f"{2 ** k}" for k in log2_keys))

    N = len(max_grids)
    for bar in barchart:
        height = bar.get_height()
        ax.annotate(
            f"{height} ({int(round(height / N * 100))}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 1),  # vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Max grid")
    plt.ylabel("Count (ratio)")
    plt.show()


def main(in_file_json: str, window_size: int):
    with open(in_file_json) as fid:
        data = json.load(fid)

    for k, v in data.items():
        if len(v) == 0:
            continue
        print(f"Key: {k}, data size: {len(v)}")
        plt.scatter(list(range(len(v))), v, marker=".", alpha=0.2, s=5.0)
        avgs_y = moving_avg(v, window_size)
        avgs_x = [i * window_size for i in range(1, len(avgs_y) + 1)]
        plt.plot(avgs_x, avgs_y, color="red")
        plt.title(k)
        plt.show()
        if k == "max_grids":
            plot_max_grids_statistics(data["max_grids"])
        else:
            plt.hist(v)
            plt.title(f"{k} histogram")
            plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.in_json_file, args.MA_window_size)
