from curses import window
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import List
def parse_args():
    parser = argparse.ArgumentParser(
        prog="ResultsAnalyzer", description="Analyze results by loading json file and plotting results"
    )
    parser.add_argument(
        "--in_json_file", help="Json file to analyze results"
    )
    parser.add_argument(
        "--MA_window_size", type=int, default=500, help="Moving average window size"
    )

    return parser.parse_args()

def moving_avg(values: List[float], window_size: int) -> List[float]:
    avgs: List[float] = []
    for beg in range(0, len(values), window_size):
        avgs.append(sum(values[beg:beg+window_size]) / window_size)
    
    return avgs

def main(in_file_json: str, window_size: int):
    with open(in_file_json, "rt") as fid:
        data = json.load(fid)
    
    for k, v in data.items():
        print(f"Key: {k}, data size: {len(v)}")
        plt.scatter(list(range(len(v))), v, marker=".", alpha=0.2)
        avgs_y = moving_avg(v, window_size)
        avgs_x = [i * window_size for i in range(1, len(avgs_y) + 1)]
        plt.plot(avgs_x, avgs_y)
        plt.title(k)
        plt.show()
    
if __name__ == "__main__":
    args = parse_args()
    main(args.in_json_file, args.MA_window_size)