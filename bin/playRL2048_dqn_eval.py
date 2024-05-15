import os
import sys

from rl_2048 import __file__ as rl_2048_file
from rl_2048.bin.playRL2048_dqn import main as dqn_main

BEST_MODEL_NAME: str = "model_E"


def get_pretrained_dir() -> str:
    root_path: str = os.path.dirname(rl_2048_file)
    return os.path.join(root_path, "pretrained_model")


def main():
    model_path: str = os.path.join(get_pretrained_dir(), f"{BEST_MODEL_NAME}.pth")

    sys.argv.extend(
        [
            "--max_iters",
            "-1",  # Can be executed for infinite times
            "--output_json_prefix",
            "./rl_2048_dqn_eval_results",
            "--eval",  # evaluation mode
            "--show_board",
            "--network_version",
            "layers_512_512_residual_0_128",
            "--trained_net_path",
            model_path,
        ]
    )
    breakpoint()
    dqn_main()


if __name__ == "__main__":
    main()
