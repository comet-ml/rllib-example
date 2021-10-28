import comet_ml
import argparse
import os

import ray
from ray.rllib.utils.test_utils import check_learning_achieved
from ray import tune
from logger import CometLoggerCallback

API_KEY = os.getenv("COMET_API_KEY")
PROJECT_NAME = "rllib-test"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=4)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=10, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    config = {
        "env": "CartPole-v0",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": args.framework,
        # Run with tracing enabled for tfe/tf2.
        "eager_tracing": args.framework in ["tfe", "tf2"],
    }

    stop = {"training_iteration": args.stop_iters}

    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        verbose=2,
        callbacks=[
            CometLoggerCallback(
                api_key=API_KEY,
                project_name=PROJECT_NAME,
                tags=["comet_example"],
            )
        ],
    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
