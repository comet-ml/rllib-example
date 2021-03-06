"""
From existing implementation in Ray
"""


from typing import Dict, List

from ray.tune.logger import LoggerCallback
from ray.tune.trial import Trial
from ray.tune.utils import flatten_dict


def _import_comet():
    try:
        import comet_ml
    except ImportError:
        raise RuntimeError("pip install 'comet-ml' to use " "CometLoggerCallback")


class CometLoggerCallback(LoggerCallback):
    # Do not log these metrics.
    _exclude_results = ["done", "should_checkpoint"]

    # These values should be logged as system info instead of metrics.
    _system_results = ["node_ip", "hostname", "pid", "date"]

    # These values should be logged as "Other" instead of as metrics.
    _other_results = ["trial_id", "experiment_id", "experiment_tag"]

    _episode_results = ["hist_stats/episode_reward", "hist_stats/episode_lengths"]

    def __init__(self, online: bool = True, tags: List[str] = None, **kwargs):
        _import_comet()
        self.online = online
        self.tags = tags
        self.experiment_kwargs = kwargs

        # Mapping from trial to experiment object.
        self._trial_experiments = {}

        self._to_exclude = set(self._exclude_results)
        self._to_system = set(self._system_results)
        self._to_other = set(self._other_results)
        self._to_episodes = set(self._episode_results)

    def log_trial_start(self, trial: "Trial"):
        import comet_ml
        from comet_ml import Experiment, OfflineExperiment

        if trial not in self._trial_experiments:
            experiment_cls = Experiment if self.online else OfflineExperiment
            # Don't log command line arguments.
            self.experiment_kwargs["parse_args"] = False
            # Don't log console output to avoid large messages & throttling.
            self.experiment_kwargs["auto_output_logging"] = False
            # Don't log git related data to avoid large messages & throttling.
            self.experiment_kwargs["log_git_metadata"] = False
            self.experiment_kwargs["log_git_patch"] = False
            experiment = experiment_cls(**self.experiment_kwargs)
            self._trial_experiments[trial] = experiment
            # Set global experiment to None to allow for multiple experiments.
            comet_ml.config.set_global_experiment(None)
        else:
            experiment = self._trial_experiments[trial]

        experiment.set_name(str(trial))
        experiment.add_tags(self.tags)

        config = trial.config.copy()
        config.pop("callbacks", None)
        experiment.log_parameters(config)

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_experiments:
            self.log_trial_start(trial)
        experiment = self._trial_experiments[trial]
        step = result["training_iteration"]

        config_update = result.pop("config", {}).copy()

        metric_logs = {}
        episode_logs = {}
        system_logs = {}
        other_logs = {}

        flat_result = flatten_dict(result, delimiter="/")
        for k, v in flat_result.items():
            if any(k.startswith(item + "/") or k == item for item in self._to_exclude):
                continue
            if any(k.startswith(item + "/") or k == item for item in self._to_other):
                other_logs[k] = v
            elif any(k.startswith(item + "/") or k == item for item in self._to_system):
                system_logs[k] = v
            elif any(
                k.startswith(item + "/") or k == item for item in self._to_episodes
            ):
                episode_logs[k] = v
            else:
                metric_logs[k] = v

        config_update.pop("callbacks", None)  # Remove callbacks
        for k, v in config_update.items():
            if isinstance(v, dict):
                experiment.log_parameters(flatten_dict({k: v}, "/"), step=step)

            else:
                experiment.log_parameter(k, v, step=step)

        experiment.log_others(other_logs)
        for k, v in system_logs.items():
            experiment.log_system_info(k, v)

        for k, v in episode_logs.items():
            experiment.log_curve(k, x=range(len(v)), y=v, step=step)

        experiment.log_metrics(metric_logs, step=step)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        self._trial_experiments[trial].end()
        del self._trial_experiments[trial]

    def __del__(self):
        for trial, experiment in self._trial_experiments.items():
            experiment.end()
        self._trial_experiments = {}
