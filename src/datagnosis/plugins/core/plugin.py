# import Standard
import sys
import platform
import importlib.util
from importlib.abc import Loader
from typing import Optional, Union, Any, Dict, List, Tuple, Type, Generator, Literal
from pathlib import Path
from abc import ABCMeta, abstractmethod
import inspect
from copy import deepcopy

# import Third Party
from pydantic import validate_arguments
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# absolute imports
import datagnosis.logger as log
import datagnosis.plugins.utils as utils
import datagnosis.plugins.core.utils as utils_core
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.utils.reproducibility import enable_reproducible_results, clear_cache
from datagnosis.utils.constants import DEVICE


# Base class for Hardness Classification Methods (HCMs)
class Plugin(metaclass=ABCMeta):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr: float,
        epochs: int,
        num_classes: int,
        device: Optional[torch.device] = DEVICE,
        logging_interval: int = 100,
        reproducible: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self._scores = None
        self.update_point = "post-epoch"
        self.logging_interval = logging_interval
        self.data_uncert_class = None
        self.has_been_fit = False
        self.score_names = None
        self.reproducible = reproducible
        clear_cache()
        if self.reproducible:
            log.debug("Fixing seed for reproducibility.")
            enable_reproducible_results(0)
        log.debug(f"Initialized parent plugin for {self.name()}")

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the plugin."""
        ...

    @staticmethod
    @abstractmethod
    def long_name() -> str:
        """The name of the plugin."""
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        """The type of the plugin."""
        ...

    @classmethod
    def hard_direction() -> str:
        """
        The direction of the scores for the plugin.
        Either 'low' if hard scores are low or 'high' if hard scores are high.
        """
        ...

    @classmethod
    def score_description() -> str:
        """A description of the scores for the plugin."""
        ...

    @classmethod
    def fqdn(cls) -> str:
        """The Fully-Qualified name of the plugin."""
        return cls.type() + "." + cls.name()

    def fit(
        self,
        datahandler: DataHandler,
        use_caches_if_exist: bool = True,
        workspace: Path = Path("workspace/"),
        *args,
        **kwargs,
    ):
        log.debug(f"self.has_been_fit {self.has_been_fit}")
        if self.has_been_fit == True:
            log.critical(f"Plugin {self.name()} has already been fit.")
            raise RuntimeError(
                f"Plugin {self.name()} has already been fit. Re-fitting is not allowed. If you wish to fit with different parameters, please create a new instance of the plugin."
            )
        all_args = locals()
        log.info(f"Fitting {self.name()} plugin")
        """Fit the plugin model"""
        self.datahandler = datahandler
        self.dataloader = datahandler.dataloader
        self.dataloader_unshuffled = datahandler.dataloader_unshuffled
        self.workspace = Path(workspace) if isinstance(workspace, str) else workspace
        self._updates_params = inspect.signature(self._updates).parameters
        kwargs_hash = utils.get_all_args_hash(all_args)

        # Move model to device
        self.model.to(self.device)

        # Set model to training mode
        self.optimizer.lr = self.lr
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, data in enumerate(self.dataloader):
                log.debug(f"Running Epoch {epoch}, batch {batch_idx}")
                update_values_cache_file = (
                    self.workspace
                    / f"{self.name()}_epoch:{epoch}_batch:{batch_idx}_{kwargs_hash}_{platform.python_version()}.bkp"
                )
                if (
                    use_caches_if_exist
                    and update_values_cache_file.exists()
                    and self.reproducible
                ):
                    log.debug("Loading update values from cache")
                    (
                        self.model,
                        outputs,
                        true_label,
                        indices,
                        batch_idx,
                    ) = utils.load_update_values_from_cache(update_values_cache_file)
                    if self.update_point == "mid-epoch":
                        log.debug("Updating scores mid-epoch")
                        self._safe_update(
                            y_pred=outputs, y_batch=true_label, sample_ids=indices
                        )
                else:
                    if (
                        use_caches_if_exist
                        and update_values_cache_file.exists()
                        and not self.reproducible
                    ):
                        log.warning(
                            "Intermediate cache file exists but reproducible is not set to true for this method. Recomputing intermediate outputs."
                        )
                    log.debug("Computing update values")

                    inputs, true_label, indices = data

                    inputs = inputs.to(self.device)
                    true_label = true_label.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    if self.update_point == "mid-epoch":
                        log.debug("Updating scores mid-epoch")
                        self._safe_update(
                            y_pred=outputs, y_batch=true_label, sample_ids=indices
                        )
                    loss = self.criterion(outputs, true_label)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    utils.cache_update_values(
                        [
                            self.model,
                            outputs,
                            true_label,
                            indices,
                            batch_idx,
                        ],
                        update_values_cache_file,
                    )

            epoch_loss = running_loss / len(self.dataloader)
            if epoch == 0 or (epoch + 1) % self.logging_interval == 0:
                log.info(f"Epoch {epoch+1}/{self.epochs}: Loss={epoch_loss:.4f}")

            # streamline repeated computation across methods
            logits = None
            targets = None
            probs = None
            indices = None
            if self.requires_intermediate == True:
                intermediates_cache_file = (
                    self.workspace
                    / f"{self.name()}_intermediates_epoch:{epoch}_{kwargs_hash}_{platform.python_version()}.bkp"
                )
                if (
                    use_caches_if_exist
                    and intermediates_cache_file.exists()
                    and self.reproducible
                ):
                    log.debug("Loading intermediate outputs from cache")
                    (
                        self.model,
                        logits,
                        targets,
                        probs,
                        indices,
                    ) = utils.load_update_values_from_cache(intermediates_cache_file)
                else:
                    if (
                        use_caches_if_exist
                        and intermediates_cache_file.exists()
                        and not self.reproducible
                    ):
                        log.warning(
                            "Intermediate cache file exists but reproducible is not set to true for this method. Recomputing intermediate outputs."
                        )
                    log.debug("Computing intermediate outputs")
                    (
                        logits,
                        targets,
                        probs,
                        indices,
                    ) = utils.get_intermediate_outputs(
                        net=self.model,
                        device=self.device,
                        dataloader=self.dataloader_unshuffled,
                    )
                    # Cache intermediate outputs
                    log.debug("Caching intermediate outputs")
                    utils.cache_update_values(
                        [self.model, logits, targets, probs, indices],
                        intermediates_cache_file,
                    )

            if self.update_point == "per-epoch":
                log.debug(f"Updating plugin after epoch {epoch+1}")
                self._safe_update(
                    net=self.model,
                    device=self.device,
                    logits=logits,
                    targets=targets,
                    probs=probs,
                    indices=indices,
                )

        if self.update_point == "post-epoch":
            log.debug("Updating plugin after training")
            self._safe_update(
                net=self.model,
                data_uncert_class=self.data_uncert_class,
                device=self.device,
                logits=logits,
                targets=targets,
                probs=probs,
            )

        self.has_been_fit = True

    def _safe_update(self, **kwargs: Any) -> None:
        if all([kwa in kwargs for kwa in self._updates_params]):
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in self._updates_params
            }
            self._updates(**filtered_kwargs)
        else:
            raise ValueError(
                f"""
Missing required arguments for {self.update_point} update. Required arguments are: 
{', '.join(self._updates_params)}. You provided: {', '.join(kwargs.keys())}.
"""
            )

    def _extract_datapoints_by_threshold(
        self,
        threshold: float,
        threshold_range: Tuple[Union[float, int], Union[float, int]],
        hardness: str,
    ) -> np.ndarray:
        """Extract datapoints from the plugin model by thresholding the scores"""
        extraction_scores = deepcopy(self.scores)
        if isinstance(extraction_scores, tuple):
            extraction_scores = extraction_scores[0]
        if threshold_range is None:
            if hardness == "hard":
                if self.hard_direction() == "low":
                    extracted = np.where(
                        extraction_scores < np.max(extraction_scores) - threshold
                    )
                else:
                    extracted = np.where(
                        extraction_scores > np.max(extraction_scores) + threshold
                    )
            else:
                if self.hard_direction() == "high":
                    extracted = np.where(
                        extraction_scores < np.max(extraction_scores) - threshold
                    )
                else:
                    extracted = np.where(
                        extraction_scores > np.max(extraction_scores) + threshold
                    )
        else:
            if threshold != 0.01:
                log.warning(
                    "You have provided a threshold_range but also a threshold. The threshold will be ignored."
                )

            extracted = np.where(
                (extraction_scores >= threshold_range[0])
                & (extraction_scores <= threshold_range[1])
            )
        extracted = extracted[0].tolist()

        return (
            self.dataloader_unshuffled.dataset[extracted],
            extraction_scores[extracted],
        )

    def _extract_datapoints_by_top_n(
        self,
        n: int,
        hardness: str = "hard",
        sort: bool = True,
    ) -> np.ndarray:
        """Extract datapoints from the plugin model by selecting the top n scores"""
        extraction_scores = deepcopy(self.scores)
        if isinstance(extraction_scores, tuple):
            extraction_scores = extraction_scores[0]
        if hardness == "hard":
            if self.hard_direction() == "low":
                extracted = np.argsort(extraction_scores)[:n]
            else:
                extracted = np.argsort(extraction_scores)[:n]
        else:
            if self.hard_direction() == "high":
                extracted = np.argsort(extraction_scores)[:n]
            else:
                extracted = np.argsort(extraction_scores)[:n]
        if sort:
            extracted = sorted(extracted)
        return (
            self.dataloader_unshuffled.dataset[extracted],
            extraction_scores[extracted],
        )

    def _extract_datapoints_by_index(self, indices: List[int]) -> np.ndarray:
        extraction_scores = deepcopy(self.scores)
        if isinstance(extraction_scores, tuple):
            extraction_scores = extraction_scores[0]
        return (self.dataloader_unshuffled.dataset[indices], extraction_scores[indices])

    def extract_datapoints(
        self,
        method: Literal["threshold", "top_n", "index"] = "threshold",
        hardness: Literal["hard", "easy"] = "hard",
        threshold_range: Optional[Tuple[float, float]] = None,
        threshold: Optional[float] = None,
        n: Optional[int] = None,
        indices: Optional[List[int]] = None,
        sort: bool = True,  # Only used for top_n
    ) -> np.ndarray:
        """
        Extract datapoints from the plugin model
        datapoints returned in the format Features, Labels, Indices, scores
        """
        if method == "threshold":
            if n is not None:
                log.warning(
                    "You have provided an `n` value, this is only used with the `top_n` method, so will be ignored. The threshold method requires a `threshold` or `threshold_range` value."
                )
            if indices is not None:
                log.warning(
                    "You have provided an `indices` list, this is only used with the `index` method, so will be ignored. The threshold method requires a `threshold` or `threshold_range` value."
                )
            if threshold is None and threshold_range is None:
                raise ValueError(
                    "You must provide either a `threshold` or `threshold_range` value."
                )
            return self._extract_datapoints_by_threshold(
                threshold, threshold_range, hardness
            )
        elif method == "top_n":
            if threshold_range is not None:
                log.warning(
                    "You have provided a `threshold_range`, this is only used with the `threshold` method, so will be ignored."
                )
            if threshold != 0.01:
                log.warning(
                    "You have provided a `threshold`, this is only used with the `threshold` method, so will be ignored."
                )
            if indices is not None:
                log.warning(
                    "You have provided an `indices` list, this is only used with the `index` method, so will be ignored."
                )
            if n is None:
                raise ValueError("You must provide an `n` value.")
            return self._extract_datapoints_by_top_n(n, hardness, sort=sort)
        elif method == "index":
            if threshold is not None:
                log.warning(
                    "You have provided a `threshold`, this is only used with the `threshold` method, so will be ignored."
                )
            if threshold_range is not None:
                log.warning(
                    "You have provided a `threshold_range`, this is only used with the `threshold` method, so will be ignored."
                )
            if n is not None:
                log.warning(
                    "You have provided an `n` value, this is only used with the `top_n` method, so will be ignored."
                )
            if indices is None:
                raise ValueError("You must provide an `indices` list.")
            return self._extract_datapoints_by_index(indices)
        else:
            raise ValueError(
                f"Unknown method {method}. Must be one of 'threshold', 'top_n'"
            )

    @abstractmethod
    def _updates(self):
        """Update the plugin model"""
        ...

    @abstractmethod
    def compute_scores(self):
        ...

    @property
    def scores(self):
        if self._scores is None:
            self.compute_scores()
        return self._scores

    @validate_arguments
    def plot_scores(
        self,
        *args,
        axis: Optional[int] = None,
        show: bool = True,
        plot_type: Literal["scatter", "dist"] = "dist",
        **kwargs,
    ):
        """Plot the scores"""
        log.info(f"Plotting {self.name()} scores")
        if self._scores is None:
            raise ValueError("No scores to plot. Run compute_scores() first.")

        if plot_type == "scatter":
            if utils_core.check_dim(self._scores) == 1:
                ax = sns.scatterplot(x=range(len(self._scores)), y=self._scores)
                ax.set(xlabel="Index", ylabel="Score")
                ax.set(title=f"Plot of {self.name()} scores")
                if show:
                    plt.show()
            elif isinstance(axis, int) and axis < len(self._scores):
                ax = sns.scatterplot(
                    x=range(len(self._scores[axis])), y=self._scores[axis]
                )
                ax.set(xlabel="Index", ylabel="Score")
                if self.score_names is not None:
                    ax.set(title=f"Plot of {self.score_names[axis]} scores")
                if show:
                    plt.show()
            else:
                log.critical(
                    "Cannot plot scatterplot for scores with more than one dimension. Either pass an axis or use plot_type='dist'"
                )
        elif plot_type == "dist":
            if utils_core.check_dim(self._scores) == 1:
                ax = sns.distplot(self._scores)
                ax.set(xlabel="Score", ylabel="Density")
                ax.set(title=f"Distribution of {self.name()} scores")
                if show:
                    plt.show()
            elif utils_core.check_dim(self._scores) >= 2:
                if isinstance(axis, int) and axis < len(self._scores):
                    ax = sns.distplot(self._scores[axis])
                    ax.set(xlabel="Score", ylabel="Density")
                    if self.score_names is not None:
                        ax.set(title=f"Distribution of {self.score_names[axis]} scores")
                    if show:
                        plt.show()
                if utils_core.check_dim(self._scores) == 2 and axis is None:
                    if not np.any(self._scores[0]) or not np.any(self._scores[1]):
                        log.warning(
                            "One of the scores is all zero. Cannot plot a 2D KDE plot. Try plotting 1D KDE plots instead by passing an axis parameter."
                        )
                    # Custom the color, add shade and bandwidth
                    ax = sns.kdeplot(
                        x=self._scores[0],
                        y=self._scores[1],
                        cmap="Reds",
                        shade=True,
                    )
                    if self.score_names is not None:
                        ax.set(xlabel=self.score_names[0], ylabel=self.score_names[1])
                        ax.set(
                            title=f"Map of {self.score_names[0]} vs {self.score_names[1]} scores"
                        )
                    if show:
                        plt.show()
        else:
            raise ValueError(
                "If scores have more than 2 dimensions. You must specify which axis to plot."
            )


class PluginLoader:
    """Plugin loading utility class.
    Used to load the plugins from the current folder.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, plugins: list, expected_type: Type, categories: list) -> None:
        self._plugins: Dict[str, Type] = {}
        self._available_plugins = {}
        for plugin in plugins:
            stem = Path(plugin).stem.split("plugin_")[-1]
            cls = self._load_single_plugin_impl(plugin)
            if cls is None:
                continue
            self._available_plugins[stem] = plugin
        self._expected_type = expected_type
        self._categories = categories

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _load_single_plugin_impl(self, plugin_name: str) -> Optional[Type]:
        """Helper for loading a single plugin implementation"""
        plugin = Path(plugin_name)
        name = plugin.stem
        ptype = plugin.parent.name

        module_name = f"datagnosis.plugins.{ptype}.{name}"
        log.debug(f"loading {module_name} plugin")
        failed = False
        for retry in range(2):
            # try:
            if module_name in sys.modules:
                mod = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, plugin)
                if spec is None:
                    raise RuntimeError("invalid spec")
                if not isinstance(spec.loader, Loader):
                    raise RuntimeError("invalid plugin type")

                mod = importlib.util.module_from_spec(spec)
                if module_name not in sys.modules:
                    sys.modules[module_name] = mod

                spec.loader.exec_module(mod)
            cls = mod.plugin
            if cls is None:
                log.critical(f"module disabled: {plugin_name}")
                return None

            failed = False
            break
            # except BaseException as e:
            #     log.critical(f"load failed: {e}")
            #     failed = True

        if failed:
            log.critical(f"module {name} load failed")
            return None

        return cls

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _load_single_plugin(self, plugin_name: str) -> bool:
        """Helper for loading a single plugin"""
        cls = self._load_single_plugin_impl(plugin_name)
        if cls is None:
            return False

        self.add(cls.name(), cls)
        return True

    def list(self) -> List[str]:
        """Get all the available plugins."""
        all_plugins = list(self._plugins.keys()) + list(self._available_plugins.keys())
        plugins = []
        for plugin in all_plugins:
            if self.get_type(plugin).type() in self._categories:
                plugins.append(plugin)

        return list(set(plugins))

    def types(self) -> List[Type]:
        """Get the loaded plugins types"""
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "PluginLoader":
        """Add a new plugin"""
        if name in self._plugins:
            log.info(f"Plugin {name} already exists. Overwriting")

        if not issubclass(cls, self._expected_type):
            raise ValueError(
                f"Plugin {name} must derive the {self._expected_type} interface."
            )
        self._plugins[name] = cls
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def load(self, buff: bytes) -> Any:
        """Load serialized plugin"""
        return Plugin.load(buff)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a new object from a plugin.
        Args:
            name: str. The name of the plugin
            &args, **kwargs. Plugin specific arguments

        Returns:
            The new object
        """
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} cannot be loaded.")

        # Use deepcopy to avoid sharing state between identical plugins
        return deepcopy(self._plugins[name](*args, **kwargs))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_type(self, name: str) -> Type:
        """Get the class type of a plugin.
        Args:
            name: str. The name of the plugin

        Returns:
            The class of the plugin
        """
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        """Iterate the loaded plugins."""
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        """The number of available plugins."""
        return len(self.list())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def reload(self) -> "PluginLoader":
        self._plugins = {}
        return self