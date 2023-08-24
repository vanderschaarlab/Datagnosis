# stdlib
import importlib.util
import inspect
import platform
import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from importlib.abc import Loader
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Type, Union

# third party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydantic import validate_call
from typing_extensions import Self

# datagnosis absolute
import datagnosis.logger as log
import datagnosis.plugins.core.utils as utils_core
import datagnosis.plugins.utils as utils
from datagnosis.plugins.core.datahandler import DataHandler
from datagnosis.utils.constants import DEVICE
from datagnosis.utils.reproducibility import clear_cache, enable_reproducible_results

# The complete list of parameters that can be passed to the _updates method of any of the plugins.
UPDATE_PARAMS: Dict[str, Any] = {
    "y_pred": None,
    "y_batch": None,
    "sample_ids": None,
    "net": None,
    "device": None,
    "logits": None,
    "targets": None,
    "probs": None,
    "indices": None,
    "data_uncert_class": None,
}


# Base class for Hardness Classification Methods (HCMs)
class Plugin(metaclass=ABCMeta):
    @validate_call(config={"arbitrary_types_allowed": True})
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
        requires_intermediate: bool = False,
    ):
        """
        This is the base class for all plugins. It is an abstract class and should not be instantiated directly.
        In order to create a new plugin, you should inherit from this class and implement the abstract methods, which
        are listed below.

        required methods:
            - name
            - long_name
            - type
            - hard_direction
            - score_description
            - _updates
            - compute_scores


        Args:
            model (torch.nn.Module): The downstream classifier you wish to use and therefore also the model you wish to judge the hardness of characterization of data points with.
            criterion (torch.nn.Module): The loss criterion you wish to use to train the model.
            optimizer (torch.optim.Optimizer): The optimizer you wish to use to train the model.
            lr (float): The learning rate you wish to use to train the model.
            epochs (int): The number of epochs you wish to train the model for.
            num_classes (int): The number of labelled classes in the classification task.
            device (Optional[torch.device], optional): The torch.device used for computation. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
            logging_interval (int, optional): The interval at which to log training progress. Defaults to 100.
            reproducible (bool, optional): A flag to indicate whether or not to fix the seed for reproducibility. Defaults to True.
            requires_intermediate (bool, optional): A flag to indicate whether or not the plugin requires intermediate data. This is dependent on the specific requirements of the plugin being implemented. Defaults to False.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device is not None else DEVICE
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.reproducible = reproducible
        self.logging_interval = logging_interval
        self.requires_intermediate = requires_intermediate
        self.update_point: str = "post-epoch"
        self.data_uncert_class = None
        self.has_been_fit: bool = False
        self.score_names: Optional[Union[Tuple, np.ndarray]] = None
        self._scores: Optional[Union[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None
        clear_cache()
        if self.reproducible:
            log.debug("Fixing seed for reproducibility.")
            enable_reproducible_results(0)
        log.debug(f"Initialized parent plugin for {self.name()}")
        # Update UPDATE_PARAMS
        UPDATE_PARAMS["device"] = self.device

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

    @staticmethod
    @abstractmethod
    def hard_direction() -> str:
        """
        The direction of the scores for the plugin.
        Either 'low' if hard scores are low or 'high' if hard scores are high.
        """
        ...

    @staticmethod
    @abstractmethod
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
        workspace: Union[Path, str] = Path("workspace/"),
        *args: Any,
        **kwargs: Any,
    ) -> Self:  # type: ignore
        """
        Fit the plugin model.

        Args:
            datahandler (DataHandler): The `datagnosis.plugins.core.datahandler.DataHandler` object that contains the data to be used for fitting.
            use_caches_if_exist (bool, optional): A flag to indicate whether or not to use cached data if it exists. Defaults to True.
            workspace (Union[Path, str], optional): A path to the workspace directory. Defaults to Path("workspace/").

        Raises:
            RuntimeError: Raises a RuntimeError if the plugin's `fit` method has already been called.
        """
        log.debug(f"self.has_been_fit {self.has_been_fit}")
        if self.has_been_fit is True:
            log.critical(f"Plugin {self.name()} has already been fit.")
            raise RuntimeError(
                f"Plugin {self.name()} has already been fit. Re-fitting is not allowed. If you wish to fit with different parameters, please create a new instance of the plugin."
            )
        all_args = locals()
        log.info(f"Fitting {self.name()}")
        """Fit the plugin model"""
        self.datahandler = datahandler
        self.dataloader = datahandler.dataloader
        self.dataloader_unshuffled = datahandler.dataloader_unshuffled
        self.workspace = Path(workspace) if isinstance(workspace, str) else workspace
        self._updates_params = inspect.signature(self._updates).parameters
        kwargs_hash = utils.get_all_args_hash(all_args)

        # Move model to device
        self.model.to(self.device)

        logits = None
        targets = None
        probs = None
        indices = None
        self.optimizer.lr = self.lr  # pyright: ignore
        for epoch in range(self.epochs):
            # Set model to training mode
            self.model.train()
            running_loss = 0.0
            for batch_idx, data in enumerate(self.dataloader):
                log.debug(f"Running Epoch {epoch}, batch {batch_idx}")
                update_values_cache_file = (
                    self.workspace
                    / f"{self.name()}_epoch-{epoch}_batch-{batch_idx}_{kwargs_hash}_{platform.python_version()}.bkp"
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
                        # Update UPDATE_PARAMS before calling _safe_update() for mid-epoch
                        UPDATE_PARAMS["y_pred"] = outputs
                        UPDATE_PARAMS["y_batch"] = true_label
                        UPDATE_PARAMS["sample_ids"] = indices
                        UPDATE_PARAMS["net"] = self.model
                        self._safe_update(**UPDATE_PARAMS)
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
                        # Update UPDATE_PARAMS before calling _safe_update() for mid-epoch
                        UPDATE_PARAMS["y_pred"] = outputs
                        UPDATE_PARAMS["y_batch"] = true_label
                        UPDATE_PARAMS["sample_ids"] = indices
                        UPDATE_PARAMS["net"] = self.model
                        self._safe_update(**UPDATE_PARAMS)
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
            if self.requires_intermediate is True:
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
                    (logits, targets, probs, indices,) = utils.get_intermediate_outputs(
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
                # Update UPDATE_PARAMS before calling _safe_update() for per-epoch
                UPDATE_PARAMS["net"] = self.model
                UPDATE_PARAMS["logits"] = logits
                UPDATE_PARAMS["targets"] = targets
                UPDATE_PARAMS["probs"] = probs
                UPDATE_PARAMS["indices"] = indices
                self._safe_update(**UPDATE_PARAMS)

        if self.update_point == "post-epoch":
            log.debug("Updating plugin after training")
            # Update UPDATE_PARAMS before calling _safe_update() for per-epoch
            UPDATE_PARAMS["net"] = self.model
            UPDATE_PARAMS["logits"] = logits
            UPDATE_PARAMS["targets"] = targets
            UPDATE_PARAMS["probs"] = probs
            UPDATE_PARAMS["data_uncert_class"] = self.data_uncert_class
            self._safe_update(**UPDATE_PARAMS)
        self.has_been_fit = True
        self.compute_scores()
        log.debug("Plugin fit complete and scores computed.")
        return self

    def _safe_update(self, **kwargs: Any) -> None:
        """
        A wrapper around the update method that makes sure that the required arguments, and only the required arguments are provided.

        Raises:
            ValueError: Raises a ValueError if the required arguments are not provided.
        """
        if all([kwa in kwargs for kwa in self._updates_params]):
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in self._updates_params
            }
            log.debug(f"Updating plugin with {filtered_kwargs}.")
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
        threshold: Optional[float],
        threshold_range: Optional[Tuple[Union[float, int], Union[float, int]]],
        hardness: str,
        sort: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List], np.ndarray]:
        """
        Internal function to extract datapoints from the plugin model by applying a threshold or range to the scores. Called by extract_datapoints.

        Args:
            threshold (Optional[float]): The threshold to apply to the scores. Must be provided if threshold_range is None.
            threshold_range (Optional[Tuple[Union[float, int], Union[float, int]]]): The range of thresholds to apply to the scores. Must be provided if threshold is None.
            hardness (str): Flag to indicate whether to extract hard or easy data points.

        Raises:
            ValueError: raises a ValueError if the plugin has not been fit yet.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the extracted datapoints and the scores of the extracted datapoints.
        """
        if self._scores is None:
            raise ValueError(
                "You must fit the plugin before extracting datapoints by threshold"
            )
        else:
            extraction_scores = deepcopy(self._scores)
            if isinstance(extraction_scores, tuple):
                extraction_scores = extraction_scores[
                    0
                ]  # TODO: should be able to extract on any score, not just first
            if threshold_range is None:
                if threshold is None:
                    raise ValueError(
                        "You must provide either a threshold or threshold_range value."
                    )
                if hardness == "hard":
                    if self.hard_direction() == "low":
                        extracted = np.where(
                            extraction_scores
                            < np.max(extraction_scores).item() - threshold
                        )
                    else:
                        extracted = np.where(
                            extraction_scores
                            > np.max(extraction_scores).item() + threshold
                        )
                else:
                    if self.hard_direction() == "high":
                        extracted = np.where(
                            extraction_scores
                            < np.max(extraction_scores).item() - threshold
                        )
                    else:
                        extracted = np.where(
                            extraction_scores
                            > np.max(extraction_scores).item() + threshold
                        )
            else:
                if threshold is not None:
                    log.warning(
                        "You have provided a threshold_range but also a threshold. The threshold will be ignored."
                    )

                extracted = np.where(
                    (extraction_scores >= threshold_range[0])
                    & (extraction_scores <= threshold_range[1])
                )
            extracted = extracted[0].tolist()

            return (
                self.dataloader_unshuffled.dataset[extracted],  # pyright: ignore
                extraction_scores[extracted],
            )

    def _extract_datapoints_by_top_n(
        self,
        n: int,
        hardness: str = "hard",
        sort_by_index: bool = True,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List], np.ndarray]:
        """Internal function to extract datapoints from the plugin model by  selecting the top n scores. Called by extract_datapoints.


        Args:
            n (int): The number of datapoints to extract.
            hardness (str, optional): Flag to indicate whether to extract hard or easy data points. Defaults to "hard".
            sort_by_index (bool, optional): Flag to indicate whether to sort the extracted datapoints by their index. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the extracted datapoints and the scores of the extracted datapoints.
        """
        if self._scores is None:
            raise ValueError(
                "You must fit the plugin before extracting datapoints by top n"
            )
        extraction_scores = deepcopy(self._scores)
        if isinstance(extraction_scores, tuple):
            extraction_scores = extraction_scores[
                0
            ]  # TODO: should be able to extract on any score, not just first
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
        if sort_by_index:
            log.info("Sorting extracted datapoints")
            log.info(extracted)
            extracted = sorted(extracted)
            log.info(extracted)
        return (
            self.dataloader_unshuffled.dataset[extracted],  # pyright: ignore
            extraction_scores[extracted],
        )

    def _extract_datapoints_by_index(
        self, indices: List[int]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List], np.ndarray]:
        """Internal function to extract datapoints from the plugin model by selecting datapoints by index. Called by extract_datapoints.

        Args:
            indices (List[int]): A list of indices to extract.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the extracted datapoints and the scores of the extracted datapoints.
        """
        if self._scores is None:
            raise ValueError(
                "You must fit the plugin before extracting datapoints by index"
            )
        extraction_scores = deepcopy(self._scores)
        if isinstance(extraction_scores, tuple):
            extraction_scores = extraction_scores[0]
        return (
            self.dataloader_unshuffled.dataset[indices],  # pyright: ignore
            extraction_scores[indices],
        )  # pyright: ignore

    def extract_datapoints(
        self,
        method: Literal["threshold", "top_n", "index"] = "threshold",
        hardness: Literal["hard", "easy"] = "hard",
        threshold: Optional[float] = None,
        threshold_range: Optional[Tuple[float, float]] = None,
        n: Optional[int] = None,
        indices: Optional[List[int]] = None,
        sort_by_index: bool = True,  # Only used for top_n
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List], np.ndarray]:
        """Extracts datapoints from the plugin model by applying a threshold or range to the scores, selecting the top n scores, or selecting datapoints by index.

        Args:
            method (Literal[threshold, top_n, index], optional): The method to use to extract datapoints. Defaults to "threshold".
            hardness (Literal[hard, easy], optional): Flag to indicate whether to extract hard or easy data points. Defaults to "hard".
            threshold (Optional[float], optional): The threshold to apply to the scores. Must be provided if the given method is "threshold" and threshold_range is None. Defaults to None.
            threshold_range (Optional[Tuple[float, float]], optional): The range of thresholds to apply to the scores. Must be provided if the given method is "threshold" and the value passed to threshold is None. Defaults to None.
            n (Optional[int], optional): The number of datapoints to extract. Must be provided if the given method is "top_n". Defaults to None.
            indices (Optional[List[int]], optional): The indices of the datapoints to extract. Must be provided if the given method is "index". Defaults to None.
            sort_by_index (bool, optional): Flag to indicate whether to sort_by_index the extracted datapoints. Defaults to True.

        Raises:
            ValueError: raised if the given method is not one of "threshold", "top_n", or "index".
            ValueError: raised if the given method is "threshold" but neither threshold nor threshold_range is provided.
            ValueError: raised if the given method is "top_n" but n is not provided.
            ValueError: raised if the given method is "index" but a list of indices is not provided.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The extracted datapoints and the scores of the extracted datapoints. Datapoints returned in the format ((Features, Labels, Indices), scores)
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
            if threshold is not None:
                log.warning(
                    "You have provided a `threshold`, this is only used with the `threshold` method, so will be ignored."
                )
            if indices is not None:
                log.warning(
                    "You have provided an `indices` list, this is only used with the `index` method, so will be ignored."
                )
            if n is None:
                raise ValueError("You must provide an `n` value.")
            return self._extract_datapoints_by_top_n(
                n, hardness, sort_by_index=sort_by_index
            )
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
                f"Unknown method {method}. Must be one of 'threshold', 'top_n', or 'index'."
            )

    @abstractmethod
    def _updates(self, *args: Any, **kwargs: Any) -> None:
        """Update the plugin model"""
        ...

    @abstractmethod
    def compute_scores(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the scores for the plugin model"""
        ...

    @property
    def scores(self) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """The scores for the plugin model

        Raises:
            ValueError: raised if the plugin has not been fit.
            ValueError: raised if the scores have not been computed.

        Returns:
            np.ndarray: The scores for the plugin model
        """
        if self.has_been_fit:
            if self._scores is not None:
                return self._scores
            else:
                raise ValueError(
                    "Scores have not been computed. Check that `compute_scores()` was called in the `fit()` method."
                )
        else:
            raise ValueError(
                "Plugin has not been fit. `fit()` must be run before getting scores."
            )

    @validate_call
    def plot_scores(
        self,
        *args: Any,
        axis: Optional[int] = None,
        show: bool = True,
        plot_type: Literal["scatter", "dist"] = "dist",
        **kwargs: Any,
    ) -> None:
        """_summary_

        Args:
            axis (Optional[int], optional): The axis to plot. If None, plot a higher dimentional plot. Defaults to None.
            show (bool, optional): Flag to indicate whether to show the plot. Defaults to True.
            plot_type (Literal[scatter, dist], optional): The type of plot to show. Can be either "scatter" or "dist". Defaults to "dist".

        Raises:
            ValueError: raised if the scores have not been computed.
            ValueError: raised if scores have more than 2 dimensions. You must specify which axis to plot.
        """
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

    @validate_call
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

    @validate_call
    def _load_single_plugin_impl(self, plugin_name: str) -> Optional[Type]:
        """Helper for loading a single plugin implementation"""
        cls = None  # This should be overwritten by the plugin below

        plugin = Path(plugin_name)
        name = plugin.stem
        ptype = plugin.parent.name

        module_name = f"datagnosis.plugins.{ptype}.{name}"
        log.debug(f"loading {module_name} plugin")
        failed = False
        for retry in range(2):
            try:
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
            except BaseException as e:
                log.critical(f"load failed: {e}")
                failed = True

        if failed:
            log.critical(f"module {name} load failed")
            return None

        if cls is None:
            log.critical(f"module {name} load failed")
            return None

        return cls

    @validate_call(config={"arbitrary_types_allowed": True})
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

    @validate_call(config={"arbitrary_types_allowed": True})
    def get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a new object from a plugin.

        Args:
            name (str): The name of the plugin

        Raises:
            ValueError: raises if the plugin doesn't exist
            ValueError: raises if the plugin cannot be loaded

        Returns:
            Any: The new object
        """
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} cannot be loaded.")

        # Use deepcopy to avoid sharing state between identical plugins
        return deepcopy(self._plugins[name](*args, **kwargs))

    @validate_call
    def get_type(self, name: str) -> Type:
        """
        Get the class type of a plugin.

        Args:
            name (str): The name of the plugin

        Raises:
            ValueError: raises if the plugin doesn't exist
            ValueError: raises if the plugin cannot be loaded

        Returns:
            Type: The class of the plugin
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

    @validate_call
    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def reload(self) -> "PluginLoader":
        self._plugins = {}
        return self
