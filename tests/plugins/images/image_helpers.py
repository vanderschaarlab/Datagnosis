# stdlib
from typing import Dict, List, Optional, Type

# datagnosis absolute
from datagnosis.plugins.core.plugin import Plugin
from datagnosis.plugins.images import ImagePlugins as Plugins


def generate_fixtures(
    name: str, plugin: Optional[Type], plugin_args: Dict = {}
) -> List:
    if plugin is None:
        return []

    def from_api() -> Plugin:
        return Plugins().get(name, **plugin_args)

    def from_module() -> Plugin:
        return plugin(**plugin_args)  # type: ignore

    return [from_api(), from_module()]
