# stdlib
import glob
from os.path import basename, dirname, isfile, join

# third party
from pydantic import validate_call  # pyright: ignore

# datagnosis absolute
from datagnosis.plugins.core.plugin import Plugin, PluginLoader  # noqa: F401,E402

def_categories = [
    "generic",
    "images",
]
plugins = {}

for cat in def_categories:
    plugins[cat] = glob.glob(join(dirname(__file__), cat, "plugin*.py"))


class Plugins(PluginLoader):
    @validate_call
    def __init__(self, categories: list = def_categories) -> None:
        plugins_to_use = []
        for cat in categories:
            plugins_to_use.extend(plugins[cat])

        super().__init__(plugins_to_use, Plugin, categories)


__all__ = [
    basename(f)[:-3]
    for f in plugins[cat]  # pyright: ignore
    for cat in plugins
    if isfile(f)
] + [
    "Plugins",
    "Plugin",
]
