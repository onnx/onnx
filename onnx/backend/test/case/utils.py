from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import pkgutil


def import_recursive(package):
    """
    Takes a package and imports all modules underneath it
    """
    pkg_dir = package.__path__
    module_location = package.__name__
    for (_module_loader, name, ispkg) in pkgutil.iter_modules(pkg_dir):
        module_name = "{}.{}".format(module_location, name)  # Module/package
        module = importlib.import_module(module_name)
        if ispkg:
            import_recursive(module)
