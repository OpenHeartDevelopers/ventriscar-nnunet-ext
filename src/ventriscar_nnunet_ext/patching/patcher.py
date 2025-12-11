# ~/dev/python/ventriscar-nnunet-ext/src/ventriscar_nnunet_ext/patching/patcher.py

import importlib
import logging
from typing import Callable, Any, List

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

log = logging.getLogger(__name__)
_original_recursive_find_python_class: Callable = recursive_find_python_class
_patch_applied: bool = False

def _patched_recursive_find_python_class(search_folders: List[str], class_name: str, current_module: str = 'nnunetv2') -> Any:
    """
    Patched version that is a perfect drop-in replacement.
    """
    found_class = _original_recursive_find_python_class(search_folders, class_name, current_module)
    
    if found_class is not None:
        return found_class

    # If the original fails, then we try our custom logic.
    log.debug("Class '%s' not found by nnunetv2. Attempting ventriscar_nnunet_ext search.", class_name)
    
    search_modules = [
        f"ventriscar_nnunet_ext.training.nnUNetTrainer.{class_name}",
        f"ventriscar_nnunet_ext.nets.{class_name}",
    ]

    for module_path in search_modules:
        try:
            module = importlib.import_module(module_path)
            found_class = getattr(module, class_name)
            log.info("Loaded '%s' via ventriscar_nnunet_ext patch.", class_name)
            return found_class
        except (ImportError, AttributeError):
            continue

    log.debug("Patch for '%s' did not find a custom class. Returning original failure.", class_name)
    return None 

def apply_nnunet_patch() -> None:
    global _patch_applied
    if _patch_applied:
        return

    import nnunetv2.utilities.find_class_by_name
    nnunetv2.utilities.find_class_by_name.recursive_find_python_class = _patched_recursive_find_python_class
    
    _patch_applied = True
    log.info("Applied ventriscar_nnunet_ext patch to nnunetv2 class discovery.")
