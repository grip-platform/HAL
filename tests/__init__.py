# HAL test suite
# Ensure the worktree's lib takes priority over any installed ~/.hal copy.
# Evict cached hal modules so re-imports resolve from the worktree.
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _sys.path[:1] != [_REPO]:
    _sys.path.insert(0, _REPO)
    for _mod in list(_sys.modules):
        if _mod == "lib" or _mod.startswith("lib.hal"):
            del _sys.modules[_mod]
