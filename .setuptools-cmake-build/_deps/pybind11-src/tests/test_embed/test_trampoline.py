from __future__ import annotations

import trampoline_module


def func():
    class Test(trampoline_module.test_override_cache_helper):
        def func(self):
            return 42

    return Test()


def func2():
    class Test(trampoline_module.test_override_cache_helper):
        pass

    return Test()
