"""Single-threaded process setup; executed by absolute path before replacing itself with Lua."""

from __future__ import annotations

import os
import resource
import sys


def main() -> None:
    try:
        for limit, value in (
            (resource.RLIMIT_CPU, 2),
            (resource.RLIMIT_FSIZE, 65_536 + 4096),
            (resource.RLIMIT_NOFILE, 16),
        ):
            resource.setrlimit(limit, (value, value))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
        # macOS rejects RLIMIT_AS. The parent explicitly monitors RSS on this platform.
    except (OSError, ValueError):
        sys.stderr.write("lua_resource_limits_unavailable")
        raise SystemExit(78) from None
    os.execv(sys.argv[1], [sys.argv[1], sys.argv[2]])


if __name__ == "__main__":
    main()
