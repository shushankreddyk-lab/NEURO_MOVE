"""Shared Windows-safe stdout guard for EEG pipelines (fixes Errno 22)."""


def _install_windows_stdout_guard():
    import os as _os
    import sys as _sys
    if _os.name != 'nt':
        return
    try:
        _sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    try:
        _sys.stderr.reconfigure(errors='replace')
    except Exception:
        pass
    try:
        import io as _io
        if isinstance(_sys.stdout, _io.TextIOWrapper):
            _sys.stdout._windows_console = False
    except Exception:
        pass

_install_windows_stdout_guard()

def safe_print(*args, **kwargs):
    import sys as _sys
    try:
        _sys.stdout.write(' '.join(str(a) for a in args) + '\n')
    except OSError:
        pass
    try:
        _sys.stdout.flush()
    except OSError:
        pass

