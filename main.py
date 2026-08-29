import atexit
import os
import sys


def _restore_terminal() -> None:
    """Restore terminal to sane state. Called on any exit."""
    try:
        if sys.stdin.isatty():
            os.system("stty sane 2>/dev/null")
    except Exception:
        pass


def _save_and_guard_terminal():
    """Save TTY state before anything touches it, restore on exit."""
    _saved_tty = None
    try:
        import termios
        if sys.stdin.isatty():
            _saved_tty = termios.tcgetattr(sys.stdin)
    except Exception:
        pass

    def _restore_saved():
        try:
            if _saved_tty is not None:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _saved_tty)
        except Exception:
            pass
        _restore_terminal()

    atexit.register(_restore_saved)


_save_and_guard_terminal()

from src.engine.app import App
from src.engine.shot_critical_v2223 import install_v2223_runtime
from src.engine.shot_async_v2224 import install_v2224_runtime
from src.engine.shot_fast_v2225 import install_v2225_runtime
from src.engine.shot_track_v2226 import install_v2226_runtime

# Install in order: V2.22.3 establishes top-level PANG priority / object
# snapshots; V2.22.4 then replaces the blocking shot path with async CV and
# async advisory/training AI work. V2.22.5 then replaces repeated full
# persistence passes with a sparse live proposal + local confirmation lane.
# V2.22.6 fixes track semantics so same-frame candidate clusters are support,
# not fake temporal hits, and adds raw audio near-miss telemetry.
install_v2223_runtime(App)
install_v2224_runtime(App)
install_v2225_runtime(App)
install_v2226_runtime(App)

if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nAvbrutet med Ctrl+C.")
    except Exception as exc:
        print(f"\nOväntat fel: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        _restore_terminal()
