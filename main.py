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

# V2.7.2 DIRECT integration. This is deliberately before App import so the
# actual game process cannot silently miss the AIRuntime ranking hooks.
try:
    from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension

    install_ranker_v6_extension(source="main.py")
except Exception as exc:
    print(f"[RANKER-V6] V2.7.2 FATAL startup error: {exc!r}", file=sys.stderr)
    raise

from src.engine.app import App


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
