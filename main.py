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
        # Fallback: always run stty sane as safety net
        _restore_terminal()

    atexit.register(_restore_saved)


# Guard terminal BEFORE any imports that might change TTY state
_save_and_guard_terminal()

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
