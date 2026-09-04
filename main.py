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
from src.engine.shot_object_local_v241 import install_v241_runtime
from src.engine.shot_object_local_v243 import install_v243_runtime
from src.engine.shot_object_local_v244 import install_v244_runtime
from src.engine.shot_context_v250 import install_v250_runtime
from src.engine.shot_region_proposal_v251 import install_v251_runtime
from src.engine.shot_region_freshness_v252 import install_v252_runtime

# Install in order: V2.22.3 establishes top-level PANG priority / object
# snapshots; V2.22.4 then replaces the blocking shot path with async CV and
# async advisory/training AI work. V2.22.5 then replaces repeated full
# persistence passes with a sparse live proposal + local confirmation lane.
# V2.22.6 fixes track semantics so same-frame candidate clusters are support,
# not fake temporal hits, and adds raw audio near-miss telemetry.
# V2.24.1 consumes the V2.24.0 shot-time camera HitRegions to constrain the
# FIRST physical proposal search. V2.22.5 full rescue remains global.
# V2.24.3 fixes the implicit content-rect origin and moves that restriction
# up to HitScanner ROI level so legacy/V1, early V2 and normal V2 share it.
# V2.24.4 then maps canonical full-camera HitRegions into V2.22.1's active
# crop/worker-local detector plane before the local ROI mask is built.
# V2.25.0 finally carries scanner shot_id through HitEvent before subscribers
# are notified, allowing GameObjects to resolve against the exact frozen snapshot.
# V2.25.1 then partitions that frozen object search area into balanced physical
# proposal/confirmation regions so one noisy object area cannot monopolise hits.
# V2.25.2 closes the remaining authority leak: early/legacy candidates may aid
# recall, but object-context hits cannot emit until exact XY has registered V2
# PRE->POST freshness evidence. The explicit V2.22.5 FULL rescue stays global.
install_v2223_runtime(App)
install_v2224_runtime(App)
install_v2225_runtime(App)
install_v2226_runtime(App)
install_v241_runtime(App)
install_v243_runtime(App)
install_v244_runtime(App)
install_v250_runtime(App)
install_v251_runtime(App)
install_v252_runtime(App)

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
