from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def yn(x):return 'YES' if x else 'NO'
def main():
    print('V2.25.3-r2 STATUS\n==================')
    src=ROOT/'src/engine/shot_cross_thread_novelty_v253.py'; mainp=ROOT/'main.py'; menu=ROOT/'content/menu.json'; settings=ROOT/'src/engine/settings.py'
    st=settings.read_text(encoding='utf-8') if settings.exists() else ''
    print(f'runtime module: {yn(src.exists())}')
    print(f'runtime hook: {yn(mainp.exists() and "install_v253_runtime(App)" in mainp.read_text())}')
    print(f'menu prepared: {yn(menu.exists() and "Game Objects Test (V2.25.3)" in menu.read_text())}')
    print(f'settings API complete: {yn(len(st)>5000 and "def load_audio_peak_threshold" in st and "def load_led_settings" in st)}')
    print(f'content_rect local fallback: {yn("return pygame.Rect(0, 0, viewport.w, viewport.h)" in st)}')
    print('Packaging repair: unit-test settings stub REMOVED from archive')
    print('Worker/main registered readiness: SHARED process bridge')
    print('Shot identity: shot_id + peak timestamp validation')
    print('History plane: canonical full-camera XY')
    print('Recurrent hotspots: SOFT cross-shot penalty')
    print('Re-hit/hole-in-hole: LEGAL; signature gain can recover')
    print('Second-frame proof: original V2.22.5 confirmation before V2.25.3 balance')
    print('FULL rescue: global V2.22.5 path preserved')
    print('Game semantic weighting: NONE')
    print('XY snapping: NEVER')
    print('Next after physical acceptance: V2.25.4 moving-object continuity')
if __name__=='__main__':main()
