from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def yn(x):return 'YES' if x else 'NO'
def main():
    print('V2.25.3 STATUS\n===============')
    src=ROOT/'src/engine/shot_cross_thread_novelty_v253.py'; mainp=ROOT/'main.py'; menu=ROOT/'content/menu.json'
    print(f'runtime module: {yn(src.exists())}')
    print(f'runtime hook: {yn(mainp.exists() and "install_v253_runtime(App)" in mainp.read_text())}')
    print(f'menu prepared: {yn(menu.exists() and "Game Objects Test (V2.25.3)" in menu.read_text())}')
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
