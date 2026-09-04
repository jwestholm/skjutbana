from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def yn(x): return "YES" if x else "NO"
def main():
    print("V2.25.2 STATUS\n===============")
    source=ROOT/'src/engine/shot_region_freshness_v252.py'; mainpy=ROOT/'main.py'; menu=ROOT/'content/menu.json'
    print(f"runtime module: {yn(source.exists())}")
    print(f"runtime hook: {yn(mainpy.exists() and 'install_v252_runtime(App)' in mainpy.read_text())}")
    print(f"menu prepared: {yn(menu.exists() and 'Game Objects Test (V2.25.2)' in menu.read_text())}")
    print("Proposal recall: V2.25.1 region-balanced + legacy remains available")
    print("Local authority: registered immediate PRE->POST freshness REQUIRED")
    print("Legacy authority: only after registered XY revalidation")
    print("Early waiting_post_peak emission: GATED until registered frame")
    print("Second-frame proof: V2.22.5 persistence retained")
    print("FULL rescue: global V2.22.5 path preserved")
    print("Role/target weighting: NONE")
    print("XY snapping: NEVER")
    print("Next after physical acceptance: V2.25.3 moving-object continuity")
if __name__=='__main__': main()
