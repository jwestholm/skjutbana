from __future__ import annotations
def main():
    print('V2.25.3 PREPARE\n===============')
    from automation.v253_apply_docs import main as docs
    from automation.v253_apply_menu import main as menu
    docs(); menu(); print('\nV2.25.3 is prepared. Run selftest/verify, then: python3 main.py')
if __name__=='__main__': main()
