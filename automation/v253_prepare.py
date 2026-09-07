from __future__ import annotations

def main():
    print('V2.25.3-r2 PREPARE\n==================')
    from automation.v253_r2_repair_settings import main as repair_settings
    from automation.v253_apply_docs import main as docs
    from automation.v253_r2_apply_docs import main as docs_r2
    from automation.v253_apply_menu import main as menu
    repair_settings()
    docs()
    docs_r2()
    menu()
    print('\nV2.25.3-r2 is prepared. Run selftest/verify, then: python3 main.py')

if __name__=='__main__':
    main()
