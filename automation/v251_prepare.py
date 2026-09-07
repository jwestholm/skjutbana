from __future__ import annotations


def main() -> None:
    print("V2.25.1 PREPARE")
    print("===============")
    from automation.v251_apply_docs import main as apply_docs
    from automation.v251_apply_menu import main as apply_menu
    apply_docs()
    apply_menu()
    print("\nV2.25.1 is prepared. Run selftest/verify, then start with: python3 main.py")


if __name__ == "__main__":
    main()
