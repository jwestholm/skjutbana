from __future__ import annotations


def main() -> None:
    print("V2.24.3 PREPARE")
    print("================")
    from automation.v243_apply_docs import main as apply_docs
    from automation.v243_apply_menu import main as apply_menu
    apply_docs()
    apply_menu()
    print("\nV2.24.3 is prepared. Start with: python3 main.py")


if __name__ == "__main__":
    main()
