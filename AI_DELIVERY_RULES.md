# AI delivery rules – Skjutbana

## Full-file packages only

For future version deliveries in this project:

- Ship complete replacement files in the ZIP.
- Do not ship `prepare`, `apply`, migration, menu-patching, settings-repair, or other installation scripts that rewrite repository files.
- If `content/menu.json` changes, ship the complete valid menu file using the repository schema version expected by `content_loader.py`.
- Do not replace `src/engine/settings.py` unless the version genuinely changes that source file. Never manufacture a partial/stub settings module for tests and place it in a release ZIP.
- Test fixtures must live outside the release tree and must never be copied into release paths.
- Documentation changes are delivered as complete `.md` files.
- A delta ZIP may contain only the files changed by the version, but every included file must itself be complete and ready to overwrite the corresponding repository file.

## V2.25.3-r3

This package is the first package following this rule. It contains no `automation/` directory and no menu fragments. The included `content/menu.json` is a complete version-1 menu based on the dev menu plus the Hit Context and Game Objects diagnostic entries.
