# Repository working rules

- Read README.md, ARCHITECTURE.md, CURRENT_STATE.md and ROADMAP.md before major work.
- Check `git status` before modifying code; preserve user changes.
- Never claim an accuracy improvement without reproducible measurements. Distinguish
  offline candidate evaluation, live-path-equivalent replay, and physical validation.
- Change one major detector hypothesis at a time; measurement work must not tune behavior.
- Preserve baseline inputs and results. Create new evaluation run directories; never
  overwrite a baseline. Keep generated output separate from source where practical.
- Do not push, merge, commit, or modify other branches unless explicitly instructed.
- Do not modify unrelated OS configuration or credentials, or delete training data.
- Run relevant evaluation selftests when changing measurement code.
- Update CURRENT_STATE.md when a completed development step changes actual project state.
- Never stage/commit `content/ai/settings.json`, generated physical traces, or local
  evaluation artifacts. Physical summary documentation is permitted.
- The canonical challenger remains SHADOW-only; model status and hash are explicit.
- Do not confuse a track's emitted `state=confirmed` with local-confirmation proof.
  Legacy `last_stable_tracks` contains only the top-eight debug view.
