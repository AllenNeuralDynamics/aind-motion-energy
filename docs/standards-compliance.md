# Motion Energy Pipeline — AIND Standards Compliance

## Context

The motion energy pipeline spans three repos under `AllenNeuralDynamics`:

| Repo | Role |
|---|---|
| [`aind-motion-energy`](https://github.com/AllenNeuralDynamics/aind-motion-energy) | Python library + CLI — the algorithm. **This repo; the organizing repo for this effort.** |
| [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) | Code Ocean capsule; thin shell over the CLI |
| [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) | Code Ocean launcher; fans out one capsule run per session |

The capsule is a genuine thin wrapper (`code/run` forwards `"$@"` to the library console script) — no logic is duplicated between library and capsule.

Work against the [AIND software practices](https://docs.allenneuraldynamics.org/en/latest/policies_practices/software_practices.html) is tracked issue-by-issue in [aind-motion-energy#16](https://github.com/AllenNeuralDynamics/aind-motion-energy/issues/16). Everything is complete except Phase 6 (metadata, below). This document records the decisions and deviations that aren't otherwise obvious from the code — see the individual (closed) issues and PRs for implementation history.

## Decisions

1. **Toolchain follows the current AIND standards page, not the older `aind-library-template`** (which is a generation behind: `flake8`, `setup.py`, Sphinx). We use `ruff` (lint + format, `line-length = 100`), `uv`, MkDocs-style numpydoc docstrings, and `mypy --strict`. Where the page and the template disagree, the page wins.
2. **Repo and package names keep the `aind-` prefix**, rather than the standard's `<modality>-<process>` form — see deviations below.
3. **Provenance (`processing.json`) belongs in the capsule, not the library.** The library stays runtime-agnostic.
4. **The AIND reusable workflows are a reference structure, not a verbatim requirement** — several can't be called unmodified (e.g. `type.yml` hardcodes `src/mypackage`), so we adapted our own equivalents.
5. **The 100%-coverage gate applies to the library only.** `aind-capsule-template` ships no `tests/`, and the practices page's own capsule guidance is "thin wrappers to Python packages" — gating a thin wrapper at 100% isn't what that requirement is for.

## Justified deviations

| Deviation | Justification |
|---|---|
| Repos keep the `aind-` prefix instead of `<modality>-<process>` | Renaming breaks the capsule Dockerfile git pin, the console-script name, the import path, and Code Ocean wiring, for no functional gain. Rule applies to new repos. |
| pytest instead of `unittest` | Already the library's convention; the modern AIND `test.yml` runs `uv run pytest tests/` anyway. |
| Capsules are not uv projects; CI uses `pip install ruff` rather than `uv run …` | A Code Ocean capsule is executed, never installed — it has no `[project]` table or build backend for uv to drive. |
| Capsule branching differs from `dev`/`main` PR flow | The Code Ocean web IDE pushes commits directly to the capsule's attached branch and cannot open PRs, so that branch can't be protected. The library complies fully; the capsules keep an unprotected `main` attached to CO with `dev` as staging. |
| No self-hosted MkDocs/Read the Docs site | A site was built (#26) but the RTD project was never imported, so it never went live; reverted (#29) rather than maintain a second, unpublished hosting path. Numpydoc docstrings and `examples/` stayed regardless. Don't re-propose without an actual RTD import. |
| Manual release process instead of automated version-bump-and-tag | The automated workflow needs a `repo-token` secret with push rights to protected `main`; the repo has no secrets configured and no service-account collaborator — real cross-team coordination, not a config tweak. Manual `uv version` + tag + `gh release create` satisfies the actual requirement (semver, GitHub Releases, changelog). Revisit if release cadence increases. |
| `dev` → `main` promotion PRs use a merge commit, not squash | Squashing that step discards the parent link between the two branches' histories, so the next promotion's conflict check falls back to a stale common ancestor — any file both branches touched independently (`CITATION.cff` on every release) shows a false add/add conflict, as happened cutting `v0.2.0`. `main`'s branch protection allows merge commits; `dev`'s requires linear history (squash-only, matching the standard), so the exception can't leak into the wrong branch. |

**Latent fragility, not yet resolved:** the library requires Python `>=3.11`, but the ME capsule Dockerfile starts from a `python3.9` Code Ocean base image and does `conda install python=3.11` on top. Works today; don't assume it's a clean 3.11 environment if debugging something version-sensitive.

## Still open

**M-1 · Emit `processing.json` across the pipeline** ([#20](https://github.com/AllenNeuralDynamics/aind-motion-energy/issues/20)) — last item by design, spans all three repos. AIND metadata standards expect derived data assets to carry a `processing.json`; the software-practices page is silent on this but `aind-data-schema` requires it. Open questions to resolve (candidates for SciComp input) before implementing:
- `ProcessName` has no motion-energy member — use `OTHER` with a descriptive `name`, or propose a new member to `aind-data-schema-models`.
- Which repo writes the file. Per decision 3 above, the capsule — but that means the ME capsule regains a thin `code/run_capsule.py` alongside the CLI call, strictly provenance-only (no compute/discovery/plotting), partially reversing the capsule's shell-only design. Alternative: the batch launcher writes one record per fanned-out run instead.
- Whether the library's metadata `dict` needs a typed model — not required for `mypy --strict` (`dict[str, Any]` suffices); only relevant if `output_parameters` wants a validated shape.

**BA-2 · Externalize hardcoded batch config** ([aind-motion-energy-batch#2](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch/issues/2)) — `ME_CAPSULE_ID`, `CO_DOMAIN`, `SCRATCH_BUCKET`, `SCRATCH_PREFIX` (currently a personal namespace), `RESULT_TAGS` are module constants in `code/run_capsule.py`; also pin `ME_CAPSULE_VERSION` (currently `None`, resolves to attached-branch HEAD). Not required by any cited standard — retained because it supports M-1's provenance record, which is only meaningful against a pinned capsule version. Drop from scope if M-1 is deferred indefinitely.

## Backlog (real defects, out of compliance scope)

No standard requires these; each is tracked as its own issue.

| Issue | What |
|---|---|
| [L-6 · aind-motion-energy#9](https://github.com/AllenNeuralDynamics/aind-motion-energy/issues/9) | Replace `print()` with `logging` — page marks structured logging TBD/not required. |
| [C-5 · aind-motion-energy-capsule#5](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule/issues/5) | Decide the app-panel / parameter surface — a product decision, not a standards question. |
| [C-7 · aind-motion-energy-capsule#7](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule/issues/7) | Detach the committed dataset pin. |
| [BA-3 · aind-motion-energy-batch#3](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch/issues/3) | Resilience: retries, per-session error isolation, incremental manifest writes — no `try`/`except` anywhere today. |
| [BA-4 · aind-motion-energy-batch#4](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch/issues/4) | Session resolution bug — `sessions.json` holds date-only names, so `resolve_raw_asset` never hits its exact-match branch. |

## Verification

```bash
# library
cd aind-motion-energy
uv sync --dev
uv run ruff check && uv run ruff format --check
uv run mypy src/aind_motion_energy --strict
uv run pytest --cov=aind_motion_energy --cov-report=term-missing --cov-fail-under=100

# confirm the 3.11/3.12 matrix locally before opening a PR, since CI runs both
uv run --python 3.12 pytest tests/
```

End to end:

1. **Library CLI** — `aind-motion-energy --input data/<clip>.mp4 --output results/ --summary-plots`; confirm the full output set appears.
2. **Capsule image** — rebuild the Code Ocean environment against the pinned release tag; confirm `pip list` lands in `/results/pip_list.txt`.
3. **Batch dry run** — `python -u run_capsule.py --dry-run` against a small `sessions.json`; confirm every session resolves to exactly one input asset.
4. **Batch real run** — one full session end to end; confirm the manifest shows `completed` with a non-null `result_asset_id` and `_me_metadata.json` reports a frame count matching full video length.
5. **After M-1 only** — confirm `processing.json` validates against `aind-data-schema`.
