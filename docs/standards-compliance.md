# Motion Energy Pipeline — AIND Standards Compliance Plan

## Context

The motion energy pipeline spans three repos under `AllenNeuralDynamics`:

| Repo | Role |
|---|---|
| [`aind-motion-energy`](https://github.com/AllenNeuralDynamics/aind-motion-energy) | Python library + CLI — the algorithm. **This repo; the organizing repo for this effort.** |
| [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) | Code Ocean capsule; thin shell over the CLI |
| [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) | Code Ocean launcher; fans out one capsule run per session |

The pipeline works end to end, and the layering is already sound — the capsule is a genuine thin wrapper (`code/run` is five lines forwarding `"$@"` to the library console script), and duplicated logic was deliberately removed in capsule commit `1a86686`. What is missing is essentially all of the engineering scaffolding the [AIND software practices](https://docs.allenneuraldynamics.org/en/latest/policies_practices/software_practices.html) require:

- **No CI anywhere.** None of the three repos has a `.github/` directory.
- **No lint or format config anywhere**, in any repo.
- **Tests exist only in the library**, and cover neither `cli.py` nor `viz.save_summary_plots` — so the required 100% coverage gate cannot pass today.
- **Docstrings are free-form prose**, not numpydoc; several modules and functions have none.
- **No versioning or releases.** No tags, no `__version__`; the capsule pins the library to a bare 40-char SHA.
- **No logging** — bare `print()` and `set -x` throughout.
- **No `processing.json` / aind-data-schema provenance** from either capsule.
- **Hardcoded config** in the batch launcher (capsule ID, `matt.becker/motion_energy` scratch prefix, CO domain) and **no error handling at all** — a single HTTP failure mid-batch aborts the run and loses the manifest.

Intended outcome: all three repos pass AIND standards, with the work broken into independently mergeable issues tracked in `aind-motion-energy` as the organizing repo.

**First action item (X-1): commit this document to `aind-motion-energy` as `docs/standards-compliance.md` and open the tracking issue there.** Everything else follows from that tracking issue.

## Decisions made

Settled with the user before writing this plan:

1. **Toolchain: follow the current standards page, not the older `aind-library-template`.** `ruff` (lint + format, `line-length = 100`), `uv` for package management, MkDocs + mkdocstrings on Read the Docs, numpydoc docstrings. The library already uses `uv` and `uv_build`, so this is the smaller lift. This choice is cosmetic — it changes source text and CI only, never capsule runtime behavior.
2. **Keep the current repo and package names.** The standards say new packages should be `<modality>-<process>` without an `aind-` prefix, but renaming would break the Dockerfile git pin, the console-script name, the import path, and Code Ocean capsule wiring for no functional gain. Treat the rule as applying to new repos; note the deviation in the compliance doc.
3. **Provenance belongs in the capsule, not the library.** The library stays runtime-agnostic and returns a structured parameters/metadata object; the capsule builds the `DataProcess`/`Processing` record and writes `processing.json`.
4. **Standards-first, minimal redesign.** Fix compliance and accept only the breaking changes that compliance actually forces (see below). Deeper redesign — restructuring the compute return shape beyond a typed metadata model, consolidating capsules, reworking the app-panel parameter contract — is out of scope.
5. **The AIND reusable workflows are a reference structure, not a verbatim requirement.** We adapt them to our packages rather than calling them unmodified. This matters because several of them cannot be used as-is (see below).

## What the AIND reusable workflows actually enforce

The standards page states: *"GitHub automation must use the AIND [reusable workflows](https://github.com/AllenNeuralDynamics/.github/tree/main/.github/workflows)."* That repo holds 21 workflows in two generations. The modern (uv/ruff) generation is what we mirror:

| Workflow | Command it runs |
|---|---|
| `lint.yml` | `uv run ruff check` — Python **3.11 / 3.12 matrix**, `astral-sh/setup-uv@v5` |
| `test.yml` | `uv run pytest tests/` — same matrix |
| `type.yml` | `uv run mypy src/mypackage --strict` |
| `release-bump-version-uv.yml` | conventional-commit version calc → `uv version` + `uv lock` → commit `ci: version bump` → push tag |
| `test-ci.yml` *(legacy)* | `flake8` + `interrogate` + `coverage` — the old `aind-library-template` stack; **do not use** |

`workflow-templates/test_lint_type.yml` shows the intended composition: lint and type check in parallel, tests gated behind both, on PRs to `$default-branch` and `dev`.

**This confirms the ruff + uv decision** — it is what CI actually runs — and surfaces four things the original draft of this plan got wrong or missed:

1. **`mypy --strict` is required.** Not a formatting concern. The library returns bare unparameterized `dict`s and the batch launcher has three functions with unannotated params/returns; strict mypy rejects both. New issues **L-12** and part of **BA-5**.
2. **Python 3.12 must pass.** The library declares `>=3.11` and nothing currently tests 3.12.
3. **Versioning works differently than assumed.** `release-bump-version-uv.yml` runs `uv version <new>` and `uv lock`, so the version **stays literal in `pyproject.toml`** and the workflow rewrites it. The `__init__.__version__` + `dynamic = ["version"]` pattern belongs to the *legacy* `release-tag.yml`. **L-2 is corrected accordingly** — no `__version__` attribute is needed.
4. **Adaptations we must make** (per decision 5):
   - `type.yml` hardcodes `src/mypackage` with no input parameter — unusable for any repo as written. We write our own equivalent mypy job.
   - `test.yml` has no coverage gate. We enforce 100% via `addopts = "--cov=aind_motion_energy --cov-fail-under=100"` in `[tool.pytest.ini_options]`, so the gate applies even under a bare `pytest tests/` invocation.
   - The template pins `runner_label: eng-tools`, a self-hosted runner. Drop that block for `ubuntu-latest` unless we have access.
   - Neither capsule is a uv project, so `uv run …` steps don't apply there; capsule CI installs linters with pip directly (see C-4, BA-6).

## Justified deviations

The AIND guidance permits per-package deviation where justified. We take four, and each should be recorded in `docs/standards-compliance.md`:

| Deviation | Justification |
|---|---|
| Repos keep the `aind-` prefix instead of `<modality>-<process>` | Renaming breaks the capsule Dockerfile git pin, the console-script name, the import path, and Code Ocean wiring, for no functional gain. Rule applies to new repos. |
| pytest instead of `unittest` | Already the library's convention; the modern AIND `test.yml` runs `uv run pytest tests/`, so this matches current CI anyway. |
| Capsules are not uv projects; CI uses `pip install ruff mypy` rather than `uv run …` | A Code Ocean capsule is executed, never installed. It is not a Python package, so it has no `[project]` table and no build backend for uv to drive. |
| Capsule branching differs from `dev`/`main` PR flow | The Code Ocean web IDE pushes commits directly to the capsule's attached branch and cannot open PRs, so that branch cannot be protected. The library complies fully; the capsules keep an unprotected `main` attached to CO with `dev` as a staging branch. See X-2. |

### Capsule lint/type config

Both capsules get a `pyproject.toml` containing **only `[tool.*]` sections** — no `[project]` table, no `[build-system]`. A `pyproject.toml` without a `[project]` table is not a package and cannot be pip-installed; it is purely the standard location tools read config from. This keeps lint settings in the same filename across all three repos while preserving the capsule-is-not-a-package boundary established in capsule commit `1a86686`. Include a header comment stating the constraint:

```toml
# NOT a package. Tool configuration only.
# Do not add a [project] table — this capsule is
# executed by Code Ocean, never installed.

[tool.ruff]
line-length = 100

[tool.mypy]
strict = true
```

---

## Breaking / high-coordination changes

Five items change contracts across repo boundaries. Everything else in this plan is independently mergeable.

| # | Change | Blast radius | Sequencing |
|---|---|---|---|
| **B-1** | **Typed metadata model** (L-8) — `compute_motion_energy` currently returns a 4-tuple whose last element is a bare `dict`, serialized verbatim to `{stem}_me_metadata.json`. Replacing it with a typed model changes both the Python return type and the JSON key set. | Library public API; `example_motion_energy.ipynb`; any downstream consumer of `_me_metadata.json`; the capsule's `processing.json` writer. | Land in the library, cut a **minor** release, then bump the capsule Dockerfile pin in the same PR as C-3. Keep the JSON key names identical if possible so only the Python type changes. |
| **B-2** | **Tagged releases replace SHA pins** (L-10) — the capsule Dockerfile installs `…@2e62934523a7b0cc3f7a25a7de474963bfc694df`. Standards require GitHub Releases and semantic versioning. | Capsule Dockerfile; forces a Code Ocean environment rebuild. | Release workflow must exist first (L-9), then C-2 changes the pin to `@v0.2.0`. |
| **B-3** | **`dev`/`main` branch strategy + conventional commits** (X-2) — all three repos are single-`main` today with imperative-sentence commit messages. | The library gets protected `dev` + `main`, so no more direct pushes there. The capsules keep an unprotected `main` for the Code Ocean web IDE ("Edited run", "Edited sessions.json") plus a `dev` staging branch. | Do this **first**, before any other PR. |
| **B-4** | **Capsule regains a Python entrypoint** (C-3) — writing `processing.json` requires Python, so `code/run` gains a thin `run_capsule.py` alongside the CLI call. This partially reverses capsule commit `1a86686`. | Capsule `code/run` contract; adds a file to `/results`. Additive for consumers, but re-opens the door to logic creeping back into the capsule. | Keep the new module strictly provenance-only — it must not reimplement compute, discovery, or plotting. Add a CI check or a comment guard. |
| **B-5** | **Batch launcher config externalization** (BA-2) — `ME_CAPSULE_ID`, `SCRATCH_BUCKET`, `SCRATCH_PREFIX` (currently the personal path `matt.becker/motion_energy`), `CO_DOMAIN`, and `RESULT_TAGS` move from module constants to a config file / CLI args. | Anyone with a saved Code Ocean run configuration for the batch capsule. | Ship with the current values as defaults so existing invocations keep working; make the config file optional. |

**Non-breaking but worth flagging:** the library requires Python `>=3.11` while both capsule Dockerfiles start from a `python3.9` Code Ocean base image (the ME capsule does `conda install python=3.11` on top; the batch capsule never bumps). That is a latent fragility, addressed in C-2 and BA-7.

---

## Issue list

30 issues in four groups: 3 cross-cutting, 12 library, 7 capsule, 8 batch. Within a group, issues are independent unless a **Depends on** line says otherwise. See **Order of operations** below for the dependency graph and phasing.

### Group X — Cross-cutting (do first)

**X-1 · Land this plan in `aind-motion-energy`** — 🥇 **first action item**
Commit this document to [aind-motion-energy/](https://github.com/AllenNeuralDynamics/aind-motion-energy) as `docs/standards-compliance.md`, establishing it as the organizing repo for the effort. Open a tracking issue there listing all 30 items as checkboxes grouped by phase, and file the individual issues (or file them lazily, phase by phase — the tracking issue is the source of truth either way).

Carry over the **Justified deviations** table verbatim; it is the artifact AIND's per-package deviation allowance expects, and it currently records four deviations, not two.

*Do this before X-2.* Branch protection does not exist yet, so this is a direct push to `main`; once X-2 lands, the same change would need a PR into `dev`.
*Size: XS*

**X-2 · Branch strategy and commit conventions** — ⚠️ breaking (B-3)
Two different layouts, because Code Ocean pushes commits directly to a capsule's attached branch and cannot open PRs — so that branch can never carry protection requiring review.

**`aind-motion-energy` (library) — full compliance.** Create `dev` from `main`, set `dev` as default, protect both with 1 required human approval, squash-merge only. Feature branches → PR → `dev` → PR → `main`.

**Both capsules — documented deviation.** Code Ocean stays attached to `main` (confirmed: the CO UI git integration allows branch selection, so this is a deliberate choice, not a constraint). `main` is therefore unprotected and is what CO's "latest" resolves to. `dev` exists as a staging branch for multi-commit programmatic work and merges into `main`. No protection on either branch; review happens by convention.

*Rationale for CO → `main` rather than CO → `dev`:* a capsule run with `version=None` resolves to the attached branch's HEAD. Attaching CO to `dev` would mean any consumer that forgets to pin a version silently runs development code. Pointing CO at `main` keeps the default safe regardless of pinning discipline. The cost is two-directional syncing — merge `main` into `dev` before starting work, and `dev` into `main` to ship — which is acceptable at these repos' commit volume. Reversible with one CO setting if the one-way flow later proves more valuable.

Document conventional-commit PR titles (`feat:` → minor, `fix:` → patch, `feat!:`/`BREAKING CHANGE` → major) in each `CONTRIBUTING.md`, and the capsule branch layout above.
*Size: S · Blocks: every other PR*

**X-3 · Security sweep**
Confirm no `.env` files or credentials in any history. Review `aind-motion-energy-batch-capsule/.codeocean/secrets.json`, which commits Code Ocean secret **slot IDs** (`NBh1G7FJBe2zoVdd`, `LCRJcVLBRyhKGzL4`) and the assumable-role name — these are handles, not secrets, but decide whether they belong in a public repo. Verify the `API_SECRET` access token has ≤6-month expiry.
*Size: S*

---

### Group L — `aind-motion-energy` (library)

**L-1 · Adopt ruff; reformat**
Add `[tool.ruff]` to `pyproject.toml` with `line-length = 100`, PEP 8 rules, and complexity/length checks (`C901` max-complexity 10, function length ≤100 lines — these back L-6). Add `ruff` to dev dependencies. Run `ruff check --fix` and `ruff format`. Also fix `src/aind_motion_energy/io.py:6` (unsorted `aind_video_utils` import names) and **consolidate the duplicated dev deps** — `[project.optional-dependencies].dev` and `[dependency-groups].dev` both declare pytest/pytest-cov with conflicting floors (8.0/5.0 vs 9.0.3/7.1.0).
*Size: S · Touches: `pyproject.toml`, all of `src/`, all of `tests/`*

**L-2 · Packaging metadata and versioning**
Add `license = {text = "MIT"}` and a `LICENSE` file (Allen Institute for Neural Dynamics), replace the personal author entry (`105821807+1mattbecker@users.noreply.github.com`) with the institute, and add classifiers. Apply the dependency-pinning rule to the internal dep: `aind-video-utils>=0.3.1,<1`. Optionally add `CITATION.cff`, which the release workflow will keep in sync.
**Keep `version = "0.1.0"` literal in `pyproject.toml`** — the uv release workflow rewrites it via `uv version`. Do *not* add a `__version__` attribute or `dynamic = ["version"]`; that is the legacy setuptools path.
*Size: S · Blocks: L-9*

**L-3 · `.github/` scaffolding**
Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates (bug / feature / user story), a PR template, and `CODEOWNERS`. Ensure a GitHub Team has Maintainer access.
*Size: S*

**L-4 · CI: lint, type, and test workflow**
Add `.github/workflows/test_lint_type.yml` adapted from the AIND template: lint and type check in parallel, tests gated behind both, on PRs to `main` and `dev`. Mirror `lint.yml` (`uv run ruff check` plus `ruff format --check`) and `test.yml` (`uv run pytest tests/`) on the **Python 3.11 / 3.12 matrix** with `astral-sh/setup-uv`. Set the coverage gate to whatever `pytest --cov` reports on the day this lands, so it is meaningful immediately; L-7 raises it to 100.
**Lint and test jobs only.** The mypy job is added by L-12, not here — wiring it now would leave CI red for the whole rollout. We write our own mypy job at that point rather than calling `type.yml`, which hardcodes `src/mypackage`. Drop `runner_label: eng-tools` for `ubuntu-latest` unless we have runner access.
**Expect 3.12 failures** — nothing has ever tested the library there.
*Size: M · Depends on: L-1*

**L-5 · numpydoc docstrings and modern type hints**
Convert every docstring to numpydoc `Parameters`/`Returns` sections. Fill the missing ones: module docstrings for `io.py`, `compute.py`, `viz.py`, `cli.py`; function docstrings for `get_video_info` and `cli.main`. Replace the ad-hoc `Returns (a, b, c, d):` bullet list in `compute_motion_energy`. Modernize `typing.Optional`/`Tuple`/`Generator` to PEP 585/604 (valid at `>=3.11`) and parameterize the bare `dict` returns. Add a docstring-coverage gate to CI.
*Size: M · Touches: all of `src/`*

**L-6 · Replace `print()` with `logging`**
[`cli.py`](../src/aind_motion_energy/cli.py) uses bare `print()` at lines 86, 118–119, 127, 141. Add a module logger, a `--log-level` flag, and structured start/finish/per-video messages. Keep the library itself silent by default (`logging.NullHandler`); configure handlers only in `main()`.
*Size: S*

**L-7 · Tests for `cli.py` and `viz.save_summary_plots`; 100% coverage gate**
The largest work item. `cli.py` has zero tests and `save_summary_plots` has zero tests — together they are the entire gap to the required 100%. Add `tests/test_cli.py` covering argument parsing, video discovery, the `stem = video.parent.name if video.stem == "video"` camera-keying heuristic, each `--format` branch, and each optional-output branch; add `save_summary_plots` tests to `tests/test_viz.py`. Add `[tool.coverage.report] fail_under = 100` with `omit = ["*__init__*"]`, and put the gate in `[tool.pytest.ini_options] addopts = "--cov=aind_motion_energy --cov-fail-under=100"` so it applies even under a bare `pytest tests/` invocation. Reuse the existing `_FakePlane` fixture pattern from [`tests/test_io.py`](../tests/test_io.py).
*Size: L · Depends on: L-4 (raises its gate)*

**L-8 · Typed metadata model** — ⚠️ breaking (B-1)
Replace the bare metadata `dict` returned by `compute_motion_energy` with a typed model (pydantic `BaseModel` or a frozen dataclass) covering the existing keys: `video_path, n_frames_decoded, n_me_frames, n_keyframes_masked, fps, codec_name, width, height, bit_depth, roi, normalized, pixel_count, start_frame, end_frame`. **Keep the serialized JSON key names identical** so `_me_metadata.json` consumers are unaffected and only the Python type changes. This model is what the capsule reads to build its `DataProcess` (C-3) — that consumer requirement should drive the field set.
*Size: M · Blocks: C-3 · Cut a minor release after merging*

**L-9 · Release workflow and first tagged release** — ⚠️ breaking (B-2)
Adapt `release-bump-version-uv.yml` to run on merge to `main`. It computes the bump from conventional-commit prefixes, runs `uv version` + `uv lock`, commits `ci: version bump [skip actions]`, and pushes the tag — so it needs a `repo-token` secret with push rights to a protected branch (coordinate with X-2). Cut `v0.2.0` with autogenerated GitHub Release notes. GitHub Releases only, no PyPI — the sole consumer is the capsule Dockerfile.
*Size: M · Depends on: L-2, X-2 · Blocks: C-2*

**L-10 · MkDocs, Read the Docs, and `examples/`**
Replace the two hand-written notes with a MkDocs + mkdocstrings site: `mkdocs.yml`, `docs/index.md`, an API reference generated from docstrings, and pages carrying over [`docs/design-notes.md`](design-notes.md) and [`docs/optimization-backlog.md`](optimization-backlog.md). Add `.readthedocs.yaml` and connect the RTD project. Move `notebooks/ibl_vs_aind_motion_energy_comparison.ipynb` to `examples/` per the folder standard.
*Size: M · Depends on: L-5*

**L-11 · README overhaul**
Add the required **support badge**, plus license / CI / coverage / python-version badges. Add installation, CLI usage with a full flag reference, an outputs section documenting the file-naming convention (including the camera-keying heuristic, which the capsule silently depends on), and links to the two capsule repos. **Fix the stale instruction** telling users to `uv export … > environment/requirements.txt` — `environment/` was deleted in commit `3b3d65d`.
*Size: S*

**L-12 · Pass `mypy --strict`**
Required by the AIND `type.yml` workflow and not previously in scope. Add `mypy` to dev deps and a `[tool.mypy]` block with `strict = true`. Expect real work: `get_video_info` returns a bare `dict`, `compute_motion_energy`'s 4-tuple includes an unparameterized `dict`, and `av`/`tqdm` may need `ignore_missing_imports` or stub packages. Do this **after L-5** (type-hint modernization) and ideally **after L-8**, since the typed metadata model removes the worst offenders. Wire the job into L-4's workflow.
*Size: M · Depends on: L-5 · Lands in Phase 5, immediately after L-8 — the typed model removes most of the work*

---

### Group C — `aind-motion-energy-capsule`

**C-1 · README and parameter contract**
Currently two lines. Document: required input assets and mount layout, every output file and its naming, the `"$@"` passthrough contract (which today exists only as a comment inside `code/run`), which parameters are baked in by the capsule (`--summary-plots`) versus supplied by the batch launcher, and a link to the library repo. Add the support badge.
*Size: S*

**C-2 · Dockerfile hardening** — ⚠️ related to B-2
Pin the library to the release tag from L-9 instead of the bare SHA. **SHA-pin `aind-dynamic-foraging-behavior-video-analysis@main`** — a branch pin is the one real reproducibility hole in the image. Pin `pynwb`. Add the standards-required `pip list > /results/pip_list.txt`. Re-evaluate the `apt-get install ffmpeg` line, which the adjacent comment itself notes is redundant with PyAV's bundled ffmpeg. Consider moving to a base image that is already Python 3.11 rather than `conda install python=3.11` over a `python3.9` image.
*Size: M · Depends on: L-9*

**C-3 · Emit `processing.json` via aind-data-schema** — ⚠️ breaking (B-4)
Add a thin `code/run_capsule.py` that calls the library, then writes a `Processing`/`DataProcess` record to `/results` capturing code URL, library version, parameters, and input/output locations. **Strictly provenance only** — no compute, discovery, or plotting logic may live here; that was deliberately removed in commit `1a86686` and must not return. Consumes the typed model from L-8.
*Size: M · Depends on: L-8*

**C-4 · Capsule repo scaffolding and lint**
Add `metadata/metadata.yml` (missing entirely — `.gitignore` references it but it was never committed), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates. Fix the `LICENSE` copyright year (2023 on a 2026 repo).
Add a `pyproject.toml` with `[tool.ruff]` / `[tool.mypy]` only — no `[project]` table, no `[build-system]` — per the "Capsule lint/type config" section above, including the header comment. Add CI running `pip install ruff mypy && ruff check code/` plus `shellcheck` over `code/run`.
*Size: S*

**C-5 · Decide the app-panel / parameter surface**
`.codeocean/app-panel.json` was deleted in `07ddf6f` — partly because its `results` entries hardcoded `video_motion_energy.png`, which the per-camera keying fix (`2e62934`) invalidated. There is now no CO UI parameter surface at all. **Judgement call for the team:** either restore an app panel aligned to the current naming, or document that the capsule is launcher-driven only. Note that a persistently-attached asset in the app panel previously caused the 403 "data asset already attached" failure recorded in the batch plan.
*Size: S · Needs a team decision before implementation*

**C-6 · Notebook cleanup and relocation**
⚠️ **Do the first step in Phase 0, before any branch changes.** The capsule working tree holds an uncommitted diff that slices the trace to the clip window in `render_motion_energy_video` — a real bug fix that exists nowhere but that working copy. Commit it on its own first; losing it to a branch operation would be an avoidable regression. The rest of this issue can then wait for Phase 3.
Move `code/example_motion_energy.ipynb` to `examples/` per the folder standard. Remove the notebook's re-implemented video discovery (its local `VIDEO_EXTS` omits `.tif`/`.tiff`, diverging from the library's set) and its hand-rolled summary figure, calling `save_summary_plots` instead. Strip stored outputs before committing.
*Size: S*

**C-7 · Detach the committed dataset pin**
`.codeocean/datasets.json` hardcodes data asset `f908dd4d-d7ed-4d52-97cf-ccd0e167c659` (mount `foraging_nwb_bonsai`). This is the same class of persistent attachment that caused the batch run's 403. Decide whether it should be attached per-run instead, or documented as a deliberate notebook-only dependency.
*Size: XS*

---

### Group BA — `aind-motion-energy-batch`

**BA-1 · README, docs, and repo hygiene**
Reconcile the local directory name (`aind-motion-energy-batch-capsule`) with the remote (`aind-motion-energy-batch`). Fix the README's mislink — it points `aind-motion-energy` at the *capsule* repo URL — and the stale "Set `ME_CAPSULE_ID`" instruction. Document the `batch_manifest.json` schema and every status value (`pending`, `no_input_asset`, `dry_run`, `running`, `compute_*`, `completed`). Fold the untracked `MOTION_ENERGY_BATCH_PLAN.md` into repo docs and drop its dead reference to a nonexistent plan file. Add the support badge.
*Size: S*

**BA-2 · Externalize hardcoded configuration** — ⚠️ breaking (B-5)
Move `ME_CAPSULE_ID`, `CO_DOMAIN`, `SCRATCH_BUCKET`, `SCRATCH_PREFIX`, and `RESULT_TAGS` out of module constants in [`code/run_capsule.py`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch/blob/main/code/run_capsule.py) into a config file and/or CLI arguments. The current `SCRATCH_PREFIX = "matt.becker/motion_energy"` bakes a personal namespace into a shared repo. Ship current values as defaults so existing invocations keep working.

**Pin `ME_CAPSULE_VERSION` — mandatory, not optional.** It is currently `None`, which resolves to the attached branch's HEAD, so batch results are not reproducible against any fixed capsule state. Publish a Code Ocean capsule version and pin it. This is the primary reproducibility guarantee for the pipeline; the X-2 branch layout is only a backstop.
*Size: M*

**BA-3 · Resilience: retries, per-session error isolation, incremental manifest**
Today there is **no `try`/`except` anywhere**, and the manifest is written only at the end of `main()` — so any HTTP error aborts the batch and loses all record of launched computations. Add: a `Retry` wrapper on the client (`total=5, backoff_factor=1, status_forcelist=[429,500,502,503,504]`), per-session exception capture into the manifest, incremental manifest writes after each state change, explicit `timeout` on `wait_until_completed` and `wait_until_ready` (both currently unbounded), and the attach-conflict self-heal (catch "data asset already attached" → detach → retry) called for in the batch plan.
*Size: M*

**BA-4 · Session resolution correctness**
`code/sessions.json` holds date-only names (`behavior_808054_2025-09-02`) with no `_HH-MM-SS`, so `resolve_raw_asset` never hits its exact-match branch and always falls through to the regex fallback, silently picking the lexicographically last match. Either require full stems (as the batch plan intends) or fail loudly on ambiguity rather than guessing. Add validation of the sessions file against the expected pattern.
*Size: S*

**BA-5 · Logging, docstrings, typing, ruff, mypy**
Replace the emoji `print()` calls (`⚠️`, `•`, `▸`, `✗`, `✓`) with `logging`. Add numpydoc docstrings to `make_client`, `load_sessions`, `trigger_run`, `capture_result`, `main`, and convert `resolve_raw_asset`'s prose docstring. Complete the type hints — `resolve_raw_asset`, `trigger_run`, and `capture_result` have unannotated params and returns (`resolve_raw_asset` should be `DataAsset | None`). Add the `[tool.ruff]` config from L-1 and get the module clean under `mypy --strict`; check whether the `codeocean` SDK ships type information, and add `ignore_missing_imports` if not.
*Size: M*

**BA-6 · Make the launcher testable, then test it**
Restructure so orchestration is importable rather than living behind `python -u run_capsule.py`, then add pytest tests with a mocked `codeocean` client covering asset resolution (both branches), fan-out, manifest state transitions, and each failure path from BA-3. Note the module hardcodes `RESULTS_DIR = Path("/root/capsule/results")`, which makes it unrunnable outside the container — make it overridable. Raise the coverage gate in the CI workflow that BA-8 established.
*Size: L · Depends on: BA-2, BA-3, BA-5 (Phase 4)*

**BA-7 · Dockerfile**
Bump from the `python3.9` base to Python 3.11 for consistency with the ME capsule and library. Pin `codeocean` to an exact version (currently `>=0.16.0`, unbounded). Add `pip list > /results/pip_list.txt`. Replace `# hash:placeholder` with a real environment hash — the environment has never been built from this file as written.
*Size: S*

**BA-8 · Batch repo scaffolding and CI**
`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, `CODEOWNERS`. Fix the `LICENSE` copyright year.
Add the `[tool.*]`-only `pyproject.toml` and the pip-based lint CI workflow, matching C-4. Tests and the coverage gate arrive later with BA-6 — this issue only brings lint online, so it stays small and unblocks Phase 1.
*Size: S*

---

## Order of operations

### Dependency graph

Only these edges are real constraints. Everything not shown is independent and can land in any order.

```
X-1 ──► X-2 ──► (every PR in every repo)
 │       (land the plan while main is still directly pushable)

L-1 ──► L-4 ──► L-7          (ruff config → CI → raise coverage gate to 100)
L-2 ──► L-9                  (packaging metadata → release workflow)
L-5 ──► L-10                 (docstrings → MkDocs API reference)
L-5 ──► L-12                 (type hints → mypy --strict)

L-8 ──► L-12                 (soft: typed model removes the worst mypy offenders)
L-8 ──► C-3                  (capsule reads the typed model to build DataProcess)
L-9 ──► C-2                  (release tag exists → Dockerfile can pin it)
C-2 ──► C-3 ──► BA-2         (capsule is correct → publish CO version → pin it)

BA-2, BA-3, BA-5 ──► BA-6    (config + resilience + typing → then test it all)
```

### Phases

Every issue appears exactly once. Within a phase, issues are independent and can run in parallel.

| Phase | Issues | Notes |
|---|---|---|
| **0 — Immediate**, in this order | **1.** X-1 · **2.** rescue C-6's uncommitted fix · **3.** X-2 · **4.** X-3 | **X-1 is the first action item**: land this plan in `aind-motion-energy` and open the tracking issue, while `main` is still directly pushable. Step 2 is in the *capsule* repo, so it does not compete with step 1 — commit that unversioned bug fix before any branch operation can lose it. X-2 comes third because it changes how every later PR lands. |
| **1 — Foundation** | L-1, L-2, L-3, C-4, BA-8 | All small, all independent, no dependencies between them. Capsule and batch CI both come online here (C-4, BA-8). |
| **2 — Library CI** | L-4 | Lint + test jobs only. |
| **3 — Quality** | L-5, L-6, L-11, C-1, C-6, C-7, BA-1, BA-4, BA-5, BA-7 | The widest phase; parallelizes freely across all three repos. |
| **4 — Long poles** | L-7, L-10, BA-3, BA-6 | Start L-7 and BA-6 as early as their dependencies allow — they are the two largest items in the plan. |
| **5 — Breaking sequence** | L-8 → L-12 → L-9 → C-2 → C-3 → BA-2 | **Strictly serial.** Each step's output is the next step's input. |
| **6 — Deferred** | C-5 | Blocked on a team decision about the app-panel parameter surface, not on any other issue. |

### Expected CI state during rollout

L-4 turns CI on before the code fully complies. This is intentional — the gates ratchet up rather than blocking Phase 1 — but it means some checks are knowingly red for a while. Do not treat these as regressions:

| Check | Added by | Passes from |
|---|---|---|
| `ruff check` / `ruff format --check` | L-4 | L-1 (already green) |
| `pytest` on 3.11 | L-4 | immediately |
| `pytest` on **3.12** | L-4 | unknown — never tested; may need a fix in Phase 2 |
| coverage gate | L-4 at **baseline**, raised to 100 by L-7 | L-7 |
| `mypy --strict` | **L-12**, not L-4 | L-12 |

Set L-4's coverage gate to whatever `pytest --cov` reports on the day it lands, so the check is meaningful immediately and only ever ratchets upward.

### Why Phase 5 is serial

This is the one ordering that cannot be relaxed, and each arrow is a real handoff:

1. **L-8** lands the typed metadata model and a minor release is cut.
2. **L-12** follows immediately — the typed model removes the bare `dict` returns that are the bulk of the `mypy --strict` work.
3. **L-9** adds the release workflow and cuts `v0.2.0`, creating the first tag.
4. **C-2** can now replace the capsule's bare SHA pin with `@v0.2.0` and rebuild the Code Ocean environment.
5. **C-3** adds `processing.json`, reading the typed model from step 1 through the image from step 4.
6. **BA-2** publishes a Code Ocean capsule version and pins `ME_CAPSULE_VERSION` to it — this must come last, so the pinned version contains all of the above.

## Verification

Per repo, after each PR:

```bash
# library
cd aind-motion-energy
uv sync --dev
uv run ruff check && uv run ruff format --check
uv run mypy --strict                                  # after L-12
uv run pytest --cov=aind_motion_energy --cov-report=term-missing --cov-fail-under=100
uv run mkdocs build --strict                          # after L-10
```

Confirm the 3.11/3.12 matrix locally before opening a PR, since CI runs both:

```bash
uv run --python 3.12 pytest tests/
```

End-to-end, after the breaking sequence lands:

1. **Library CLI on the local clip** — run `aind-motion-energy --input data/bottom_camera_clip_1_43.578s_to_73.578s.mp4 --output results/ --summary-plots` and confirm the full output set appears with unchanged names, and that `_me_metadata.json` keys match the pre-L-8 file byte for byte (modulo formatting).
2. **Capsule image** — rebuild the Code Ocean environment against the new release tag, confirm `pip list` lands in `/results/pip_list.txt`, and confirm `processing.json` validates against aind-data-schema.
3. **Batch dry run** — `python -u run_capsule.py --dry-run` against a 2–3 session `sessions.json`; confirm every session resolves to exactly one input asset and the manifest is written incrementally (kill the process partway and check the partial manifest survives).
4. **Batch real run** — one full session end to end; confirm the manifest shows `completed` with a non-null `result_asset_id`, the external asset appears under the configured scratch prefix, and `_me_metadata.json` reports a frame count matching full video length.
5. **CI** — open a throwaway PR into `dev` in each repo and confirm the workflow runs, the coverage gate enforces, and merging is blocked without an approval.
