# Motion Energy Pipeline — AIND Standards Compliance Plan

## Context

The motion energy pipeline spans three repos under `AllenNeuralDynamics`:

| Repo | Role |
|---|---|
| [`aind-motion-energy`](https://github.com/AllenNeuralDynamics/aind-motion-energy) | Python library + CLI — the algorithm. **This repo; the organizing repo for this effort.** |
| [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) | Code Ocean capsule; thin shell over the CLI |
| [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) | Code Ocean launcher; fans out one capsule run per session |

The pipeline works end to end, and the layering is already sound — the capsule is a genuine thin wrapper (`code/run` is a few lines forwarding `"$@"` to the library console script), and duplicated logic was deliberately removed in capsule commit `1a86686`. What is missing is engineering scaffolding the [AIND software practices](https://docs.allenneuraldynamics.org/en/latest/policies_practices/software_practices.html) require.

**Landed so far** (Phase 1 library items, merged to `dev` 2026-09-10): ruff config and reformat (L-1, #17), packaging metadata with `LICENSE`, `CITATION.cff`, institute authorship and the internal dependency pin (L-2, #18), and `.github/` scaffolding (L-3, #19). All three honored their trimmed scope — no `CODEOWNERS`, no PR template.

**Still outstanding:**

- **No CI in the library.** The capsules have a lint workflow; `aind-motion-energy` has no `.github/workflows/` at all. Ruff is configured but nothing runs it.
- **Tests cover neither `cli.py` nor `viz.save_summary_plots`** — together they are the entire gap to the required 100% coverage gate.
- **Docstrings are free-form prose**, not numpydoc; several modules and functions have none, and the bare `dict` returns are unparameterized.
- **No versioning or releases.** No tags; the capsule pins the library to a bare 40-char SHA.
- **No MkDocs site, no Read the Docs, no `examples/` folder.**
- **No `processing.json`** provenance from either capsule.

## Scope test

This plan changed shape after an audit against its own cited sources. **Every item below traces to a specific requirement in one of four places:**

1. The [AIND software practices page](https://docs.allenneuraldynamics.org/en/latest/policies_practices/software_practices.html).
2. The three template repos it names — [`aind-library-template`](https://github.com/AllenNeuralDynamics/aind-library-template), [`aind-capsule-template`](https://github.com/AllenNeuralDynamics/aind-capsule-template), [`aind-pipeline-template`](https://github.com/AllenNeuralDynamics/aind-pipeline-template).
3. The [AIND reusable workflows](https://github.com/AllenNeuralDynamics/.github/tree/main/.github/workflows) the page mandates.
4. AIND metadata standards (`aind-data-schema`) — the one item deliberately kept without a citation on the practices page; see **M-1**.

**Anything that did not trace to one of those was removed from compliance scope**, regardless of merit. The goal is the minimum change set that reaches compliance while keeping all three repos functional — not a general cleanup. Items cut for this reason are listed under [Out of scope](#out-of-scope) with their issue numbers; several are real defects and remain open as ordinary backlog.

### What the templates actually contain

The audit's most useful finding. The templates are much thinner than assumed:

| Template | Contents |
|---|---|
| `aind-library-template` | `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, 3 issue templates (bug / feature / user story), `LICENSE`, `CITATION.cff`, `dependabot.yml`, `link-issues-by-milestone.yml`, docs, src, tests. **No `CODEOWNERS`. No PR template.** |
| `aind-capsule-template` | `.codeocean/environment.json`, `.gitignore`, `LICENSE`, `README.md`, `code/run`, `code/run_capsule.py`, `environment/Dockerfile`. **Nothing else** — no CI, no `pyproject.toml`, no `CODE_OF_CONDUCT.md`, no issue templates, no `metadata/metadata.yml`, no `tests/`. |
| `aind-pipeline-template` | `metadata/metadata.yml`, `LICENSE`, `README.md`, `pipeline/nextflow.config`, two workflows. **This — not the capsule template — is where `metadata/metadata.yml` comes from.** |

Consequences: code ownership documentation (`CODEOWNERS`, PR template) is backed by nothing and was stripped in `9c64c0e` / capsule `e39ff91`; capsule scaffolding shrinks to lint config only; `metadata/metadata.yml` is not a capsule requirement.

Also confirmed absent from the practices page entirely: **any logging requirement** (structured logging is explicitly marked *TBD*, and the capsule template itself ships `set -ex`), and **any mention of `processing.json` or provenance**.

## Decisions

Settled with the user:

1. **Toolchain: follow the current standards page, not the older `aind-library-template`.** `ruff` (lint + format, `line-length = 100`), `uv` for package management, MkDocs + mkdocstrings on Read the Docs, numpydoc docstrings. The library already uses `uv` and `uv_build`, so this is the smaller lift. Note that the library template is a generation behind the page it is cited by — it ships `flake8`, `setup.py`, and Sphinx. **Where the page and the template disagree, the page wins.**
2. **Keep the current repo and package names.** The standards say new packages should be `<modality>-<process>` without an `aind-` prefix, but renaming would break the Dockerfile git pin, the console-script name, the import path, and Code Ocean capsule wiring for no functional gain. Treat the rule as applying to new repos; record the deviation.
3. **Provenance belongs in the capsule, not the library.** The library stays runtime-agnostic; the capsule builds the `DataProcess`/`Processing` record and writes `processing.json`. See **M-1**.
4. **Standards-first, minimal change.** Fix compliance and accept only the breaking changes compliance actually forces. Deeper redesign is out of scope.
5. **The AIND reusable workflows are a reference structure, not a verbatim requirement.** We adapt them rather than calling them unmodified, because several cannot be used as-is (see below).
6. **Coverage's 100% gate applies to the library only.** The page states it under a heading covering packages, capsules and pipelines, but `aind-capsule-template` ships no `tests/` directory and the page's own capsule guidance is "capsules should be thin wrappers to Python packages". Gating a thin wrapper at 100% is not what that sentence is for.

## What the AIND reusable workflows actually enforce

The standards page states: *"GitHub automation must use the AIND [reusable workflows](https://github.com/AllenNeuralDynamics/.github/tree/main/.github/workflows)."* That repo holds 21 workflows in two generations. The modern (uv/ruff) generation is what we mirror:

| Workflow | Command it runs |
|---|---|
| `lint.yml` | `uv run ruff check` — Python **3.11 / 3.12 matrix**, `astral-sh/setup-uv@v5` |
| `test.yml` | `uv run pytest tests/` — same matrix |
| `type.yml` | `uv run mypy src/mypackage --strict` |
| `release-bump-version-uv.yml` | conventional-commit version calc → `uv version` + `uv lock` → commit `ci: version bump` → push tag |
| `util-link-issues-by-milestone.yml` | backs the page's "PRs should always be linked with an issue that is part of a Milestone" |
| `test-ci.yml` *(legacy)* | `flake8` + `interrogate` + `coverage` — the old `aind-library-template` stack; **do not use** |

`workflow-templates/test_lint_type.yml` shows the intended composition: lint and type check in parallel, tests gated behind both, on PRs to `$default-branch` and `dev`.

**This confirms the ruff + uv decision** — it is what CI actually runs — and pins down four things:

1. **`mypy --strict` is required.** Not a formatting concern; it is a job in the reference composition. New issue **L-12**.
2. **Python 3.12 must pass.** The library declares `>=3.11` and nothing currently tests 3.12.
3. **Versioning works differently than assumed.** `release-bump-version-uv.yml` runs `uv version <new>` and `uv lock`, so the version **stays literal in `pyproject.toml`** and the workflow rewrites it. The `__init__.__version__` + `dynamic = ["version"]` pattern belongs to the *legacy* `release-tag.yml`. **L-2 is corrected accordingly** — no `__version__` attribute is needed.
4. **Adaptations we must make** (per decision 5):
   - `type.yml` hardcodes `src/mypackage` with no input parameter — unusable for any repo as written. We write our own equivalent mypy job.
   - `test.yml` has no coverage gate. We enforce 100% via `addopts` in `[tool.pytest.ini_options]`, so the gate applies even under a bare `pytest tests/`.
   - The template pins `runner_label: eng-tools`, a self-hosted runner. Drop that block for `ubuntu-latest` unless we have access.
   - Neither capsule is a uv project, so `uv run …` steps don't apply there; capsule CI installs `ruff` with pip directly (C-4, BA-8).

## Justified deviations

The AIND guidance permits per-package deviation where justified. We take four:

| Deviation | Justification |
|---|---|
| Repos keep the `aind-` prefix instead of `<modality>-<process>` | Renaming breaks the capsule Dockerfile git pin, the console-script name, the import path, and Code Ocean wiring, for no functional gain. Rule applies to new repos. |
| pytest instead of `unittest` | Already the library's convention; the modern AIND `test.yml` runs `uv run pytest tests/`, so this matches current CI anyway. |
| Capsules are not uv projects; CI uses `pip install ruff` rather than `uv run …` | A Code Ocean capsule is executed, never installed. It is not a Python package, so it has no `[project]` table and no build backend for uv to drive. |
| Capsule branching differs from `dev`/`main` PR flow | The Code Ocean web IDE pushes commits directly to the capsule's attached branch and cannot open PRs, so that branch cannot be protected. The library complies fully; the capsules keep an unprotected `main` attached to CO with `dev` as a staging branch. See X-2. |

### Capsule lint config

Both capsules get a `pyproject.toml` containing **only `[tool.*]` sections** — no `[project]` table, no `[build-system]`. A `pyproject.toml` without a `[project]` table is not a package and cannot be pip-installed; it is purely the standard location tools read config from. This keeps lint settings in the same filename across all three repos while preserving the capsule-is-not-a-package boundary established in capsule commit `1a86686`. Include a header comment stating the constraint:

```toml
# NOT a package. Tool configuration only.
# Do not add a [project] table — this capsule is
# executed by Code Ocean, never installed.

[tool.ruff]
line-length = 100
```

---

## Breaking / high-coordination changes

Two items change contracts across repo boundaries. Everything else in this plan is independently mergeable.

| # | Change | Blast radius | Sequencing |
|---|---|---|---|
| **B-1** | **`dev`/`main` branch strategy + conventional commits** (X-2) | The library gets protected `dev` + `main`, so no more direct pushes. The capsules keep an unprotected `main` for the Code Ocean web IDE plus a `dev` staging branch. | Done first, before any other PR. ✅ complete |
| **B-2** | **Tagged releases replace SHA pins** (L-9) — the capsule Dockerfile installs `…@2e62934…`. Standards require GitHub Releases and semantic versioning. | Capsule Dockerfile; forces a Code Ocean environment rebuild. | Release workflow must exist first (L-9), then C-2 changes the pin to `@v0.2.0`. |

**Non-breaking but worth flagging:** the library requires Python `>=3.11` while the ME capsule Dockerfile starts from a `python3.9` Code Ocean base image and does `conda install python=3.11` on top. Latent fragility, noted in C-2.

---

## Issue list

**23 issues** in five groups: 3 cross-cutting (complete), 10 library, 4 capsule, 5 batch, 1 metadata. Within a group, issues are independent unless a **Depends on** line says otherwise.

### Group X — Cross-cutting ✅ complete

**X-1 · Land this plan** (#1) — done in `cc1778a`, revised after the source audit.

**X-2 · Branch strategy and commit conventions** (#2) — breaking (B-1).
`aind-motion-energy` (library) — full compliance. `dev` created from `main`, `dev` default, both protected with 1 required human approval, squash-merge only. Feature branches → PR → `dev` → PR → `main`.
Both capsules — documented deviation. Code Ocean stays attached to `main`, which is therefore unprotected. `dev` exists as a staging branch and merges into `main`.
*Rationale for CO → `main` rather than CO → `dev`:* a capsule run with `version=None` resolves to the attached branch's HEAD. Attaching CO to `dev` would mean any consumer that forgets to pin a version silently runs development code. Pointing CO at `main` keeps the default safe regardless of pinning discipline.
Conventional-commit PR titles (`feat:` → minor, `fix:` → patch, `feat!:`/`BREAKING CHANGE` → major) documented in each `CONTRIBUTING.md`.

**X-3 · Security sweep** (#3) — done. No `.env` files or credentials in any history. `aind-motion-energy-batch/.codeocean/secrets.json` commits Code Ocean secret **slot IDs** and an assumable-role name — handles, not secrets. `API_SECRET` token expiry ≤6 months verified.

---

### Group L — `aind-motion-energy` (library)

**L-1 · Adopt ruff; reformat** (#4) — ✅ merged in #17
Add `[tool.ruff]` to `pyproject.toml` with `line-length = 100`, PEP 8 rules, and the page's *recommended* complexity/length checks (`C901` max-complexity 10, function length ≤100 lines). Add `ruff` to dev dependencies. Run `ruff check --fix` and `ruff format`. Also fix `src/aind_motion_energy/io.py:6` (unsorted `aind_video_utils` import names) and **consolidate the duplicated dev deps** — `[project.optional-dependencies].dev` and `[dependency-groups].dev` both declare pytest/pytest-cov with conflicting floors (8.0/5.0 vs 9.0.3/7.1.0), which will make CI's resolved versions unpredictable.
*Backing: page — PEP 8, 100 chars, ruff. Size: S · Touches: `pyproject.toml`, all of `src/`, all of `tests/`*

**L-2 · Packaging metadata and versioning** (#5) — ✅ merged in #18
Add a `LICENSE` file (the library has none; the template ships one) and `license = {text = "MIT"}`. Apply the dependency-pinning recommendation to the internal dep: `aind-video-utils>=0.3.1,<1`. Add `CITATION.cff` (present in the template; the release workflow keeps it in sync).
**Keep `version = "0.1.0"` literal in `pyproject.toml`** — the uv release workflow rewrites it via `uv version`. Do *not* add a `__version__` attribute or `dynamic = ["version"]`; that is the legacy setuptools path.
**Trimmed:** replacing the personal author entry is *not* required — the page marks per-author `pyproject.toml` attribution as **TBD**. Do it if convenient, but it is not a compliance item.
*Backing: page — semver, internal dependency pinning; template — LICENSE, CITATION.cff. Size: S · Blocks: L-9*

**L-3 · `.github/` scaffolding** (#6) — ✅ merged in #19
Add `CODE_OF_CONDUCT.md` and the three issue templates (bug / feature / user story) from `aind-library-template`. `CONTRIBUTING.md` already landed in `9c6d5df`. Ensure a GitHub Team has Maintainer access — the page requires at least one, added as Maintainer.
**Trimmed:** `CODEOWNERS` and the PR template are in **no** template and no standard; added in `615a798`, correctly stripped in `9c64c0e`. Do not re-add.
*Backing: template contents; page — GitHub Team as Maintainer. Size: XS*

**L-4 · CI: lint and test workflow** (#7)
Add `.github/workflows/test_lint_type.yml` adapted from the AIND template: lint and test on PRs to `main` and `dev`. Mirror `lint.yml` (`uv run ruff check` plus `ruff format --check`) and `test.yml` (`uv run pytest tests/`) on the **Python 3.11 / 3.12 matrix** with `astral-sh/setup-uv`. Set the coverage gate to whatever `pytest --cov` reports on the day this lands, so it is meaningful immediately; L-7 raises it to 100.
Also add `link-issues-by-milestone.yml` from the library template — the page requires PRs to be linked to an issue in a Milestone, and this is the automation that enforces it.
**Lint and test jobs only.** The mypy job is added by L-12, not here — wiring it now leaves CI red for the whole rollout. Drop `runner_label: eng-tools` for `ubuntu-latest` unless we have runner access.
**Expect 3.12 failures** — nothing has ever tested the library there.
*Backing: page — automation must use the reusable workflows, milestone-linked PRs. Size: M · Depends on: L-1*

**L-5 · numpydoc docstrings and modern type hints** (#8)
Convert every docstring to numpydoc `Parameters`/`Returns` sections. Fill the missing ones: module docstrings for `io.py`, `compute.py`, `viz.py`, `cli.py`; function docstrings for `get_video_info` and `cli.main`. Replace the ad-hoc `Returns (a, b, c, d):` bullet list in `compute_motion_energy`. Annotate every parameter and return type; modernize `typing.Optional`/`Tuple`/`Generator` to PEP 585/604 (valid at `>=3.11`) and parameterize the bare `dict` returns as `dict[str, Any]`.
**Trimmed:** no docstring-coverage CI gate. There is no such reusable workflow and the page does not ask for one; `interrogate` belongs to the legacy `test-ci.yml` stack we explicitly do not use.
*Backing: page — NUMPY-format docstrings, type-hint annotation. Size: M · Touches: all of `src/`*

**L-7 · Tests for `cli.py` and `viz.save_summary_plots`; 100% coverage gate** (#10)
The largest work item. `cli.py` has zero tests and `save_summary_plots` has zero tests — together they are the entire gap to the required 100%. Add `tests/test_cli.py` covering argument parsing, video discovery, the `stem = video.parent.name if video.stem == "video"` camera-keying heuristic, each `--format` branch, and each optional-output branch; add `save_summary_plots` tests to `tests/test_viz.py`. Add `[tool.coverage.report] fail_under = 100` with `omit = ["*__init__*"]`, and put the gate in `[tool.pytest.ini_options] addopts = "--cov=aind_motion_energy --cov-fail-under=100"` so it applies even under a bare `pytest tests/`. Reuse the existing `_FakePlane` fixture pattern from [`tests/test_io.py`](https://github.com/AllenNeuralDynamics/aind-motion-energy/blob/main/tests/test_io.py).
*Backing: page — "Coverage must be at 100%". Size: L · Depends on: L-4 (raises its gate)*

**L-9 · Release workflow and first tagged release** (#12) — breaking (B-2)
Adapt `release-bump-version-uv.yml` to run on merge to `main`. It computes the bump from conventional-commit prefixes, runs `uv version` + `uv lock`, commits `ci: version bump [skip actions]`, and pushes the tag — so it needs a `repo-token` secret with push rights to a protected branch (coordinate with X-2). Cut `v0.2.0` with autogenerated GitHub Release notes, which satisfies the changelog requirement. GitHub Releases only, no PyPI — the page says internal-use packages install directly from GitHub Releases, and the sole consumer is the capsule Dockerfile.
*Backing: page — semver, conventional commits, GitHub Releases, changelog. Size: M · Depends on: L-2 · Blocks: C-2*

**L-10 · MkDocs, Read the Docs, and `examples/`** (#13)
Replace the hand-written notes with a MkDocs + mkdocstrings site: `mkdocs.yml`, `docs/index.md`, an API reference generated from docstrings, and pages carrying over [`docs/design-notes.md`](design-notes.md) and [`docs/optimization-backlog.md`](optimization-backlog.md). Add `.readthedocs.yaml` and connect the RTD project. Move `notebooks/ibl_vs_aind_motion_energy_comparison.ipynb` to `examples/`.
*Backing: page — "must be generated using MkDocs with mkdocstrings hosted through Read the Docs", root-level `docs/` and `examples/`. Size: M · Depends on: L-5*

**L-11 · README overhaul** (#14)
Add the required **support badge** (`supported` or `unsupported` — the page allows only those two), plus license / CI / coverage / python-version badges. Add installation, CLI usage with a full flag reference, an outputs section documenting the file-naming convention (including the camera-keying heuristic, which the capsule silently depends on), and links to the two capsule repos. **Fix the stale instruction** telling users to `uv export … > environment/requirements.txt` — `environment/` was deleted in commit `3b3d65d`.
*Backing: page — support badge must be present and accurate. Size: S*

**L-12 · Pass `mypy --strict`** (#15)
A job in the reference `test_lint_type.yml` composition, so it is required. Add `mypy` to dev deps and a `[tool.mypy]` block with `strict = true`. Expect real work: `get_video_info` returns a bare `dict`, `compute_motion_energy`'s 4-tuple includes an unparameterized `dict`, and `av`/`tqdm` may need `ignore_missing_imports` or stub packages. `dict[str, Any]` from L-5 is sufficient — a typed model is **not** required to pass strict mode. Wire the job into L-4's workflow, writing our own step rather than calling `type.yml`, which hardcodes `src/mypackage`.
*Backing: reusable workflows — `type.yml`. Size: M · Depends on: L-5*

---

### Group C — `aind-motion-energy-capsule`

**C-1 · README and parameter contract** (#1)
Currently two lines. Add the support badge. Document: required input assets and mount layout, every output file and its naming, the `"$@"` passthrough contract (which today exists only as a comment inside `code/run`), which parameters are baked in by the capsule (`--summary-plots`) versus supplied by the batch launcher, and a link to the library repo.
*Backing: page — support badge. The rest is the minimum a README needs to be usable. Size: S*

**C-2 · Dockerfile: release pin and dependency log** (#2) — related to B-2
Pin the library to the release tag from L-9 instead of the bare SHA. Add `pip list > /results/pip_list.txt`, the page's recommended dependency log for containerized environments.
**Trimmed:** SHA-pinning `aind-dynamic-foraging-behavior-video-analysis@main`, pinning `pynwb`, re-evaluating the redundant `apt-get install ffmpeg`, and swapping to a Python 3.11 base image are all reasonable but none is required by any source. Fold them in opportunistically while the file is open; do not block on them.
*Backing: page — pip_list recommendation; L-9 makes the tag exist. Size: S · Depends on: L-9*

**C-4 · Capsule lint config and CI** (#4)
Add the `[tool.*]`-only `pyproject.toml` per the "Capsule lint config" section above, and CI running `pip install ruff && ruff check code/`. Fix the `LICENSE` copyright year (2023 on a 2026 repo). Largely landed in `c3b0729`; `e39ff91` correctly stripped the rest.
**Trimmed:** `metadata/metadata.yml` (that file belongs to `aind-pipeline-template`, not the capsule template), `CODE_OF_CONDUCT.md`, and issue/PR templates — `aind-capsule-template` contains none of them.
*Backing: page — ruff/PEP 8 applies to capsules; automation requirement. Size: XS*

**C-6 · Move the example notebook to `examples/`** (#6)
Move `code/example_motion_energy.ipynb` to `examples/` per the root-level folder requirement, stripping stored outputs before committing.
**Trimmed:** removing the notebook's re-implemented video discovery and hand-rolled summary figure is a genuine duplication fix but is not a standards item. Do it in the same PR if cheap; it is not what gates compliance.
*Backing: page — "examples must be in an `examples/` folder". Size: XS*

---

### Group BA — `aind-motion-energy-batch`

**BA-1 · README and repo hygiene** (#1)
Add the support badge. Reconcile the local directory name (`aind-motion-energy-batch-capsule`) with the remote (`aind-motion-energy-batch`). Fix the README's mislink — it points `aind-motion-energy` at the *capsule* repo URL — and the stale "Set `ME_CAPSULE_ID`" instruction. Document the `batch_manifest.json` schema and every status value (`pending`, `no_input_asset`, `dry_run`, `running`, `compute_*`, `completed`). Fold the untracked `MOTION_ENERGY_BATCH_PLAN.md` into repo docs and drop its dead reference to a nonexistent plan file.
*Backing: page — support badge. The rest is README accuracy. Size: S*

**BA-2 · Externalize hardcoded configuration** (#2)
Move `ME_CAPSULE_ID`, `CO_DOMAIN`, `SCRATCH_BUCKET`, `SCRATCH_PREFIX`, and `RESULT_TAGS` out of module constants in `code/run_capsule.py` into a config file and/or CLI arguments. `SCRATCH_PREFIX = "matt.becker/motion_energy"` bakes a personal namespace into a shared repo. Ship current values as defaults so existing invocations keep working. Pin `ME_CAPSULE_VERSION`, currently `None`, which resolves to the attached branch's HEAD.
**Scope note:** none of this is required by any cited source — the personal prefix is not a secret, and capsule version pinning is not a stated requirement. It is retained on reproducibility grounds and because M-1's provenance record is only meaningful against a pinned capsule version. If M-1 is deferred indefinitely, this should move out of scope too.
*Backing: none directly; supports M-1. Size: M*

**BA-5 · Docstrings, typing, and ruff** (#5)
Add numpydoc docstrings to `make_client`, `load_sessions`, `trigger_run`, `capture_result`, `main`, and convert `resolve_raw_asset`'s prose docstring. Complete the type hints — `resolve_raw_asset`, `trigger_run`, and `capture_result` have unannotated params and returns (`resolve_raw_asset` should be `DataAsset | None`). Add the `[tool.ruff]` config from L-1 and get the module ruff-clean.
**Trimmed:** replacing the emoji `print()` calls with `logging` moved to #9's sibling issue — the page's logging standard is TBD and the capsule template ships `set -ex`. `mypy --strict` on the batch launcher is also dropped: `type.yml` targets `src/<package>` in a uv project, which this is not.
*Backing: page — NUMPY docstrings, type hints, ruff, all under "All Packages (Python, Capsule, Pipeline)". Size: M*

**BA-7 · Dockerfile** (#7)
Replace `# hash:placeholder` with a real environment hash — the environment has never been built from this file as written, which makes the committed Dockerfile misleading. Add `pip list > /results/pip_list.txt`.
**Trimmed:** the Python 3.11 bump and an exact `codeocean` pin are not required — the batch launcher does not install the library, and `codeocean` is an external dependency, not an internal one.
*Backing: page — pip_list recommendation. Size: XS*

**BA-8 · Batch lint config and CI** (#8)
The `[tool.*]`-only `pyproject.toml` and the pip-based ruff CI workflow, matching C-4. Fix the `LICENSE` copyright year. Largely landed already.
**Trimmed:** same as C-4 — no `CODE_OF_CONDUCT.md`, no issue/PR templates, no `CODEOWNERS`.
*Backing: page — ruff/PEP 8, automation. Size: XS*

---

### Group M — Metadata

**M-1 · Emit `processing.json` across the pipeline** — 🔚 **last item; spans all three repos**
The one item retained without a citation on the practices page. AIND metadata standards expect derived data assets to carry a `processing.json`, and this pipeline produces derived assets, so it will be required eventually even though the software-practices page is silent on it. **Deliberately sequenced last and specified as a single cross-repo issue**, so its design can be settled later — with SciComp if needed — without blocking anything else in this plan.

Nothing else in the plan depends on M-1, and M-1 depends only on L-9 (a version string to record).

*Implementation sketch, from `aind-data-schema` v2.9.0 (`examples/processing.py`):*

```python
from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.processing import (
    DataProcess, Processing, ProcessName, ProcessStage,
)

p = Processing.create_with_sequential_process_graph(
    pipelines=[Code(name=..., url=..., version=..., input_data=[DataAsset(name=...)])],
    data_processes=[DataProcess(
        process_type=ProcessName....,
        stage=ProcessStage.PROCESSING,
        code=Code(url=<library repo>, version=<release tag from L-9>, parameters=<CLI args>),
        experimenters=[...],
        start_date_time=..., end_date_time=...,
        output_path=..., output_parameters=<motion energy metadata>,
    )],
)
p.write_standard_file(output_directory="/results")
```

*Open questions to resolve before implementing (candidates for SciComp):*
- **`ProcessName` has no motion-energy member.** Closest are `VIDEO_ROI_TIMESERIES_EXTRACTION` and `OTHER` (which requires `name` or `notes` to be populated). Either use `OTHER` with a descriptive `name`, or propose a new member to `aind-data-schema-models`.
- **Which repo writes the file.** Per decision 3, the capsule — but writing it requires Python, so the ME capsule regains a thin `code/run_capsule.py` alongside the CLI call. That partially reverses commit `1a86686`, so the new module must be **strictly provenance-only**: no compute, discovery, or plotting logic. The alternative is the batch launcher writing one record per fanned-out run, which keeps the ME capsule a pure shell but puts provenance a layer away from the computation.
- **Whether the library's metadata `dict` needs a typed model.** The old L-8 assumed yes. It is not needed for `mypy --strict` (`dict[str, Any]` suffices), so the only reason would be if M-1's `output_parameters` wants a validated shape. Decide here, not before. If a model is added, keep the serialized JSON key names identical so existing `_me_metadata.json` consumers are unaffected.
- **`aind-data-schema` version and whether `Processing` should be appended to** rather than created fresh, since the page's schema doc says the file "should be appended to with each subsequent stage".

*Backing: AIND metadata standards (`aind-data-schema`), not the software-practices page. Size: L · Depends on: L-9 · Blocks: nothing*

---

## Out of scope

Removed from compliance scope by the source audit. **These issues stay open as ordinary backlog** — several are real defects — but they are not part of this effort and are not on the tracking checklist.

| Issue | Why it left compliance scope |
|---|---|
| **L-6** (#9) · Replace `print()` with `logging` | The page marks structured logging **TBD: not yet required**. The capsule template itself ships `set -ex`. Real quality improvement, zero standards backing. |
| **BA-3** (#3) · Retries, per-session error isolation, incremental manifest | A genuine robustness gap — no `try`/`except` anywhere, and the manifest is written only at the end of `main()`, so an HTTP error loses all record of launched computations. No standard requires it. |
| **BA-4** (#4) · Session resolution correctness | A genuine bug — `sessions.json` holds date-only names, so `resolve_raw_asset` never hits its exact-match branch and silently picks the lexicographically last regex match. No standard requires it. |
| **C-5** (#5) · App-panel / parameter surface decision | A product decision about the Code Ocean UI, not a standards question. |
| **C-7** (#7) · Detach the committed dataset pin | A Code Ocean configuration decision, not a standards question. |

Closed outright:

| Issue | Why |
|---|---|
| **BA-6** (#6) · Make the launcher testable, then test it | Per decision 6, the 100% coverage gate applies to the library only. `aind-capsule-template` ships no `tests/` directory, and the page's capsule guidance is "thin wrappers". This was the largest item in the plan. |
| **L-8** (#11) · Typed metadata model | Not required by any source — `dict[str, Any]` satisfies both the type-hint requirement and `mypy --strict`. Folded into **M-1**, which is the only consumer that might justify it. |
| **C-3** (#3) · Emit `processing.json` via aind-data-schema | Superseded by **M-1**, which covers the same ground as one cross-repo issue sequenced last. |

Also **not** re-added, having been correctly stripped in `9c64c0e` and capsule `e39ff91`: `CODEOWNERS`, PR templates, capsule `CODE_OF_CONDUCT.md`, capsule issue templates, capsule `metadata/metadata.yml`. None appears in the template that governs its repo.

---

## Order of operations

### Dependency graph

Only these edges are real constraints. Everything not shown is independent and can land in any order.

```
X-1 ──► X-2 ──► (every PR in every repo)          ✅ complete

L-1 ──► L-4 ──► L-7          (ruff config → CI → raise coverage gate to 100)
L-2 ──► L-9                  (packaging metadata → release workflow)
L-5 ──► L-10                 (docstrings → MkDocs API reference)
L-5 ──► L-12                 (type hints → mypy --strict)
L-9 ──► C-2                  (release tag exists → Dockerfile can pin it)
L-9 ──► M-1                  (a version string to record in the provenance record)
```

Cutting L-8 and C-3 removed the plan's only serial multi-repo chain. Nothing now blocks on a breaking change except C-2, which waits on a tag.

### Phases

Every issue appears exactly once. Within a phase, issues are independent and can run in parallel.

| Phase | Issues | Notes |
|---|---|---|
| **0 — Immediate** ✅ | X-1, X-2, X-3 | Complete. |
| **1 — Foundation** | ~~L-1, L-2, L-3~~, C-4, BA-8 | Library items ✅ merged to `dev` (#17, #18, #19). C-4 and BA-8 are largely landed. |
| **2 — Library CI** | L-4 | Lint + test jobs, plus milestone linking. |
| **3 — Quality** | L-5, L-11, C-1, C-6, BA-1, BA-5, BA-7 | The widest phase; parallelizes freely across all three repos. |
| **4 — Long poles** | L-7, L-10, L-12 | L-7 is the largest remaining item; start it as soon as L-4 lands. L-12 needs L-5. |
| **5 — Release and pin** | L-9 → C-2 | Serial, two steps: cut the tag, then point the capsule at it. |
| **6 — Metadata** | M-1 (and BA-2, if kept) | Last, by design. Specify with SciComp input; nothing else waits on it. |

### Expected CI state during rollout

L-4 turns CI on before the code fully complies. This is intentional — the gates ratchet up rather than blocking Phase 1 — but some checks are knowingly red for a while. Do not treat these as regressions:

| Check | Added by | Passes from |
|---|---|---|
| `ruff check` / `ruff format --check` | L-4 | L-1 (already green) |
| `pytest` on 3.11 | L-4 | immediately |
| `pytest` on **3.12** | L-4 | unknown — never tested; may need a fix in Phase 2 |
| coverage gate | L-4 at **baseline**, raised to 100 by L-7 | L-7 |
| `mypy --strict` | **L-12**, not L-4 | L-12 |

Set L-4's coverage gate to whatever `pytest --cov` reports on the day it lands, so the check is meaningful immediately and only ever ratchets upward.

## Verification

Per repo, after each PR:

```bash
# library
cd aind-motion-energy
uv sync --dev
uv run ruff check && uv run ruff format --check
uv run mypy src/aind_motion_energy --strict            # after L-12
uv run pytest --cov=aind_motion_energy --cov-report=term-missing --cov-fail-under=100
uv run mkdocs build --strict                           # after L-10
```

Confirm the 3.11/3.12 matrix locally before opening a PR, since CI runs both:

```bash
uv run --python 3.12 pytest tests/
```

End to end, after Phase 5 lands:

1. **Library CLI on the local clip** — run `aind-motion-energy --input data/bottom_camera_clip_1_43.578s_to_73.578s.mp4 --output results/ --summary-plots` and confirm the full output set appears with unchanged names.
2. **Capsule image** — rebuild the Code Ocean environment against the new release tag and confirm `pip list` lands in `/results/pip_list.txt`.
3. **Batch dry run** — `python -u run_capsule.py --dry-run` against a 2–3 session `sessions.json`; confirm every session resolves to exactly one input asset.
4. **Batch real run** — one full session end to end; confirm the manifest shows `completed` with a non-null `result_asset_id` and `_me_metadata.json` reports a frame count matching full video length.
5. **CI** — open a throwaway PR into `dev` and confirm the workflow runs, the coverage gate enforces, and merging is blocked without an approval.
6. **After M-1 only** — confirm `processing.json` validates against `aind-data-schema`.
