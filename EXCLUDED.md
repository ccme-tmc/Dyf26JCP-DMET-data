# What is not in this archive

The origin folder held 2397 files / 401.2 MiB. This archive publishes **2345
files / 297.8 MiB**. This page lists everything that was left out, why, and how
to get it back.

The rules themselves are machine-readable in [`tools/rules.yaml`](tools/rules.yaml)
and are applied by [`tools/build_archive.py`](tools/build_archive.py).

## Excluded by rule

| Pattern | Files | Size | Reason | How to regenerate |
|---|---:|---:|---|---|
| `VP*.log` | 30 | 60.1 MiB | Legacy Molcas spin-phonon element dumps (`<Ψ₁\|H_s-ph\|Ψ₂>` in long text form, one block per mode). The PySCF-path `VP*.out` matrices that carry the same physics are archived instead. | Re-run the Molcas spin-phonon step, or fit the archived `VP*.out` with `lamp-tau`. |
| `*.chk` | 2 | 12.5 MiB | Binary PySCF checkpoint files (`DyCp_rohf.chk`); not human-readable and fully rebuildable. | `eq/` contains `eq.xyz`, `jobeq.sh` and `dmet_cassiso.py`; rerunning the equilibrium job recreates the `.chk`. |
| `*.whl` | 1 | 11.8 MiB | A `pymol` wheel that was downloaded into the working folder by accident; third-party binary, unrelated to the paper. | `pip install pymol` (or `pymol-open-source`). |
| `nohup.out` | 2 | 11.6 MiB | Scheduler stdout of the cluster jobs. | Re-run the job; the scientific output is in `eqpyout` / `CFPs_py.out`. |
| `VASP.log` | 2 | 7.0 MiB | Byte-identical duplicate of `freq.log` in the same folder (same SHA-1); `freq.log` is archived. | — |
| `*.pyc` | 13 | 0.5 MiB | Python bytecode cache. | Re-import the module. |
| `**/__pycache__/**`, `**/.vscode/**` | 2 | ≈0 | Python cache, editor settings. | — |

Total: 52 files / 103.4 MiB.

Nothing above 50 MiB is allowed by the rules, which also keeps the archive well
below GitHub's 100 MiB per-file limit (the largest archived file is 11.2 MiB).

## Molcas equilibrium logs removed by the author

Four log files that existed in `Spacetruncation/DyAC/inpfile/` when the archive
was assembled are **no longer present in the origin folder**; the author
confirmed they belong to abandoned branches and are not needed:

| File | Size | Status |
|---|---:|---|
| `eq_new.log` | 211 739 388 B | Molcas equilibrium output, unused branch |
| `eq_exp.log` | 211 556 369 B | Molcas equilibrium output, unused branch |
| `eq_exp_new.log` | 347 379 B | Molcas equilibrium output, unused branch |
| `eq.log` | 330 435 B | Molcas equilibrium output, unused branch |

Consequences:

* `Spacetruncation/DyAC/inpfile/CFPs.yml` (`mag_filename: inpfile/eq_exp_new.log`)
  and `CFPs_Re.yml` (`mag_filename: inpfile/eq.log`) therefore cannot be run
  from this archive. They are kept unchanged because they document the protocol
  that was used; point `mag_filename` at `inpfile/eqpyout`, or re-run the Molcas
  equilibrium calculation, to use them.
* The other configurations found in this archive are complete except for a few
  required keys added later by the analysis code; the *Using the data* section
  of the [README](README.md) explains how to supply them.

## Intentionally not transformed

The author's cluster paths inside `*.out` / `*.log` / `*.sh` files were **not**
rewritten (unlike the sanitized copy shipped as an example in the code
repository). Keeping the files byte-identical is what makes `manifest.csv`
useful; edit the paths yourself before re-running on a different machine.
