# SMMRT-DMET-ST-data

> **Accelerating ab initio spin-phonon relaxation simulation of single-ion magnets by quantum embedding and spatial truncation**
> Y. Deng et al., *J. Chem. Phys.* (2026)

Data archive for *ab initio* spin-phonon relaxation simulations of single-ion
magnets (DMET + CASSCF-SO / crystal-field / spin-phonon / relaxation analysis).

The companion code lives in **[ccme-tmc/LAMP_tau](https://github.com/ccme-tmc/LAMP_tau)**.
This repository holds the underlying input and output data for the three
compounds (DyAC, DybBr, DyCp) and for the spatial-truncation study: equilibrium
electronic-structure outputs, crystal-field parameters (CFPs), spin-phonon
coupling matrices, master-equation matrices and τ(T) data, phonon calculations,
figures, and the scripts that were used at run time.

**2345 files / 297.8 MiB**, byte-for-byte copies of the calculation output. The
files that were left out are listed in [EXCLUDED.md](EXCLUDED.md); `manifest.csv`
records the size and SHA-256 of everything that is archived.

## Contents

| Directory | System | Files | Size | Key contents | Runnable config |
|---|---|---:|---:|---|---|
| `Dyacac3_cell_Analyse/` | Dy(acac)₃ (DyAC), periodic cell | 203 | 32.0 MiB | `inpfile/CFPs_py.out`, `TPsv1/py0817_all_pt_sig10/` (42 matrices), `TPs_file/` (22 τ(T) files), VASP phonons, 27 figures | `inpfile/CFPs_v1.yml` |
| `DybbBr/` | DybBr | 609 | 47.1 MiB | `inpfile/CFPs_py.out`, `TPs/` (4 parameter sets, 104 matrices), `Figspy/` (444 figures), VASP phonons, 460 figures in total | `inpfile/CFPs_Re_test2.yml` |
| `DyCp_Orcapyscf/` | DyCp, ORCA vs. PySCF | 1203 | 131.7 MiB | `inpfile/CFPs_oc.out` + `CFPs_py.out`, `VPoc*.out` + `VPpy*.out`, `TPs/` (6 sets, 221 matrices), `Figsang/` (837 figures), analytic `.hess` | `inpfile/4f_ratecaulcu_input.yml` |
| `Spacetruncation/DyAC/` | DyAC, truncation series | 156 | 21.0 MiB | `inpfile/CFPs{,_new,_ori,_py,_test}.out`, `TPs_file/` (61 τ(T) files, one per truncation threshold) | `inpfile/CFPs_Re_test2.yml` |
| `Spacetruncation/DybBr/` | DybBr, truncation series | 57 | 11.0 MiB | `inpfile/CFPs_py.out`, `TPs_file/` (31 τ(T) files) | `inpfile/CFPs_Re_test2.yml` |
| `Spacetruncation/DyCp/` | DyCp, truncation series | 117 | 55.0 MiB | `inpfile/CFPs_oc.out`, `VPoc*.out`, `TPs/oc0108_all_p/` (18 matrices), `TPs_file/` (48 τ(T) files) | `inpfile/4f_ratecaulcu_input.yml` |

Folders are reproduced **verbatim** from the author's calculation archive,
including their original (slightly informal) names, so that file paths quoted in
notebooks and in the paper's SI still resolve.

## Using the data

The files are stored exactly as they were produced, in the original folder
layout: every YAML path is **relative to the system root** (the folder that
contains `inpfile/`), so `cd` into a system folder before pointing a script at
its `inpfile/*.yml`.

The YAML files are the ones actually used at run time and were written for the
code of that moment. The current analysis code is
[LAMP_tau](https://github.com/ccme-tmc/LAMP_tau); if it asks for a key an old
file does not have (typically `g_z`, `CFPs_Type`, `Ab_Type`, `If_Perb`,
`Perb_T`), copy the line from
[`examples/relaxation/inpfile/CFPs_v1.yml`](https://github.com/ccme-tmc/LAMP_tau/blob/main/examples/relaxation/inpfile/CFPs_v1.yml).
The YAML files were deliberately **not** patched: they are the author's record
of what was actually run, and this repository exists so that record stays
intact.

Two of them — `Spacetruncation/DyAC/inpfile/CFPs.yml` and `CFPs_Re.yml` — point
`mag_filename` at Molcas equilibrium logs the authors no longer keep; see
[EXCLUDED.md](EXCLUDED.md).

## Data dictionary

| Item | Meaning |
|---|---|
| `inpfile/CFPs*.out` | Crystal-field parameters B(k,q) of every displaced structure: 201 Cartesian modes × 10 displacement points, or 3 × xyz / normal-mode points depending on `CFPs_X`. |
| `inpfile/VP*.out` | Spin-phonon coupling element matrices ⟨Ψᵢ|H_s-ph|Ψⱼ⟩ per mode (complex, block format, header records `Total modes`, `Max total angular momentum`, `select_sigma`). |
| `inpfile/freq*.log`, `POSCAR`, `*.hess` | VASP phonon frequencies/eigenvectors, equilibrium structure, and the analytic Hessian used for the 4f-rate calculation. |
| `inpfile/eqpyout`, `eqout` | Equilibrium DMET+CASSCF-SO (PySCF) / CASSCF (Molcas, ORCA) outputs: g tensor, spin-orbit eigenstates, magnetic moment matrix SOC. |
| `inpfile/ESO*/`, `inpfile/ESO_TTO/` | Tensor-operator matrix elements ⟨J₁M₁|O_k^q|J₂M₂⟩ generated with EasySpin (`.eso`, real/imaginary parts). |
| `inpfile/Atom_list`, `dpxs` | Atoms retained by the spatial-truncation selection; periodic-cell displacement indices. |
| `TPs/`, `TPsv1/` | Master-equation matrices `.tp` (density-matrix propagator per temperature) and `.tps` (τ(T) solutions, one dataset per parameter choice). |
| `TPs_file/*.dat` | Final τ(T) curves: column 1 = 1/T, column 2 = ln τ, column 3 = flag. |
| `*.png`, `*.eps` | Figures as produced during the analysis (panel files in `mode*` subfolders belong to the orientation analysis). |
| `*.py`, `*.sh`, `*.ipynb` | Scripts **as they were at run time**. The maintained, packaged versions live in [LAMP_tau](https://github.com/ccme-tmc/LAMP_tau) (`mag_anal/`, `SPC_Tool/`); the two are not always byte-identical. |

Formats are described in detail in the code repository:
[`docs/input_formats.md`](https://github.com/ccme-tmc/LAMP_tau/blob/main/docs/input_formats.md)
and [`docs/workflow.md`](https://github.com/ccme-tmc/LAMP_tau/blob/main/docs/workflow.md).

## Size and manifest

* 2345 files, 312 274 833 bytes (297.8 MiB); largest file 11.2 MiB (`output_origin`).
* `manifest.csv` lists every file with its size and SHA-256, so a copy can be
  checked byte for byte if needed. `tools/` keeps the small script that produced
  this tree from the author's origin folder, together with the exclusion rules
  (explained in [EXCLUDED.md](EXCLUDED.md)).

```bash
python tools/build_archive.py --check                    # compare a clone with manifest.csv
python tools/build_archive.py --source <dir> --dest .    # rebuild from an origin folder
```

## What is not archived

Scheduler logs, binary checkpoints, bytecode caches, a stray PyMOL wheel,
duplicate VASP logs and 60 MiB of legacy Molcas spin-phonon element dumps were
left out; the four Molcas equilibrium logs the author marked as unused are also
absent. Sizes, reasons and how to regenerate them: [EXCLUDED.md](EXCLUDED.md).

## Provenance and caveats

* Every archived file is a **byte-identical** copy of the author's file; nothing
  was renumbered, reordered or re-formatted (that is why the checksums are
  meaningful).
* Several `*.out`, `*.log` and `*.sh` files still contain the author's cluster
  paths (e.g. `/fs2/home/...`, `/THL6/home/...`). They are kept for traceability
  and must be edited before re-running on another machine.
* Number formats follow the programs that wrote them (FORTRAN `D` exponents in
  some Molcas files, `e` in PySCF ones).
* `DyCp_Orcapyscf/` and `Spacetruncation/DyCp/` (resp. `DybbBr/`) share many
  byte-identical files with each other and with `Dyacac3_cell_Analyse/` on
  purpose: they are independent working copies, and git stores their content
  only once.

## License

Data and documentation: **CC BY 4.0** (see [LICENSE](LICENSE)).
The `*.py` snapshots are additionally available under MIT, as in the code
repository. Third-party programs that produced these files (VASP, OpenMolcas,
ORCA, PySCF/LAMP-emb, EasySpin) are **not** distributed here and keep their own
licenses.

## Citation

If you use these data, please cite the paper

> Y. Deng et al., "Accelerating ab initio spin-phonon relaxation simulation of
> single-ion magnets by quantum embedding and spatial truncation",
> *J. Chem. Phys.* (2026),

together with this archive (`SMMRT-DMET-ST-data`, v1.0.0) and the code
([LAMP_tau](https://github.com/ccme-tmc/LAMP_tau)). Machine-readable metadata is
in [CITATION.cff](CITATION.cff); the journal DOI will be added once it is
available, and a Zenodo DOI if the data set is additionally archived there.

## Contact

Open an issue on GitHub for problems with this archive. For the science,
contact the corresponding authors of the paper.
