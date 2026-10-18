IMM/

Local optional IMM files are not tracked in this repository:

- The MATLAB reference implementation provided by Prof. Reuter.
- `br24_imm.py`, the Python implementation used by the BR24 IMM export.

The Bayesian CTRV tracker and its BR24 data processing do not require these
files. The IMM export reports a missing implementation if `br24_imm.py` is
not present. The Python implementation has not yet been validated against
the MATLAB reference.
