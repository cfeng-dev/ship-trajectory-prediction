"""Compare MATLAB and Python BR24 IMM diagnostics cycle by cycle."""

import bayestraj.validation.br24_imm_comparison as checks
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
INPUT_CSV = project_path("data/processed/BR24Dataset20131107.csv")
MATLAB_TRACE = project_path("data/processed/BR24_imm_matlab_trace.mat")
OUTPUT_CSV = project_path("results/validation/BR24_imm_implementation_comparison.csv")
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-8


def main(matlab_trace=None, output_csv=None, *, argv=()):
    """Run the MATLAB comparison with this script's settings."""
    return checks.run_check(
        input_csv=INPUT_CSV,
        matlab_trace=MATLAB_TRACE if matlab_trace is None else matlab_trace,
        output_csv=OUTPUT_CSV if output_csv is None else output_csv,
        imm_python=IMM_PYTHON,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
        description=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    main(argv=None)
