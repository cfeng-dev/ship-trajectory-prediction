# CmdStan Troubleshooting

## Installation fails on Windows

If the installation fails because commands such as `mingw32-make` or `cut`
cannot be found, ensure that the following directories are included in the user
`PATH` environment variable:

```text
%USERPROFILE%\.cmdstan\RTools40\mingw64\bin
%USERPROFILE%\.cmdstan\RTools40\usr\bin
```

Restart the terminal after updating `PATH`, then rerun:

```bash
uv run python -m cmdstanpy.install_cmdstan
```

For more detailed diagnostic output, run:

```bash
uv run python -m cmdstanpy.install_cmdstan --verbose --cores 1
```
