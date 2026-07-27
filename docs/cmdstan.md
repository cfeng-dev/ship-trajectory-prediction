# CmdStan Setup

## Verify the installation

Start the Python interpreter in the project environment:

```bash
uv run python
```

Then execute:

```python
import cmdstanpy

print(cmdstanpy.cmdstan_path())
```

A valid installation path, for example

```text
C:\Users\<username>\.cmdstan\cmdstan-2.39.0
```

confirms that CmdStan has been installed successfully.

Exit the Python interpreter:

```python
exit()
```

## Troubleshoot installation on Windows

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
