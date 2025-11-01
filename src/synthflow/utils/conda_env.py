"""Utility functions for running commands in specific conda environments."""

import subprocess
from pathlib import Path
from typing import Optional


def run_in_conda_env(
    cmd: list[str],
    conda_env: str,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    shell: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command in a specific conda environment.

    Parameters
    ----------
    cmd : list[str]
        Command to run (as a list of arguments)
    conda_env : str
        Name of the conda environment to activate
    cwd : Optional[Path]
        Working directory for the command
    check : bool
        Whether to raise CalledProcessError on non-zero exit
    capture_output : bool
        Whether to capture stdout and stderr
    text : bool
        Whether to return output as text (not bytes)
    shell : bool
        Whether to use shell execution

    Returns
    -------
    subprocess.CompletedProcess
        Result of the subprocess execution

    Raises
    ------
    subprocess.CalledProcessError
        If check=True and command returns non-zero exit code
    """
    # Find conda installation
    conda_base = _find_conda_base()
    if conda_base is None:
        raise RuntimeError("Conda not found. Please ensure conda is installed and in PATH.")

    # Create command to activate conda env and run command
    # Use conda run to execute command in the environment
    conda_run_cmd = [
        str(conda_base / "bin" / "conda"),
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
    ] + cmd

    return subprocess.run(
        conda_run_cmd,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=text,
        shell=shell,
    )


def run_python_in_conda_env(
    python_script: str | Path,
    conda_env: str,
    args: Optional[list[str]] = None,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a Python script in a specific conda environment.

    Parameters
    ----------
    python_script : str | Path
        Path to Python script to run
    conda_env : str
        Name of the conda environment to activate
    args : Optional[list[str]]
        Additional arguments to pass to the script
    cwd : Optional[Path]
        Working directory for the command
    check : bool
        Whether to raise CalledProcessError on non-zero exit
    capture_output : bool
        Whether to capture stdout and stderr
    text : bool
        Whether to return output as text (not bytes)

    Returns
    -------
    subprocess.CompletedProcess
        Result of the subprocess execution

    Raises
    ------
    subprocess.CalledProcessError
        If check=True and command returns non-zero exit code
    """
    # Find conda installation
    conda_base = _find_conda_base()
    if conda_base is None:
        raise RuntimeError("Conda not found. Please ensure conda is installed and in PATH.")

    python_cmd = [
        str(conda_base / "bin" / "conda"),
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
        "python",
        str(python_script),
    ]

    if args:
        python_cmd.extend(args)

    return subprocess.run(
        python_cmd,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def _find_conda_base() -> Optional[Path]:
    """Find conda base installation directory."""
    import os

    # Check CONDA_PREFIX (if already in a conda env)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # Go up to base: CONDA_PREFIX is usually <base>/envs/<env_name>
        conda_base = Path(conda_prefix).parent.parent
        if (conda_base / "bin" / "conda").exists():
            return conda_base

    # Check CONDA_EXE (direct path to conda executable)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_base = Path(conda_exe).parent.parent
        if (conda_base / "bin" / "conda").exists():
            return conda_base

    # Try common conda locations
    home = Path.home()
    common_paths = [
        home / "anaconda3",
        home / "miniconda3",
        home / "conda",
        Path("/opt/conda"),
        Path("/usr/local/anaconda3"),
        Path("/usr/local/miniconda3"),
    ]

    for path in common_paths:
        if (path / "bin" / "conda").exists():
            return path

    # Try to find conda in PATH
    import shutil

    conda_path = shutil.which("conda")
    if conda_path:
        conda_base = Path(conda_path).parent.parent
        if (conda_base / "bin" / "conda").exists():
            return conda_base

    return None

