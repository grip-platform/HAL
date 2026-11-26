"""Regression tests for the cross-platform test virtual-environment wrapper."""
import os
import subprocess
from pathlib import Path

import pytest

from tests._bash import find_bash


SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "test-venv.sh"


def _write_executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _bash_or_skip():
    bash = find_bash()
    if bash is None:
        pytest.skip("Git Bash is not installed")
    return bash


def _environment(tmp_path, bash):
    home = tmp_path / "home"
    home.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    bash_dir = str(Path(bash).resolve().parent) if bash != "bash" else "/bin"
    env = os.environ.copy()
    env.update(
        HOME=str(home),
        HAL_TEST_VENV=str(tmp_path / "venv"),
        PATH=os.pathsep.join((str(fake_bin), bash_dir)),
        MARKERS=str(tmp_path / "markers"),
    )
    env.pop("HAL_TEST_PYTHON", None)
    return env, fake_bin


def _candidate_stub(name, version="3 11", broken=False, empty_venv=False):
    if broken:
        return "#!/bin/sh\nexit 1\n"
    return f"""#!/bin/sh
case "$*" in
  *version_info*) echo "{version}" ;;
  *\"-m venv\"*)
    for venv_dir do :; done
    mkdir -p "$MARKERS" "$venv_dir"
    : > "$MARKERS/{name}"
    {'' if empty_venv else '''mkdir -p "$venv_dir/bin"
    cat > "$venv_dir/bin/python" <<'EOF'
#!/bin/sh
exit 0
EOF
    chmod +x "$venv_dir/bin/python"'''}
    ;;
esac
"""


@pytest.mark.parametrize(
    ("available", "winner"),
    [
        (("python3", "python", "py"), "python3"),
        (("python", "py"), "python"),
        (("py",), "py"),
    ],
)
def test_interpreter_order(tmp_path, available, winner):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    for name in ("python3", "python", "py"):
        _write_executable(
            fake_bin / name,
            _candidate_stub(name) if name in available else _candidate_stub(name, broken=True),
        )

    result = subprocess.run(
        [bash, str(SCRIPT)], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    markers = Path(env["MARKERS"])
    assert (markers / winner).is_file()
    assert sorted(path.name for path in markers.iterdir()) == [winner]


def test_broken_python3_falls_through_to_python(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    _write_executable(fake_bin / "python3", _candidate_stub("python3", broken=True))
    _write_executable(fake_bin / "python", _candidate_stub("python"))
    _write_executable(fake_bin / "py", _candidate_stub("py"))

    result = subprocess.run(
        [bash, str(SCRIPT)], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    markers = Path(env["MARKERS"])
    assert (markers / "python").is_file()
    assert not (markers / "python3").exists()


def test_python_310_is_skipped_for_python_311(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    _write_executable(fake_bin / "python3", _candidate_stub("python3", version="3 10"))
    _write_executable(fake_bin / "python", _candidate_stub("python", version="3 11"))
    _write_executable(fake_bin / "py", _candidate_stub("py", version="3 12"))

    result = subprocess.run(
        [bash, str(SCRIPT)], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    markers = Path(env["MARKERS"])
    assert (markers / "python").is_file()
    assert not (markers / "python3").exists()


def test_one_word_override_is_used(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    env["HAL_TEST_PYTHON"] = "override-python"
    _write_executable(fake_bin / "override-python", _candidate_stub("override"))

    result = subprocess.run([bash, str(SCRIPT)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (Path(env["MARKERS"]) / "override").is_file()


def test_two_word_override_is_used(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    env["HAL_TEST_PYTHON"] = "override-launcher -3"
    _write_executable(fake_bin / "override-launcher", _candidate_stub("override-two-word"))

    result = subprocess.run([bash, str(SCRIPT)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (Path(env["MARKERS"]) / "override-two-word").is_file()


def test_broken_override_falls_through_to_normal_search(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    env["HAL_TEST_PYTHON"] = "broken-override -3"
    _write_executable(fake_bin / "broken-override", _candidate_stub("override", broken=True))
    _write_executable(fake_bin / "python3", _candidate_stub("python3"))

    result = subprocess.run([bash, str(SCRIPT)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    markers = Path(env["MARKERS"])
    assert (markers / "python3").is_file()
    assert not (markers / "override").exists()


def test_existing_python_310_venv_is_rejected_without_deleting_it(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    _write_executable(fake_bin / "python3", _candidate_stub("python3"))
    venv = Path(env["HAL_TEST_VENV"])
    venv_python = venv / "bin" / "python"
    _write_executable(venv_python, _candidate_stub("venv", version="3 10"))

    result = subprocess.run([bash, str(SCRIPT)], env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert result.stderr.strip() == (
        f"test-venv.sh: virtual environment at {venv} does not use Python 3.11+; "
        f"delete {venv} and run this script again"
    )
    assert venv_python.is_file()


def test_empty_created_venv_reports_missing_python(tmp_path):
    bash = _bash_or_skip()
    env, fake_bin = _environment(tmp_path, bash)
    _write_executable(fake_bin / "python3", _candidate_stub("python3", empty_venv=True))
    _write_executable(fake_bin / "python", _candidate_stub("python", broken=True))
    _write_executable(fake_bin / "py", _candidate_stub("py", broken=True))

    result = subprocess.run(
        [bash, str(SCRIPT)], env=env, capture_output=True, text=True
    )

    assert result.returncode == 1
    assert result.stderr.strip() == (
        f"test-venv.sh: virtual environment at {env['HAL_TEST_VENV']} "
        "has no executable Python after creation"
    )


def test_python_mode_requires_existing_venv(tmp_path):
    bash = _bash_or_skip()
    env, _ = _environment(tmp_path, bash)

    result = subprocess.run(
        [bash, str(SCRIPT), "python", "-c", "print(1)"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"no venv at {env['HAL_TEST_VENV']} yet" in result.stderr
    assert "run 'bash bin/test-venv.sh' first" in result.stderr


@pytest.mark.parametrize("relative_python", ("bin/python", "Scripts/python.exe"))
def test_python_mode_discovers_both_venv_layouts(tmp_path, relative_python):
    bash = _bash_or_skip()
    env, _ = _environment(tmp_path, bash)
    output = tmp_path / "venv-python-output"
    venv_python = Path(env["HAL_TEST_VENV"]) / relative_python
    _write_executable(
        venv_python,
        f'#!/bin/sh\nprintf "%s\\n" "$*" > "{output}"\n',
    )

    result = subprocess.run(
        [bash, str(SCRIPT), "python", "-m", "pytest", "-q", "tests"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == "-m pytest -q tests"


def test_find_bash_non_windows_always_returns_bash():
    assert find_bash(platform="macos", which=lambda name: None) == "bash"


def test_find_bash_skips_system32_and_finds_git_bash(tmp_path):
    git = tmp_path / "Git" / "mingw64" / "bin" / "git.exe"
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    _write_executable(git, "")
    _write_executable(git_bash, "")

    def which(name):
        return {"bash": r"C:\\Windows\\System32\\bash.exe", "git": str(git)}.get(name)

    assert find_bash(platform="windows", which=which) == str(git_bash)


def test_find_bash_returns_none_when_windows_has_no_git_bash():
    def which(name):
        return r"C:\\Windows\\System32\\bash.exe" if name == "bash" else None

    assert find_bash(platform="windows", which=which) is None
