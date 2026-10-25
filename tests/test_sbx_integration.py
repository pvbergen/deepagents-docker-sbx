"""End-to-end tests against real sbx sandboxes.

These need a host where sbx can boot microVMs (a hypervisor, `sbx login`, and an initialized
`sbx policy`). They only run when opted in with ``DEEPAGENTS_SBX_INTEGRATION=1``:

    DEEPAGENTS_SBX_INTEGRATION=1 uv run pytest -m sbx
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from deepagents_docker_sbx import DockerSBX
from deepagents_docker_sbx._sbx import run_sbx, sbx_unavailable_reason


def _skip_reason() -> str | None:
    if os.environ.get("DEEPAGENTS_SBX_INTEGRATION") != "1":
        return "set DEEPAGENTS_SBX_INTEGRATION=1 to run sbx integration tests"
    reason = sbx_unavailable_reason(timeout=30)
    if reason is not None:
        return f"sbx is not available: {reason}"
    return None


_SKIP_REASON = _skip_reason()

pytestmark = [
    pytest.mark.sbx,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


def _sandbox_names() -> list[str]:
    return run_sbx(["ls", "-q"], timeout=30).stdout.split()


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DockerSBX]:
    backend = DockerSBX(shared_dir=tmp_path_factory.mktemp("sbx-shared"), timeout=60)
    try:
        yield backend
    finally:
        backend.close()


def test_execute_echo(sandbox: DockerSBX) -> None:
    result = sandbox.execute("echo hello-from-sbx")

    assert result.exit_code == 0
    assert "hello-from-sbx" in result.output


def test_working_directory_is_shared_dir(sandbox: DockerSBX) -> None:
    result = sandbox.execute("pwd")

    assert result.output.strip() == str(sandbox.shared_dir)


def test_default_template_has_python(sandbox: DockerSBX) -> None:
    result = sandbox.execute("python3 --version")

    assert result.exit_code == 0
    assert "Python 3" in result.output


def test_sandbox_writes_are_visible_on_host(sandbox: DockerSBX) -> None:
    result = sandbox.execute("echo from-sandbox > from_sandbox.txt")

    assert result.exit_code == 0
    assert (sandbox.shared_dir / "from_sandbox.txt").read_text().strip() == "from-sandbox"


def test_host_writes_are_visible_in_sandbox(sandbox: DockerSBX) -> None:
    write_result = sandbox.write("/from_host.txt", "from-host\n")
    assert write_result.error is None

    result = sandbox.execute("cat from_host.txt")

    assert result.output.strip() == "from-host"


def test_nonzero_exit_and_stderr(sandbox: DockerSBX) -> None:
    result = sandbox.execute("echo oops >&2; exit 3")

    assert result.exit_code == 3
    assert "[stderr] oops" in result.output


def test_timeout_kills_the_process_group(sandbox: DockerSBX) -> None:
    pid_file = f"timeout-{time.monotonic_ns()}.pid"

    # The backgrounded `sleep` records its own PID, so the check below targets exactly that
    # process (matching on command lines would also hit the checking command's own shells).
    result = sandbox.execute(f"sleep 300 & echo $! > {pid_file}; wait", timeout=2)

    assert result.exit_code == 124
    pid = (sandbox.shared_dir / pid_file).read_text().strip()
    check = sandbox.execute(f"kill -0 {pid} 2>/dev/null && echo alive || echo gone")
    assert check.output.strip() == "gone"


def test_output_cap_truncates_and_stops_command(tmp_path: Path) -> None:
    with DockerSBX(shared_dir=tmp_path, max_output_bytes=4096) as backend:
        result = backend.execute("yes aaaaaaaaaaaaaaaa")

    assert result.truncated is True
    assert "terminated after exceeding the output limit" in result.output


# The sbx proxy answers blocked HTTP(S) requests with a 403 rather than refusing the
# connection, so curl needs `-f` to turn a block into a nonzero exit code.
_PYPI_CURL = "curl -fsS --max-time 10 -o /dev/null https://pypi.org/simple/"


def test_outbound_traffic_allowed_by_default(tmp_path: Path) -> None:
    with DockerSBX(shared_dir=tmp_path) as backend:
        status = backend.execute(
            "curl -sS --max-time 10 -o /dev/null -w '%{http_code}' https://pypi.org/simple/"
        )
        if status.output.strip() == "403":
            pytest.skip("the global sbx policy blocks pypi.org; allow it to run this control")
        result = backend.execute(_PYPI_CURL)

    assert result.exit_code == 0, result.output


def test_outbound_traffic_can_be_denied(tmp_path: Path) -> None:
    with DockerSBX(shared_dir=tmp_path, allow_outbound_traffic=False) as backend:
        rules = run_sbx(["policy", "ls", backend.sandbox_name, "--wide"], timeout=30)
        result = backend.execute(_PYPI_CURL)

    assert "**" in rules.stdout, rules.stdout
    assert result.exit_code != 0


def test_close_removes_sandbox(tmp_path: Path) -> None:
    backend = DockerSBX(shared_dir=tmp_path)
    name = backend.sandbox_name
    assert name in _sandbox_names()

    backend.close()

    assert name not in _sandbox_names()
    assert tmp_path.exists()
