import gc
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents_docker_sbx import DockerSBX, SBXError
from deepagents_docker_sbx._sbx import SBXRunResult
from deepagents_docker_sbx.backend import SANDBOX_AGENT


def _sbx_run_ok() -> SBXRunResult:
    return SBXRunResult(returncode=0, stdout="", stderr="")


def _make_run_sbx_side_effect(**exec_config: object):
    """Build a side_effect for mocked run_sbx (handles create/exec/stop/rm)."""

    def _run_sbx(args: list[str], **kwargs: object) -> SBXRunResult:
        if args[0] == "exec":
            if "error" in exec_config:
                raise exec_config["error"]
            return exec_config.get(
                "exec",
                SBXRunResult(returncode=0, stdout="", stderr=""),
            )
        return _sbx_run_ok()

    return _run_sbx


def _create_args(run_sbx: MagicMock) -> list[str]:
    args = run_sbx.call_args_list[0][0][0]
    assert args[0] == "create"
    return args


def _flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_create_uses_shell_agent_with_shared_dir_workspace(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        args = _create_args(run_sbx)
        assert args[-2:] == [SANDBOX_AGENT, str(tmp_path.resolve())]
        assert SANDBOX_AGENT == "shell"
        assert _flag_value(args, "--name") == sandbox.sandbox_name
        assert sandbox.sandbox_name == f"deepagents-{sandbox.id}"
        assert "--no-progress" in args
        assert len(sandbox.id) == 12
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_create_defaults_leave_sizing_image_and_network_to_sbx(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        args = _create_args(run_sbx)
        for flag in ("--template", "--memory", "--cpus", "--deny-network"):
            assert flag not in args
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_create_passes_custom_image_as_template(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path, image="my-org/sandbox:1")
    try:
        assert _flag_value(_create_args(run_sbx), "--template") == "my-org/sandbox:1"
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_create_disables_outbound_traffic_with_deny_all_rule(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path, allow_outbound_traffic=False)
    try:
        assert _flag_value(_create_args(run_sbx), "--deny-network") == "**"
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_create_applies_resource_limits_and_extra_args(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(
        shared_dir=tmp_path,
        memory="1g",
        cpus=2,
        extra_run_args=["--env", "FOO=bar"],
    )
    try:
        args = _create_args(run_sbx)
        assert _flag_value(args, "--memory") == "1g"
        assert _flag_value(args, "--cpus") == "2"
        assert _flag_value(args, "--env") == "FOO=bar"
        # Extra flags must come before the agent/workspace positionals.
        assert args.index("--env") < args.index(SANDBOX_AGENT)
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_raises_when_sandbox_create_fails(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = SBXRunResult(returncode=1, stdout="", stderr="template not found")

    with pytest.raises(SBXError, match="failed to create sandbox: template not found"):
        DockerSBX(shared_dir=tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": 0}, "timeout must be positive"),
        ({"cpus": 0}, "cpus must be a positive whole number"),
        ({"cpus": 1.5}, "cpus must be a positive whole number"),
        ({"max_output_bytes": 0}, "max_output_bytes must be positive"),
    ],
)
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_constructor_rejects_invalid_limits(
    _available: MagicMock,
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DockerSBX(**kwargs)


@patch(
    "deepagents_docker_sbx.backend.sbx_unavailable_reason",
    return_value="not signed in to Docker",
)
def test_raises_when_sbx_unavailable(_available: MagicMock, tmp_path: Path) -> None:
    with pytest.raises(SBXError, match=r"sbx is not available \(not signed in.*sbx login"):
        DockerSBX(shared_dir=tmp_path)


def test_sbx_error_is_public_runtime_error() -> None:
    assert issubclass(SBXError, RuntimeError)


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_runs_command_in_sandbox(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        exec=SBXRunResult(returncode=0, stdout="hello\n", stderr=""),
    )

    sandbox = DockerSBX(shared_dir=tmp_path, image="test-image:local")
    try:
        result = sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output

        exec_args = run_sbx.call_args_list[1][0][0]
        assert exec_args[:4] == [
            "exec",
            "--workdir",
            str(tmp_path.resolve()),
            sandbox.sandbox_name,
        ]
        assert exec_args[-1] == "echo hello"
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_formats_stderr_and_nonzero_exit(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        exec=SBXRunResult(returncode=2, stdout="", stderr="something broke\n"),
    )

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("false")
        assert result.exit_code == 2
        assert "[stderr] something broke" in result.output
        assert "Exit code: 2" in result.output
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_reports_no_output(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        exec=SBXRunResult(returncode=0, stdout="", stderr=""),
    )

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("true")
        assert result.output == "<no output>"
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_truncates_large_output(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        exec=SBXRunResult(returncode=0, stdout="x" * 200, stderr=""),
    )

    sandbox = DockerSBX(shared_dir=tmp_path, max_output_bytes=50)
    try:
        result = sandbox.execute("printf x")
        assert result.truncated is True
        assert len(result.output) <= 50 + len("\n\n... Output truncated at 50 bytes.")
        assert "Output truncated at 50 bytes" in result.output
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_rejects_empty_command(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("")
        assert result.exit_code == 1
        assert "non-empty string" in result.output
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_after_close_returns_error(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    sandbox.close()
    result = sandbox.execute("echo hello")
    assert result.exit_code == 1
    assert "closed" in result.output.lower()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_timeout_with_custom_message(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        error=SBXError("sbx command timed out after 1 seconds"),
    )

    sandbox = DockerSBX(shared_dir=tmp_path, timeout=1)
    try:
        result = sandbox.execute("sleep 10", timeout=1)
        assert result.exit_code == 124
        assert "custom timeout" in result.output.lower()
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_timeout_with_default_message(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        error=SBXError("sbx command timed out after 120 seconds"),
    )

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("sleep 10")
        assert result.exit_code == 124
        assert "timeout parameter" in result.output.lower()
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_when_sbx_binary_missing(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        error=SBXError("sbx executable not found on PATH"),
    )

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("echo hello")
        assert result.exit_code == 1
        assert "not found on PATH" in result.output
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_rejects_non_positive_timeout(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        with pytest.raises(ValueError, match="timeout must be positive"):
            sandbox.execute("echo hello", timeout=0)
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_write_and_read_via_virtual_paths(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path, image="test-image:local")
    try:
        write_result = sandbox.write("/notes.txt", "alpha\n")
        assert write_result.error is None
        assert (tmp_path / "notes.txt").read_text() == "alpha\n"

        read_result = sandbox.read("/notes.txt")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "alpha" in read_result.file_data["content"]
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_close_removes_sandbox(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    sandbox_name = sandbox.sandbox_name
    sandbox.close()

    assert len(run_sbx.call_args_list) == 2
    assert run_sbx.call_args_list[1][0][0] == ["rm", "--force", sandbox_name]


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_close_is_idempotent(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    sandbox.close()
    sandbox.close()

    assert len(run_sbx.call_args_list) == 2


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_close_skips_remove_when_auto_remove_disabled(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path, auto_remove=False)
    sandbox.close()

    assert len(run_sbx.call_args_list) == 2
    assert run_sbx.call_args_list[1][0][0] == ["stop", sandbox.sandbox_name]


@patch("deepagents_docker_sbx.backend.tempfile.mkdtemp")
@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_close_removes_owned_shared_dir(
    _available: MagicMock,
    run_sbx: MagicMock,
    mkdtemp: MagicMock,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "owned-shared"
    shared.mkdir()
    mkdtemp.return_value = str(shared)
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX()
    sandbox.close()

    assert not shared.exists()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_close_preserves_user_shared_dir(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    sandbox = DockerSBX(shared_dir=tmp_path)
    sandbox.close()

    assert tmp_path.exists()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_context_manager_closes_sandbox(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    with DockerSBX(shared_dir=tmp_path) as sandbox:
        assert len(sandbox.id) == 12

    assert len(run_sbx.call_args_list) == 2


def _exec_calls(run_sbx: MagicMock) -> list[list[str]]:
    return [call[0][0] for call in run_sbx.call_args_list if call[0][0][0] == "exec"]


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_failed_start_removes_sandbox(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = SBXRunResult(returncode=1, stdout="", stderr="boom")

    with pytest.raises(SBXError):
        DockerSBX(shared_dir=tmp_path)

    assert any(call[0][0][:2] == ["rm", "--force"] for call in run_sbx.call_args_list)


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_timed_out_create_removes_sandbox(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    def _run_sbx(args: list[str], **kwargs: object) -> SBXRunResult:
        if args[0] == "create":
            raise SBXError("sbx command timed out after 600 seconds")
        return _sbx_run_ok()

    run_sbx.side_effect = _run_sbx

    with pytest.raises(SBXError, match="timed out"):
        DockerSBX(shared_dir=tmp_path)

    assert run_sbx.call_args_list[-1][0][0][:2] == ["rm", "--force"]


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_caps_output_at_the_source(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect()

    sandbox = DockerSBX(shared_dir=tmp_path, max_output_bytes=4096)
    try:
        sandbox.execute("echo hi")
        exec_kwargs = run_sbx.call_args_list[1][1]
        assert exec_kwargs["max_output_bytes"] == 4096
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_reaps_process_group_after_timeout(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        error=SBXError("sbx command timed out after 1 seconds"),
    )

    sandbox = DockerSBX(shared_dir=tmp_path)
    try:
        result = sandbox.execute("sleep 100", timeout=1)
        assert result.exit_code == 124

        command_exec, cleanup_exec = _exec_calls(run_sbx)
        pid_file = command_exec[-2]
        assert pid_file.startswith("/tmp/.deepagents-exec-")
        assert "kill -9" in cleanup_exec[4]
        assert cleanup_exec[-1] == pid_file
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_execute_reaps_process_group_when_output_capped(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.side_effect = _make_run_sbx_side_effect(
        exec=SBXRunResult(returncode=-9, stdout="x" * 60, stderr="", truncated=True),
    )

    sandbox = DockerSBX(shared_dir=tmp_path, max_output_bytes=50)
    try:
        result = sandbox.execute("yes")
        assert result.truncated is True
        assert "terminated after exceeding the output limit" in result.output

        command_exec, cleanup_exec = _exec_calls(run_sbx)
        assert "kill -9" in cleanup_exec[4]
        assert cleanup_exec[-1] == command_exec[-2]
    finally:
        sandbox.close()


@patch("deepagents_docker_sbx.backend.run_sbx")
@patch("deepagents_docker_sbx.backend.sbx_unavailable_reason", return_value=None)
def test_abandoned_sandbox_is_not_kept_alive_by_atexit(
    _available: MagicMock,
    run_sbx: MagicMock,
    tmp_path: Path,
) -> None:
    run_sbx.return_value = _sbx_run_ok()

    ref = weakref.ref(DockerSBX(shared_dir=tmp_path))
    gc.collect()

    assert ref() is None
    assert any(call[0][0][:2] == ["rm", "--force"] for call in run_sbx.call_args_list)
