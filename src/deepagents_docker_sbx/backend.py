from __future__ import annotations

import atexit
import tempfile
import uuid
import weakref
from collections.abc import Callable
from functools import partial
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from ._sbx import (
    format_sbx_error,
    run_sbx,
    sbx_unavailable_reason,
)
from .errors import SBXError

DEFAULT_EXECUTE_TIMEOUT = 120
# The `shell` agent starts a sandbox with no agent binary; its default template
# (`docker/sandbox-templates:shell`) ships Python, uv, git and curl.
SANDBOX_AGENT = "shell"
SBX_CHECK_TIMEOUT = 30
SBX_CREATE_TIMEOUT = 600
CLEANUP_TIMEOUT = 60

# Each `sbx exec` is its own session/process-group leader, so
# the wrapper records its PID (== the process group id) and a later `kill -9 -<pgid>` reaps the
# whole tree, including children reparented to PID 1. $1 is the pid file, $2 the user command.
_EXEC_WRAPPER = """
echo $$ > "$1" 2>/dev/null
sh -c "$2"
status=$?
rm -f "$1" 2>/dev/null
exit $status
"""

# Kill the process group recorded by `_EXEC_WRAPPER`. $1 is the pid file.
_KILL_WRAPPER = """
pgid=""
[ -f "$1" ] && read pgid < "$1"
rm -f "$1" 2>/dev/null
[ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null
exit 0
"""


def _atexit_close(ref: weakref.ReferenceType[DockerSBX]) -> None:
    """Close a sandbox at interpreter exit without keeping it alive until then."""
    sandbox = ref()
    if sandbox is not None:
        sandbox.close()


class DockerSBX(FilesystemBackend, SandboxBackendProtocol):
    """Docker Sandboxes (`sbx`) backed sandbox backend for DeepAgents."""

    def __init__(
        self,
        *,
        template: str | None = None,
        allow_outbound_traffic: bool = True,
        shared_dir: str | Path | None = None,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        memory: str | None = None,
        cpus: int | None = None,
        auto_remove: bool = True,
        extra_create_args: list[str] | None = None,
    ) -> None:
        """Create an sbx sandbox and shared directory.

        Args:
            template: Sandbox template image (``sbx create --template``). Defaults to the
                template of the ``shell`` agent. Custom images must meet the sbx base-image
                requirements (an ``agent`` user with UID 1000, ``/bin/sh`` and ``/bin/bash``,
                curl, git and CA certificates); plain images such as ``python:3.12-bookworm``
                do not.
            allow_outbound_traffic: ``False`` adds a per-sandbox deny-all network rule. ``True``
                leaves egress to the global ``sbx policy``, which this option cannot widen.
            shared_dir: Host directory mounted into the sandbox at the same absolute path and
                used as the working directory. A temporary directory is created when omitted.
            timeout: Default command timeout in seconds.
            max_output_bytes: Maximum stdout/stderr captured per command. Output is streamed
                and the command is killed once a stream reaches this cap, so a command writing
                unbounded output cannot exhaust host memory.
            memory: Sandbox memory limit (for example ``"1g"``; sbx requires at least
                ``512m``). Defaults to the sbx default.
            cpus: Number of whole CPUs. Defaults to the sbx default.
            auto_remove: Remove the sandbox on ``close()``; otherwise it is only stopped.
            extra_create_args: Additional ``sbx create`` flags (for example ``["-e", "FOO=bar"]``).
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)
        if cpus is not None and (not isinstance(cpus, int) or cpus <= 0):
            msg = f"cpus must be a positive whole number, got {cpus}"
            raise ValueError(msg)
        if max_output_bytes <= 0:
            msg = f"max_output_bytes must be positive, got {max_output_bytes}"
            raise ValueError(msg)

        self._owns_shared_dir = shared_dir is None
        self._shared_dir = Path(
            tempfile.mkdtemp(prefix="deepagents-sbx-shared-") if shared_dir is None else shared_dir,
        ).resolve()
        self._shared_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(root_dir=self._shared_dir, virtual_mode=True)

        self._template = template
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._memory = memory
        self._cpus = cpus
        self._allow_outbound_traffic = allow_outbound_traffic
        self._auto_remove = auto_remove
        self._extra_create_args = list(extra_create_args or [])

        self._sandbox_id: str = f"{uuid.uuid4().hex[:12]}"
        self._sandbox_name = f"deepagents-{self._sandbox_id}"
        self._closed = False
        self._atexit_hook: Callable[[], None] | None = None

        self._start_sandbox()
        self._atexit_hook = partial(_atexit_close, weakref.ref(self))
        atexit.register(self._atexit_hook)

    @property
    def shared_dir(self) -> Path:
        """Host path of the folder shared with the sandbox."""
        return self._shared_dir

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox instance."""
        return self._sandbox_id

    @property
    def sandbox_name(self) -> str:
        """Name of the sandbox as shown by `sbx ls`."""
        return self._sandbox_name

    def _create_args(self) -> list[str]:
        args = ["create", "--name", self._sandbox_name, "--no-progress"]
        if self._template is not None:
            args += ["--template", self._template]
        if self._memory is not None:
            args += ["--memory", self._memory]
        if self._cpus is not None:
            args += ["--cpus", str(self._cpus)]
        if not self._allow_outbound_traffic:
            args += ["--deny-network", "**"]
        return [*args, *self._extra_create_args, SANDBOX_AGENT, str(self._shared_dir)]

    def _start_sandbox(self) -> None:
        reason = sbx_unavailable_reason(timeout=SBX_CHECK_TIMEOUT)
        if reason is not None:
            msg = (
                f"sbx is not available ({reason}). Install Docker Sandboxes, run `sbx login`, "
                "and initialize a network policy with `sbx policy init`."
            )
            raise SBXError(msg)

        try:
            result = run_sbx(self._create_args(), timeout=SBX_CREATE_TIMEOUT)
        except SBXError:
            # A timed-out `sbx create` may still have created the sandbox.
            self._force_remove_sandbox()
            raise
        if result.returncode != 0:
            self._force_remove_sandbox()
            msg = format_sbx_error(result)
            raise SBXError(f"failed to create sandbox: {msg}")

    def _force_remove_sandbox(self) -> None:
        """Best-effort `sbx rm --force`, used on failed startup and on close."""
        try:
            run_sbx(["rm", "--force", self._sandbox_name], timeout=CLEANUP_TIMEOUT)
        except SBXError:
            pass

    def _kill_exec_group(self, pid_file: str) -> None:
        """Kill the in-sandbox process group left behind by a timed-out/killed exec."""
        try:
            run_sbx(
                ["exec", self._sandbox_name, "sh", "-c", _KILL_WRAPPER, "sh", pid_file],
                timeout=CLEANUP_TIMEOUT,
                max_output_bytes=4096,
            )
        except SBXError:
            pass

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox."""
        if self._closed:
            return ExecuteResponse(
                output="Error: Sandbox has been closed.",
                exit_code=1,
                truncated=False,
            )

        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        pid_file = f"/tmp/.deepagents-exec-{uuid.uuid4().hex}"  # noqa: S108
        sbx_args = [
            "exec",
            "--workdir",
            str(self._shared_dir),
            self._sandbox_name,
            "sh",
            "-c",
            _EXEC_WRAPPER,
            "sh",
            pid_file,
            command,
        ]

        try:
            completed = run_sbx(
                sbx_args,
                timeout=effective_timeout,
                max_output_bytes=self._max_output_bytes,
            )
        except SBXError as exc:
            detail = str(exc)
            if "timed out" in detail:
                # Killing the `sbx exec` client leaves the command running in the
                # sandbox; reap its process group so it cannot burn the sandbox's
                # CPU budget forever.
                self._kill_exec_group(pid_file)
                if timeout is not None:
                    msg = (
                        f"Error: Command timed out after {effective_timeout} seconds "
                        "(custom timeout). The command may be stuck or require more time."
                    )
                else:
                    msg = (
                        f"Error: Command timed out after {effective_timeout} seconds. "
                        "For long-running commands, re-run using the timeout parameter."
                    )
                return ExecuteResponse(output=msg, exit_code=124, truncated=False)
            if "not found on PATH" in detail:
                return ExecuteResponse(
                    output=(
                        "Error executing command (FileNotFoundError): "
                        "sbx executable not found on PATH"
                    ),
                    exit_code=1,
                    truncated=False,
                )
            return ExecuteResponse(
                output=f"Error executing command (SBXError): {exc}",
                exit_code=1,
                truncated=False,
            )

        if completed.truncated:
            self._kill_exec_group(pid_file)

        output_parts: list[str] = []
        if completed.stdout:
            output_parts.append(completed.stdout)
        if completed.stderr:
            stderr_lines = completed.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(output_parts) if output_parts else "<no output>"
        truncated = completed.truncated
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            truncated = True
        if truncated:
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
        if completed.truncated:
            output += " The command was terminated after exceeding the output limit."

        if completed.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {completed.returncode}"

        return ExecuteResponse(
            output=output,
            exit_code=completed.returncode,
            truncated=truncated,
        )

    def close(self) -> None:
        """Remove the sandbox (or only stop it when ``auto_remove`` is disabled)."""
        if self._closed:
            return
        self._closed = True

        if self._atexit_hook is not None:
            atexit.unregister(self._atexit_hook)

        # Best-effort shutdown; the sandbox may already be gone. `sbx rm` stops it first.
        if self._auto_remove:
            self._force_remove_sandbox()
        else:
            try:
                run_sbx(["stop", self._sandbox_name], timeout=CLEANUP_TIMEOUT)
            except SBXError:
                pass

        if self._owns_shared_dir:
            import shutil

            shutil.rmtree(self._shared_dir, ignore_errors=True)

    def __enter__(self) -> DockerSBX:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass
