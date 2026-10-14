from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("torchy.infra")


class ColabProcess:
    """Popen-compatible handle for a process running inside a remote Colab tmux session."""

    def __init__(
        self,
        session: ColabSession,
        cmd: list[str] | str,
        proc_id: str,
        log_path: Path | str,
    ):
        if not log_path:
            raise ValueError(f"log_path is required for process '{proc_id}'; each ColabProcess must have its own log file.")
        self.session = session
        self.cmd = cmd if isinstance(cmd, list) else shlex.split(cmd)
        self.proc_id = proc_id
        self.log_path = Path(log_path)
        self.returncode: int | None = None
        self.pid: int = hash(proc_id) % 1000000  # synthetic pid for Popen compatibility
        self._done_event = threading.Event()
        self._received_bytes = 0
        self._terminated = False

        self._stream_thread = threading.Thread(
            target=self._stream_supervisor,
            name=f"stream-{self.proc_id}",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_supervisor(self) -> None:
        """Stream logs from the remote out.log with auto-resume across SSH drops."""
        remote_dir = f"/tmp/mycolab_proc/{self.proc_id}"
        out_log = f"{remote_dir}/out.log"
        done_file = f"{remote_dir}/done"

        out_file = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = open(self.log_path, "ab")

        try:
            while self.returncode is None and not self._terminated:
                from_byte = self._received_bytes + 1
                script = f"""
                tail -c +{from_byte} -f {shlex.quote(out_log)} &
                TPID=$!
                while [ ! -f {shlex.quote(done_file)} ]; do
                    if ! tmux has-session -t {shlex.quote(self.proc_id)} 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                sleep 1
                kill $TPID 2>/dev/null || true
                wait $TPID 2>/dev/null || true
                if [ -f {shlex.quote(done_file)} ]; then
                    EXIT_CODE=$(cat {shlex.quote(done_file)})
                    if [ "$EXIT_CODE" -eq 255 ]; then EXIT_CODE=254; fi
                    exit $EXIT_CODE
                else
                    exit 1
                fi
                """
                ssh_cmd = ["ssh", self.session.session, script]
                try:
                    p = subprocess.Popen(
                        ssh_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except Exception as e:
                    logger.warning(f"[{self.proc_id}] SSH stream launch failed: {e}")
                    time.sleep(2)
                    self.session.ensure_master()
                    continue

                assert p.stdout is not None
                while True:
                    chunk = p.stdout.read(4096)
                    if not chunk:
                        break
                    if out_file:
                        out_file.write(chunk)
                        out_file.flush()
                    self._received_bytes += len(chunk)

                code = p.wait()
                if self._terminated:
                    break

                if code == 255:  # Transport failure
                    logger.info(f"[{self.proc_id}] SSH dropped (exit 255); re-establishing master...")
                    self.session.ensure_master()
                    time.sleep(2)
                    continue
                else:
                    self.returncode = code
                    self._done_event.set()
                    break
        finally:
            if out_file:
                out_file.close()

    def poll(self) -> int | None:
        """Check if process has completed. Returns exit code or None if running."""
        if self.returncode is not None:
            return self.returncode

        # Fast remote probe if done file exists
        remote_done = f"/tmp/mycolab_proc/{self.proc_id}/done"
        res = subprocess.run(
            ["ssh", self.session.session, f"[ -f {shlex.quote(remote_done)} ] && cat {shlex.quote(remote_done)} || echo RUNNING"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        val = res.stdout.strip()
        if val and val != "RUNNING" and val.lstrip("-").isdigit():
            self.returncode = int(val)
            self._done_event.set()
            return self.returncode
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process exits or timeout expires."""
        if self.returncode is not None:
            return self.returncode
        if not self._done_event.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.returncode if self.returncode is not None else 0

    def terminate(self) -> None:
        """Terminate the process by killing its remote tmux session."""
        self._terminated = True
        subprocess.run(
            ["ssh", self.session.session, f"tmux kill-session -t {shlex.quote(self.proc_id)} 2>/dev/null || true"],
            capture_output=True,
            timeout=5,
        )
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        self._done_event.set()

    def kill(self) -> None:
        """Force kill the process."""
        self.terminate()

    def stop(self) -> None:
        """Alias for terminate."""
        self.terminate()

    def stop_vm(self) -> None:
        """Stop the underlying Colab VM session."""
        self.session.stop_vm()


class VMStopper:
    """Supports both boolean check (if session.stop_vm:) and direct call (session.stop_vm())."""

    def __init__(self, session: ColabSession, enabled: bool = False):
        self._session = session
        self.enabled = enabled

    def __bool__(self) -> bool:
        return bool(self.enabled)

    def __call__(self) -> None:
        self._session.terminate_vm()


class ColabSession:
    """Manages a Colab instance lifecycle and SSH ControlMaster multiplex connection."""

    def __init__(
        self,
        session: str,
        gpu: str | None = None,
        tpu: str | None = None,
        high_mem: bool = False,
        volumes: list[str] | dict[str, str] | None = None,
        local_forwards: list[str] | None = None,
        remote_forwards: list[str] | None = None,
        env: dict[str, str] | None = None,
        stop_vm: bool = False,
        auto_new: bool = True,
        reuse: bool = False,
    ):
        self.session = session
        self.gpu = gpu
        self.tpu = tpu
        self.high_mem = high_mem
        self.local_forwards = local_forwards or []
        self.remote_forwards = remote_forwards or []
        self._stop_vm_handler = VMStopper(self, enabled=stop_vm)

        # Parse volumes into list of "bucket:mount[:ro]"
        self.volumes: list[str] = []
        if isinstance(volumes, dict):
            for k, v in volumes.items():
                self.volumes.append(f"{k}:{v}")
        elif isinstance(volumes, list):
            self.volumes = volumes

        self.env = env or {}
        self.processes: list[ColabProcess] = []

        if auto_new:
            self.new(reuse=reuse)
        else:
            self.ensure_master()

    @property
    def stop_vm(self) -> VMStopper:
        return self._stop_vm_handler

    @stop_vm.setter
    def stop_vm(self, value: bool | VMStopper) -> None:
        if isinstance(value, VMStopper):
            self._stop_vm_handler = value
        else:
            self._stop_vm_handler.enabled = bool(value)

    def new(self, reuse: bool = False) -> None:
        """Create or configure the session via `mycolab new`."""
        if reuse:
            self.ensure_master()
            return

        cmd = ["mycolab", "new", "-s", self.session]
        if self.gpu:
            cmd.extend(["--gpu", self.gpu])
        elif self.tpu:
            cmd.extend(["--tpu", self.tpu])
        if self.high_mem:
            cmd.append("--high-mem")

        for v in self.volumes:
            cmd.extend(["-v", v])

        for lf in self.local_forwards:
            cmd.extend(["-L", lf])

        for rf in self.remote_forwards:
            cmd.extend(["-R", rf])

        for k, v in self.env.items():
            cmd.extend(["-e", f"{k}={v}"])

        logger.info(f"[{self.session}] Creating runtime: {' '.join(cmd)}")
        res = subprocess.run(cmd, check=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create session {self.session}: exit {res.returncode}")

        self.ensure_master()

    def ensure_master(self) -> None:
        """Ensure the SSH multiplex master is alive with port forwards and keepalives."""
        # 1. Check if master is running
        check = subprocess.run(
            ["ssh", "-O", "check", self.session],
            capture_output=True,
        )
        if check.returncode == 0:
            # Master running; apply any forwardings if needed
            for lf in self.local_forwards:
                subprocess.run(["ssh", "-O", "forward", "-L", lf, self.session], capture_output=True)
            for rf in self.remote_forwards:
                subprocess.run(["ssh", "-O", "forward", "-R", rf, self.session], capture_output=True)
            return

        # 2. Force cleanup stale socket
        subprocess.run(["ssh", "-O", "exit", self.session], capture_output=True)

        # 3. Start background master with aggressive keepalives
        args = [
            "ssh", "-fN",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=yes",
        ]
        for lf in self.local_forwards:
            args.extend(["-L", lf])
        for rf in self.remote_forwards:
            args.extend(["-R", rf])
        args.append(self.session)

        logger.info(f"[{self.session}] Starting SSH master connection: {' '.join(args)}")
        subprocess.run(args, check=True)

    def Popen(
        self,
        cmd: list[str] | str,
        log_path: Path | str,
        dir: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ColabProcess:
        """Launch a command inside an isolated tmux session on the Colab VM.

        Args:
            cmd: Command to execute (list of arguments or string).
            log_path: Path to the local log file. Required; each process must have its own log.
            dir: Mapping of {"source_dir": "target_colab_dir"}. Syncs source to target,
                 and runs the command with the target directory as the working directory.
            env: Extra environment variables for this process.
        """
        if not log_path:
            raise ValueError(f"[{self.session}] log_path is required; each ColabProcess must have its own log file.")

        self.ensure_master()

        remote_work_dir = None
        if dir:
            for src, dst in dir.items():
                src_path = Path(src).resolve()
                dst_path = dst.rstrip("/")
                if remote_work_dir is None:
                    remote_work_dir = dst_path
                logger.info(f"[{self.session}] Syncing {src_path} -> {dst_path}")

                gitignore = src_path / ".gitignore"
                filter_args = ["--filter=:- .gitignore"] if gitignore.exists() else []

                subprocess.run(
                    [
                        "rsync", "-az", *filter_args, "-e", "ssh",
                        "--rsync-path", f"mkdir -p {shlex.quote(dst_path)} && rsync",
                        f"{src_path}/",
                        f"{self.session}:{dst_path}/",
                    ],
                    check=True,
                )

        proc_id = f"proc_{uuid.uuid4().hex[:8]}"
        remote_dir = f"/tmp/mycolab_proc/{proc_id}"

        # Build command string
        cmd_str = " ".join(shlex.quote(c) for c in cmd) if isinstance(cmd, list) else cmd
        if remote_work_dir:
            cmd_str = f"cd {shlex.quote(remote_work_dir)} && {cmd_str}"

        full_env = env or {}
        if full_env:
            exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in full_env.items())
            cmd_str = f"export {exports} && {cmd_str}"

        out_log = f"{remote_dir}/out.log"
        done_file = f"{remote_dir}/done"
        wrapped_main = f"( {cmd_str} ) 2>&1 | tee {shlex.quote(out_log)}; echo ${{PIPESTATUS[0]}} > {shlex.quote(done_file)}"

        steps = [
            f"mkdir -p {shlex.quote(remote_dir)}",
            f"rm -f {shlex.quote(out_log)} {shlex.quote(done_file)}",
            f"touch {shlex.quote(out_log)}",
            f"tmux new-session -d -s {shlex.quote(proc_id)} -n main {shlex.quote(wrapped_main)}",
        ]

        launch_script = " && ".join(steps)
        logger.info(f"[{self.session}] Spawning tmux session '{proc_id}' for: {cmd_str}")
        subprocess.run(["ssh", self.session, launch_script], check=True)

        proc = ColabProcess(self, cmd, proc_id, log_path=log_path)
        self.processes.append(proc)
        return proc

    def stop(self) -> None:
        """Terminate all processes tracked under this session."""
        for p in self.processes:
            p.terminate()

    def terminate_vm(self) -> None:
        """Stop the remote Colab VM."""
        logger.info(f"[{self.session}] Stopping Colab VM...")
        subprocess.run(["colab", "stop", "-s", self.session], check=False)
        subprocess.run(["ssh", "-O", "exit", self.session], capture_output=True)


def monitor_process(process: Any, stop_event: threading.Event, error_queue: list[Exception], name: str) -> None:
    """Monitor a subprocess/ColabProcess and signal errors via shared queue."""
    process.wait()
    if process.returncode != 0:
        err_msg = f"{name.capitalize()} failed with exit code {process.returncode}"
        error_queue.append(RuntimeError(err_msg))
    stop_event.set()


class ProcessGroup:
    """Manages process lifecycles and monitors their completion, mirroring prime-rl ProcessGroup.

    Supports both standard subprocess.Popen and ColabProcess. Handles Ctrl+C (SIGINT/SIGTERM)
    by terminating all processes and stopping Colab VMs if stop_vm is enabled.
    """

    def __init__(self):
        self.processes: dict[str, Any] = {}
        self.sessions: dict[str, ColabSession] = {}
        self.monitor_threads: list[threading.Thread] = []
        self.error_queue: list[Exception] = []
        self.stop_events: dict[str, threading.Event] = {}

    def __enter__(self) -> ProcessGroup:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is KeyboardInterrupt:
            logger.warning("Received interrupt signal, terminating all processes and VMs...")
        elif isinstance(exc, Exception):
            logger.error(f"Error occurred: {exc}")
        self.cleanup()
        if exc_type is KeyboardInterrupt:
            sys.exit(1)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.warning(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(1)

    def add(self, name: str, process: Any, session: ColabSession | None = None) -> Any:
        """Add a process (Popen or ColabProcess) to be monitored by ProcessGroup."""
        self.processes[name] = process
        sess = session or getattr(process, "session", None)
        if sess is not None:
            sess_name = getattr(sess, "session", str(id(sess)))
            if sess_name not in self.sessions:
                self.sessions[sess_name] = sess

        stop_event = threading.Event()
        self.stop_events[name] = stop_event
        thread = threading.Thread(
            target=monitor_process,
            args=(process, stop_event, self.error_queue, name),
            name=f"monitor-{name}",
            daemon=True,
        )
        thread.start()
        self.monitor_threads.append(thread)
        return process

    def wait(self, *names: str) -> None:
        """Block until the named processes exit; exit immediately if any started process fails."""
        events = [self.stop_events[name] for name in names if name in self.stop_events]
        while True:
            if self.error_queue:
                logger.error(f"Error in process group: {self.error_queue[0]}")
                logger.error("Terminating all processes...")
                self.cleanup()
                sys.exit(1)
            if all(event.is_set() for event in events):
                return
            time.sleep(0.5)

    def cleanup(self) -> None:
        """Terminate all processes and stop Colab VMs if stop_vm is configured."""
        logger.info("Cleaning up ProcessGroup processes...")
        for name, proc in self.processes.items():
            if hasattr(proc, "poll") and proc.poll() is None:
                logger.info(f"Terminating {name}...")
                if hasattr(proc, "terminate"):
                    proc.terminate()
                elif hasattr(proc, "kill"):
                    proc.kill()

        for name, sess in self.sessions.items():
            if getattr(sess, "stop_vm", False):
                logger.info(f"stop_vm=True on session '{name}'; stopping VM...")
                if callable(getattr(sess, "terminate_vm", None)):
                    sess.terminate_vm()
                elif callable(getattr(sess, "stop_vm", None)):
                    sess.stop_vm()
