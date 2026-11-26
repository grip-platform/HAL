"""HAL Daemon — always-on background process.

Runs the HTTP server, mesh heartbeat, model warmup, and scheduler
as a persistent background daemon. No concept of "sessions" — HAL
is continuously available.

Usage:
    hal --daemon start          # Start background daemon
    hal --daemon stop           # Graceful shutdown
    hal --daemon status         # Show PID, uptime, connections

    # Or directly:
    from lib.hal.daemon import HALDaemon
    daemon = HALDaemon(port=3850)
    daemon.start()              # Foreground (blocking)
    daemon.daemonize()          # Background (forks)
"""
import os
import signal
import sys
import time

from lib.hal.home import HAL_HOME as _HAL_HOME

_PID_FILE = os.path.join(_HAL_HOME, "state", "hal", "daemon.pid")
_LOG_FILE = os.path.join(_HAL_HOME, "state", "hal", "daemon.log")


class HALDaemon:
    """Always-on HAL daemon process."""

    def __init__(self, port: int = 3850):
        self._port = port
        self._running = False

    def start(self) -> None:
        """Start daemon in foreground (blocking)."""
        self._write_pid()
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        from lib.hal.discovery import discover_local_providers
        from lib.hal.mesh import MeshRegistry
        from lib.hal.registry import get_registry
        from lib.hal.scheduler import Scheduler
        from lib.hal.server import HALServer
        from lib.hal.memory_manager import MemoryManager

        registry = get_registry()
        discover_local_providers(registry)

        # Start subsystems
        self._memory_manager = MemoryManager(registry=registry)
        self._memory_manager.start_keepalive()

        self._mesh = MeshRegistry()
        self._mesh.announce(status="daemon")

        self._scheduler = Scheduler()
        self._scheduler.register_handler("warmup_models", lambda: self._memory_manager.warm_all())
        self._scheduler.register_handler("mesh_heartbeat", lambda: self._mesh.announce(status="daemon"))
        self._scheduler.start()

        _log("HAL daemon started", port=self._port, pid=os.getpid())

        # Start HTTP server (blocking)
        server = HALServer(port=self._port, registry=registry)
        server.start()

    def daemonize(self) -> int:
        """Fork into background. Returns child PID."""
        pid = os.fork()
        if pid > 0:
            return pid  # Parent returns child PID

        # Child: detach from terminal
        os.setsid()
        sys.stdin = open(os.devnull, "r")
        sys.stdout = open(_LOG_FILE, "a")
        sys.stderr = sys.stdout

        self.start()
        return 0

    def stop(self) -> bool:
        """Stop a running daemon by PID file. Returns True if stopped."""
        pid = self._read_pid()
        if not pid:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(10):
                try:
                    os.kill(pid, 0)  # Check if alive
                    time.sleep(0.5)
                except OSError:
                    break
            self._cleanup()
            return True
        except OSError:
            self._cleanup()
            return False

    @staticmethod
    def status() -> dict:
        """Get daemon status."""
        pid = HALDaemon._read_pid_static()
        if not pid:
            return {"running": False}

        try:
            os.kill(pid, 0)  # Check if alive
            return {
                "running": True,
                "pid": pid,
                "pid_file": _PID_FILE,
            }
        except OSError:
            return {"running": False, "stale_pid": pid}

    def _handle_signal(self, signum, frame):
        """Graceful shutdown on SIGTERM/SIGINT."""
        _log("Shutdown signal received", signal=signum)
        self._running = False
        if hasattr(self, "_scheduler"):
            self._scheduler.stop()
        if hasattr(self, "_mesh"):
            self._mesh.cleanup()
        if hasattr(self, "_memory_manager"):
            self._memory_manager.stop()
        self._cleanup()
        sys.exit(0)

    def _write_pid(self) -> None:
        """Write PID file."""
        os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
        with open(_PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _cleanup(self) -> None:
        """Remove PID file."""
        try:
            os.remove(_PID_FILE)
        except OSError:
            pass

    @staticmethod
    def _read_pid_static() -> int:
        """Read PID from file. Returns 0 if not found."""
        try:
            with open(_PID_FILE, "r") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def _read_pid(self) -> int:
        return self._read_pid_static()


def _log(message: str, **kwargs) -> None:
    """Append to daemon log."""
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    line = f"[{timestamp}] {message} {extra}\n"
    try:
        with open(_LOG_FILE, "a") as f:
            f.write(line)
    except OSError:
        pass
