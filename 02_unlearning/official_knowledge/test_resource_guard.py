"""Exercise stop decisions without a GPU, model, or real process signals."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from official_knowledge import resource_guard as guard


class ResourceGuardTests(unittest.TestCase):
    def simulate(self, *, pressure=0, elapsed=0, foreign_pid=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = f"/user.slice/user-{os.getuid()}.slice/app.slice/test.scope"
            group = root / "cgroups" / control.lstrip("/")
            group.mkdir(parents=True)
            for name, text in {"memory.peak": "123", "memory.events": "oom 0\n",
                               "memory.pressure": f"full avg10={pressure} avg60=0 avg300=0 total=0\n"}.items():
                (group / name).write_text(text)
            def paths(*args):
                if args[0] == "/sys/fs/cgroup":
                    return root / "cgroups"
                if args[0] == "/proc":
                    return root / "unknown-process"
                return Path(*args)
            process = Mock()
            process.poll.return_value = None
            unit = f"ulp-small-{os.getpid()}-test.scope"
            policy = dict(guard.POLICY, own_pressure_timeout_seconds=0)
            def command(argv, **kwargs):
                return SimpleNamespace(returncode=0, stdout=control if "show" in argv else "")
            with patch.object(guard, "Path", side_effect=paths), \
                 patch.object(guard, "available_memory", return_value=200), \
                 patch.object(guard.subprocess, "run", side_effect=command) as run, \
                 patch.object(guard.subprocess, "check_output", return_value="424242\n" if foreign_pid else ""), \
                 patch.object(guard.time, "time", side_effect=[100, 100 + elapsed, 100 + elapsed]), \
                 patch.object(guard.time, "sleep"):
                # A disappearing foreign PID is skipped; inaccessible live
                # membership is treated as foreign, so use PermissionError.
                if foreign_pid:
                    original = Path.read_text
                    def read_text(path, *args, **kwargs):
                        if "unknown-process" in str(path):
                            raise PermissionError("another user")
                        return original(path, *args, **kwargs)
                    with patch.object(Path, "read_text", read_text):
                        reason = guard.monitor(process, unit, ["0"], policy, root / "receipt.json", root / "log")
                else:
                    reason = guard.monitor(process, unit, ["0"], policy, root / "receipt.json", root / "log")
            stops = [call.args[0] for call in run.call_args_list if "stop" in call.args[0]]
            self.assertEqual(stops, [["systemctl", "--user", "stop", unit]])
            process.wait.assert_called_once()
            return reason

    def test_sustained_memory_pressure_stops_only_own_scope(self):
        self.assertEqual(self.simulate(pressure=90), "own_memory_pressure_stall")

    def test_no_progress_stops_only_own_scope(self):
        self.assertEqual(self.simulate(elapsed=1900), "no_training_progress_timeout")

    def test_other_gpu_process_stops_only_own_scope(self):
        self.assertEqual(self.simulate(foreign_pid=True), "another_process_appeared_on_selected_gpu")


if __name__ == "__main__":
    unittest.main()
