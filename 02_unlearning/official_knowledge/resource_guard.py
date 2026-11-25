"""Limits and monitoring for this user's experiment scopes only."""
import json
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time


POLICY = {"training_memory_max_gib": 144, "training_memory_high_gib": 144,
          "evaluation_memory_max_gib": 32, "evaluation_memory_high_gib": 32,
          "host_start_reserve_gib": 32, "host_stop_available_gib": 32,
          "host_start_pressure_max_percent": 10,
          "own_pressure_max_percent": 50, "own_pressure_timeout_seconds": 60,
          "no_progress_timeout_seconds": 1800,
          "cuda_allocator_fraction": .8, "swap_max_bytes": 0, "poll_seconds": 2}


def available_memory():
    return int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))) / 1024**2


def memory_pressure(path=Path("/proc/pressure/memory")):
    line = next(s for s in path.read_text().splitlines() if s.startswith("full "))
    return float(dict(value.split("=", 1) for value in line.split()[1:])["avg10"])


def idle_gpus():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = {line.split(",")[0].strip() for line in applications.splitlines() if len(line.split(",")) == 2 and line.split(",")[1].strip().isdigit()}
    snapshot = []
    for line in raw.splitlines():
        index, uuid, used, free, utilization = (x.strip() for x in line.split(","))
        snapshot.append({"index": int(index), "uuid": uuid, "used_mib": int(used), "free_mib": int(free), "utilization": int(utilization), "has_compute_process": uuid in occupied})
    idle = [str(g["index"]) for g in snapshot if not g["has_compute_process"] and g["used_mib"] < 512 and g["free_mib"] >= 23000 and g["utilization"] < 5]
    return idle, snapshot


def reserve_gpus(candidates, count, directory):
    """Coordinate this account's workers while a selected GPU is loading."""
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise ValueError("GPU reservation directory must belong to this account")
    selected, handles = [], []
    for gpu in candidates:
        handle = (directory / ("gpu-" + gpu + ".lock")).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        selected.append(gpu)
        handles.append(handle)
        if len(selected) == count:
            return selected, handles
    for handle in handles:
        handle.close()
    return [], []


def scope_command(command, unit, policy, training):
    if not re.fullmatch(r"ulp-small-[a-z0-9-]+[.]scope", unit):
        raise ValueError("Unexpected experiment scope name")
    kind = "training" if training else "evaluation"
    return ["systemd-run", "--user", "--scope", "--collect", "--quiet", "--unit=" + unit,
            "--property=MemoryMax=" + str(policy[kind + "_memory_max_gib"]) + "G",
            "--property=MemoryHigh=" + str(policy[kind + "_memory_high_gib"]) + "G",
            "--property=MemorySwapMax=0", "--property=OOMPolicy=kill", "--"] + command


def scope_path():
    text = Path("/proc/self/cgroup").read_text().strip()
    return Path("/sys/fs/cgroup") / text.split(":", 2)[2].lstrip("/")


def verify_current_scope(unit, expected_max):
    path = scope_path()
    if path.name != unit:
        raise RuntimeError("The model must run inside its own limited experiment scope")
    actual = {k: int((path / k).read_text()) for k in ("memory.max", "memory.high", "memory.swap.max", "memory.oom.group")}
    if actual["memory.max"] > expected_max or actual["memory.swap.max"] != 0 or actual["memory.oom.group"] != 1:
        raise RuntimeError("The experiment's memory limits were not applied")
    return actual


def monitor(process, unit, gpus, policy, receipt, progress_log=None):
    """Yield the GPU or RAM by stopping only our exact, newly created unit."""
    if not unit.startswith(f"ulp-small-{os.getpid()}-"):
        raise ValueError("Refusing to stop an experiment scope not created by this coordinator")
    started, pressure_since = time.time(), None
    last_io_epoch, last_io_progress, previous_io = started, time.monotonic(), 0
    reason, observed = None, {"scope": unit, "minimum_host_available_gib": None, "observed_scope_peak_bytes": 0}
    while process.poll() is None:
        memory = available_memory()
        observed["minimum_host_available_gib"] = memory if observed["minimum_host_available_gib"] is None else min(memory, observed["minimum_host_available_gib"])
        if memory < policy["host_stop_available_gib"]:
            reason = "host_memory_pressure"
        group = subprocess.run(["systemctl", "--user", "show", unit, "--property=ControlGroup", "--value"], text=True, capture_output=True)
        control = group.stdout.strip() if group.returncode == 0 else ""
        if control and control.startswith(f"/user.slice/user-{os.getuid()}.slice/"):
            path = Path("/sys/fs/cgroup") / control.lstrip("/")
            try:
                observed["observed_scope_peak_bytes"] = max(observed["observed_scope_peak_bytes"], int((path / "memory.peak").read_text()))
                observed["own_memory_pressure_percent"] = memory_pressure(path / "memory.pressure")
                observed["memory_events"] = (path / "memory.events").read_text()
                written = 0
                try:
                    for pid in (path / "cgroup.procs").read_text().split():
                        try:
                            written += int(next(line.split()[1] for line in Path("/proc", pid, "io").read_text().splitlines() if line.startswith("wchar:")))
                        except (FileNotFoundError, PermissionError, ProcessLookupError):
                            pass
                except FileNotFoundError:
                    pass
                if written > previous_io + 1024**2:
                    previous_io = written
                    last_io_epoch, last_io_progress = time.time(), time.monotonic()
                observed["scope_written_bytes"] = written
                if observed["own_memory_pressure_percent"] > policy["own_pressure_max_percent"]:
                    if time.monotonic() - last_io_progress < policy["own_pressure_timeout_seconds"]:
                        pressure_since = None
                    else:
                        pressure_since = time.monotonic() if pressure_since is None else pressure_since
                        if time.monotonic() - pressure_since >= policy["own_pressure_timeout_seconds"]:
                            reason = "own_memory_pressure_stall"
                else:
                    pressure_since = None
            except FileNotFoundError:
                pass
            raw = (subprocess.check_output(["nvidia-smi", "--id=" + ",".join(gpus), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
                   if gpus else "")
            for value in raw.splitlines():
                if not value.strip().isdigit():
                    continue
                try:
                    membership = Path("/proc", value.strip(), "cgroup").read_text()
                except FileNotFoundError:
                    continue
                except PermissionError:
                    membership = ""
                if control not in membership:
                    reason = "another_process_appeared_on_selected_gpu"
                    break
        if progress_log is not None:
            try:
                last_progress = max(started, last_io_epoch, Path(progress_log).stat().st_mtime)
            except FileNotFoundError:
                last_progress = max(started, last_io_epoch)
            observed["seconds_since_progress"] = round(time.time() - last_progress, 1)
            if time.time() - last_progress > policy["no_progress_timeout_seconds"]:
                reason = "no_training_progress_timeout"
        if reason:
            # The name is unique to our coordinator; systemd controls this
            # user's transient scope, not any other user or their session.
            if process.poll() is not None:
                reason = None
                break
            stopped = subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, text=True)
            if stopped.returncode and process.poll() is None:
                raise RuntimeError("Could not stop this experiment's own scope")
            observed["stopped_own_scope_reason"] = reason
            break
        Path(receipt).write_text(json.dumps(observed, indent=2) + "\n")
        time.sleep(policy["poll_seconds"])
    process.wait()
    Path(receipt).write_text(json.dumps(observed, indent=2) + "\n")
    return reason
