"""Mount-free execution sandbox for CodeContests submissions.

The factory container blocks mount(2) through AppArmor, so Bubblewrap cannot start there.
When invoked as root, this executor needs only user/pid/net/ipc/uts namespaces, chroot,
setuid and seccomp-BPF — zero mount(2) calls. Each submission runs as an unprivileged
outside uid in private namespaces, chrooted into a root-owned minimal jail, with
RLIMIT_NPROC=1 and a seccomp-BPF filter blocking process/thread creation. The launcher
enforces the wall clock, kills the private PID namespace at the deadline and reaps it.
`probe()` verifies all properties inside the real sandbox before any data is read; it
never runs an unbounded process test.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from time import monotonic, sleep

EXECUTOR = "work3-userns-chroot-v1"
SYSTEM_PYTHON = Path("/usr/bin/python3")
LAUNCHER = Path(__file__).resolve().with_name("pair_sandbox_launch.py")
WALL_SECONDS = 2.5
BACKSTOP_SECONDS = 3.0
TIMEOUT_EXIT, USERNS_DENIED_EXIT, SECCOMP_DENIED_EXIT = 124, 126, 127
TEXT_CAP = 192
READY = b"WORK3_SANDBOX_READY\n"
EXCLUDED_DIRS = {"site-packages", "dist-packages", "test", "tests", "idlelib", "tkinter", "turtledemo", "ensurepip"}
ROOT_ENTRIES = ("usr", "lib", "lib64", "lib32", "libx32")
FALLBACK_IFACES = ("lo", "tunl0", "sit0", "ip6tnl0", "gre0", "gretap0", "erspan0",
                   "ip6gre0", "ip6gretap0", "ip6erspan0")
LIBRARY = re.compile(r"(?:=>\s*)?(/\S+)\s+\(0x[0-9a-f]+\)")
LAYOUT = ("import json,sys,sysconfig; print(json.dumps({'stdlib': sysconfig.get_paths()['stdlib'], "
          "'dynload': sysconfig.get_config_var('DESTSHARED') or '', 'version': sys.version.split()[0]}))")


def sha256_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def normalized(text):
    return " ".join(text.split())


class SandboxStartError(RuntimeError):
    def __init__(self, returncode, detail):
        super().__init__(f"isolated Python did not start: {detail}")
        self.returncode = returncode


def jail_sources():
    """Map jail-relative paths to host files: interpreter, standard library, linked libraries."""
    layout = json.loads(subprocess.check_output([str(SYSTEM_PYTHON), "-I", "-S", "-c", LAYOUT], text=True, timeout=60))
    stdlib = Path(layout["stdlib"])
    roots = [stdlib]
    dynload = Path(layout["dynload"]) if layout["dynload"] else stdlib / "lib-dynload"
    if dynload.is_dir() and stdlib not in dynload.parents:
        roots.append(dynload)
    files = {"usr/bin/python3": SYSTEM_PYTHON.resolve()}
    for root in roots:
        for folder, dirs, names in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
            for name in sorted(names):
                path = Path(folder) / name
                target = path.resolve()
                if target.is_file():
                    files[str(path).lstrip("/")] = target
    binaries = [str(SYSTEM_PYTHON.resolve())] + sorted(str(path) for path in dynload.glob("*.so"))
    listing = subprocess.run(["ldd", *binaries], capture_output=True, text=True, timeout=120, check=False).stdout
    for library in sorted(set(LIBRARY.findall(listing))):
        target = Path(library).resolve()
        if target.is_file():
            files[library.lstrip("/")] = target
    if not any(key.startswith(("lib", "usr/lib")) and ".so" in key for key in files):
        raise RuntimeError("could not resolve the shared libraries of /usr/bin/python3 with ldd")
    return files, layout["version"]


def private_root():
    base = Path(os.environ.get("WORK3_SANDBOX_ROOT") or f"/tmp/work3-pair-sandbox-{os.geteuid()}")
    base.mkdir(mode=0o700, exist_ok=True)
    info = os.lstat(base)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise RuntimeError(f"sandbox root {base} must be a private directory owned by uid {os.geteuid()}")
    return base


def build_jail():
    """Copy the sources into a content-addressed, read-only jail (built once)."""
    files, version = jail_sources()
    entries = sorted((key, sha256_file(path), bool(path.stat().st_mode & 0o111)) for key, path in files.items())
    digest = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    base = private_root()
    jail, done = base / f"jail-{digest[:24]}", base / f"jail-{digest[:24]}.ok"
    with (base / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not done.is_file():
            if jail.exists():
                shutil.rmtree(jail)
            staging = Path(tempfile.mkdtemp(prefix="staging-", dir=base))
            for key, _, executable in entries:
                destination = staging / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(files[key], destination)
                destination.chmod(0o755 if executable else 0o644)
            for folder, dirs, _ in os.walk(staging):
                for name in dirs:
                    (Path(folder) / name).chmod(0o755)
            staging.chmod(0o755)
            staging.rename(jail)
            done.write_text(digest + "\n", encoding="utf-8")
    return jail, digest, version


def sealed_code(code):
    """An in-memory, write-sealed copy of the submission passed to the sandbox as a descriptor."""
    descriptor = os.memfd_create("work3-submission", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    data = code.encode("utf-8")
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
    return descriptor


PROBE = r"""import errno, fcntl, os, resource, socket, struct, sys, threading
def verify():
    if os.getuid() == 0 or os.geteuid() == 0:
        return 'submission runs as uid 0 inside the sandbox'
    if os.getppid() != 1 or os.getpid() > 4:
        return 'the PID namespace is not private'
    if resource.getrlimit(resource.RLIMIT_NPROC) != (1, 1):
        return 'RLIMIT_NPROC is not 1'
    try:
        child = os.fork()
    except OSError as exc:
        child, fork_errno = None, exc.errno
    else:
        fork_errno = 0
    if child == 0:
        os._exit(0)
    if child is not None:
        os.waitpid(child, 0)
        return 'fork was allowed'
    if EXPECT_ERRNO and fork_errno != EXPECT_ERRNO:
        return f'unexpected fork errno: {fork_errno} != {EXPECT_ERRNO}'
    try:
        threading.Thread(target=lambda: None).start()
        return 'thread creation was allowed'
    except RuntimeError:
        pass
    names = sorted(name for _, name in socket.if_nameindex())
    extra_ifaces = set(names) - set(FALLBACK_IFACES)
    if 'lo' not in names or extra_ifaces:
        return f'the network namespace has unexpected interfaces: {names}'
    ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for name in names:
            req = struct.pack('16s24x', name.encode('ascii'))
            flags = struct.unpack_from('H', fcntl.ioctl(ctl.fileno(), 0x8913, req), 16)[0]
            if flags & 0x1:
                return f'interface is UP in private netns: {name}'
            try:
                fcntl.ioctl(ctl.fileno(), 0x8915, req)
                return f'interface has an IPv4 address in private netns: {name}'
            except OSError:
                pass
    finally:
        ctl.close()
    for address in (('127.0.0.1', 9), ('10.0.0.1', 9), ('1.1.1.1', 53)):
        probe = socket.socket()
        probe.settimeout(0.2)
        try:
            probe.connect(address)
            return f'network reachable: {address[0]}'
        except OSError:
            pass
        finally:
            probe.close()
    extra = set(os.listdir('/')) - set(ROOT_ENTRIES)
    if extra:
        return f'unexpected jail entries: {sorted(extra)}'
    for path in ('/proc', '/sys', '/dev', '/etc', '/home', '/root', '/tmp'):
        if os.path.exists(path):
            return f'host path visible: {path}'
    folders = ['/', '/usr', '/usr/bin', os.path.dirname(os.__file__)]
    for folder in folders:
        try:
            handle = os.open(os.path.join(folder, '.work3-write-probe'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            continue
        os.close(handle)
        return f'jail directory is writable: {folder}'
    for path in ('/usr/bin/python3', os.__file__):
        try:
            open(path, 'ab').close()
            return f'jail file is writable: {path}'
        except OSError:
            pass
        try:
            os.chmod(path, 0o777)
            return f'jail file mode can change: {path}'
        except OSError:
            pass
    try:
        os.pwrite(3, b'x', 0)
        return 'the submission descriptor is writable'
    except OSError:
        pass
    return sys.stdin.read().strip()
print(verify())
""".replace("ROOT_ENTRIES", repr(ROOT_ENTRIES)).replace("FALLBACK_IFACES", repr(FALLBACK_IFACES))

NPROC_PROBE = """import errno, os, threading
try:
    child = os.fork()
except OSError as exc:
    if exc.errno != errno.EAGAIN:
        print(f'expected EAGAIN from RLIMIT_NPROC, got {exc.errno}')
        raise SystemExit(0)
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    print('fork succeeded without seccomp')
    raise SystemExit(0)
try:
    threading.Thread(target=lambda: None).start()
    print('thread succeeded without seccomp')
except RuntimeError:
    print('nproc-ok')
"""


class Verifier:
    """Run each submission against one input at a time in the mount-free sandbox."""

    def __init__(self):
        if sys.platform != "linux" or not SYSTEM_PYTHON.is_file() or not LAUNCHER.is_file():
            raise RuntimeError("Linux with /usr/bin/python3 is required for submission execution")
        if os.geteuid() == 0:
            self.uid = int(os.environ.get("WORK3_SANDBOX_UID", "65534"))
            self.gid = int(os.environ.get("WORK3_SANDBOX_GID", "65534"))
            self.mode = "drop-first"
        else:
            self.uid, self.gid = os.geteuid(), os.getegid()
            self.mode = "nonroot-ro"
        if self.uid <= 0 or self.gid <= 0:
            raise RuntimeError("WORK3_SANDBOX_UID and WORK3_SANDBOX_GID must be non-zero")
        self.jail, self.jail_digest, self.python = build_jail()
        self.seccomp = True

    def command(self, descriptor, seccomp=None):
        flag = "on" if (self.seccomp if seccomp is None else seccomp) else "off"
        return [str(SYSTEM_PYTHON), "-I", "-S", "-B", str(LAUNCHER), self.mode, flag, str(self.jail),
                str(self.uid), str(self.gid), str(WALL_SECONDS), str(descriptor)]

    def evaluate(self, code, cases, *, seccomp=None):
        if code is None:
            return [("invalid", "") for _ in cases], 0, 0.0
        started = monotonic()
        descriptor = sealed_code(code + "\n")
        try:
            results = [self.run_case(descriptor, given, expected, seccomp=seccomp) for given, expected in cases]
        finally:
            os.close(descriptor)
        return results, len(cases), monotonic() - started

    def run_case(self, descriptor, given, expected, *, seccomp=None):
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(self.command(descriptor, seccomp=seccomp), stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, pass_fds=(descriptor,), start_new_session=True)
            try:
                process.communicate(given.encode("utf-8"), timeout=WALL_SECONDS + BACKSTOP_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, 9)  # the launcher is not reaped yet, so its group id is still ours
                except ProcessLookupError:
                    pass
                process.wait()
                raise RuntimeError("sandbox launcher missed its own deadline") from None
            stderr.seek(0)
            marker = stderr.readline(256)
            if marker != READY:
                detail = (marker + stderr.read(1024)).decode("utf-8", errors="replace").strip()
                raise SandboxStartError(process.returncode, detail or f"exit {process.returncode}")
            if process.returncode == TIMEOUT_EXIT:
                return "timeout", ""
            stdout.seek(0)
            output = normalized(stdout.read(65537).decode("utf-8", errors="replace"))
        if process.returncode != 0:
            return "error", ""
        return ("pass" if output == expected else "wrong"), output[:TEXT_CAP + 1]

    def uid_pids(self):
        """PIDs currently running under the sandbox uid."""
        if os.geteuid() != 0:
            return set()
        found = set()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text()
            except OSError:
                continue
            uid = next((line.split()[1] for line in status.splitlines() if line.startswith("Uid:")), None)
            if uid == str(self.uid):
                found.add(int(entry.name))
        return found

    def probe(self, check_leftovers=True):
        before = self.uid_pids() if check_leftovers else set()
        modes = ("drop-first", "map-helper") if os.geteuid() == 0 else ("nonroot-ro",)
        failure = None
        for mode in modes:
            self.mode = mode
            self.seccomp = True
            try:
                [(status, detail)], _, _ = self.evaluate(
                    PROBE.replace("EXPECT_ERRNO", "errno.EPERM"), [("work3-isolation\n", "work3-isolation")])
                break
            except SandboxStartError as exc:
                if exc.returncode == SECCOMP_DENIED_EXIT:
                    self.seccomp = False
                    [(status, detail)], _, _ = self.evaluate(
                        PROBE.replace("EXPECT_ERRNO", "errno.EAGAIN"), [("work3-isolation\n", "work3-isolation")])
                    break
                if exc.returncode != USERNS_DENIED_EXIT:
                    raise RuntimeError(f"Sandbox preflight failed: {exc}") from exc
                failure = exc
        else:
            raise RuntimeError(f"Sandbox preflight failed: user namespaces are unavailable ({failure}). "
                               "The executor needs user namespaces; it does not need mount, Bubblewrap or privileges.")
        if status != "pass":
            raise RuntimeError(f"Sandbox isolation preflight rejected the verifier: {detail or status}")
        if self.seccomp:
            [(nproc_status, nproc_detail)], _, _ = self.evaluate(NPROC_PROBE, [("", "nproc-ok")], seccomp=False)
            if nproc_status != "pass":
                raise RuntimeError(f"Sandbox RLIMIT_NPROC preflight failed without seccomp: {nproc_detail or nproc_status}")
        started = monotonic()
        [(sleeping, _)], _, _ = self.evaluate("import time\ntime.sleep(30)", [("", "never")])
        wall = monotonic() - started
        [(spinning, _)], _, _ = self.evaluate("while True:\n    pass", [("", "never")])
        if sleeping != "timeout" or wall > WALL_SECONDS + 1.5 or spinning not in ("error", "timeout"):
            raise RuntimeError(f"Sandbox limit preflight failed: sleep={sleeping} after {wall:.2f}s, cpu loop={spinning}")
        remaining = sorted(self.uid_pids() - before) if check_leftovers else []
        for _ in range(20):
            if not remaining:
                break
            sleep(0.05)
            remaining = sorted(self.uid_pids() - before)
        if remaining:
            raise RuntimeError(f"new processes owned by sandbox uid {self.uid} remain: {remaining[:8]}")
        return {"executor": EXECUTOR, "mode": self.mode, "seccomp": self.seccomp, "isolation": "pass",
                "nproc_independent": "pass", "sleep_limit": sleeping, "sleep_seconds": round(wall, 3),
                "cpu_limit": spinning, "leftover_check": "pass" if check_leftovers else "skipped"}

    def version(self):
        return {"executor": EXECUTOR, "mode": self.mode, "seccomp": self.seccomp, "python": self.python,
                "python_sha256": sha256_file(SYSTEM_PYTHON.resolve()), "launcher_sha256": sha256_file(LAUNCHER),
                "jail_sha256": self.jail_digest, "sandbox_uid": self.uid, "sandbox_gid": self.gid}
