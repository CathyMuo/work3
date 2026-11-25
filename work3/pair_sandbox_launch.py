"""Mount-free launcher for one untrusted Python submission (Linux only).

The verifier runs this file with the system interpreter (`/usr/bin/python3 -I -S -B`).
When invoked as root (e.g. inside a container whose AppArmor profile blocks mount(2)),
it never calls mount(2) and needs no Bubblewrap. Layers, outermost first:

1. The submission runs as an unprivileged outside uid (never uid 0), so the kernel
   enforces RLIMIT_NPROC and the submission cannot write the root-owned jail.
2. Private user, network, IPC, UTS and PID namespaces. The network namespace only has a
   down loopback device. The supervisor is PID 1 of the private PID namespace, so every
   process inside dies atomically with it.
3. chroot into a minimal, content-addressed jail holding only `/usr/bin/python3`, its
   standard library and the shared libraries it links.
4. Hard rlimits (including RLIMIT_NPROC=1), PR_SET_NO_NEW_PRIVS, and a kernel
   seccomp-BPF filter blocking process/thread creation and namespace/mount syscalls.
5. The launcher enforces the wall-clock limit itself, sends SIGKILL to PID 1 of the
   private PID namespace at the deadline and reaps it, leaving no orphans or zombies.

Usage: pair_sandbox_launch.py MODE SECCOMP JAIL UID GID WALL_SECONDS CODE_FD
MODE is `drop-first`, `map-helper` (both mount-free, for root) or `nonroot-ro`
(uses a private read-only bind mount when invoked as a non-root user on dev hosts).
SECCOMP is `on` or `off`.
"""

import ctypes
import errno
import os
import resource
import signal
import struct
import sys

CLONE_NEWNS = 0x00020000
CLONE_NEWUTS = 0x04000000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000
NAMESPACES = CLONE_NEWUSER | CLONE_NEWNET | CLONE_NEWIPC | CLONE_NEWUTS | CLONE_NEWPID
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 4, 32, 4096, 16384, 262144
PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
BPF_LD_W_ABS, BPF_JMP_JEQ_K, BPF_RET_K = 0x20, 0x15, 0x06
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_EPERM = 0x00050000 | errno.EPERM
SECCOMP_POLICY = {
    "x86_64": (0xC000003E, (56, 57, 58, 62, 101, 165, 166, 272, 308, 310, 311, 322, 435)),
    "aarch64": (0xC00000B7, (39, 40, 97, 117, 129, 220, 268, 270, 271, 281, 435)),
}
INSIDE_ID = 1000
TIMEOUT_EXIT = 124
SETUP_EXIT = 125
USERNS_DENIED_EXIT = 126
SECCOMP_DENIED_EXIT = 127
LIMITS = ((resource.RLIMIT_CPU, 1), (resource.RLIMIT_AS, 268435456), (resource.RLIMIT_FSIZE, 65536),
          (resource.RLIMIT_NOFILE, 32), (resource.RLIMIT_NPROC, 1), (resource.RLIMIT_CORE, 0))
INNER = ("import os,sys; sys.stderr.write('WORK3_SANDBOX_READY\\n'); sys.stderr.flush(); "
         "s=os.pread(3,os.fstat(3).st_size,0); os.close(3); "
         "g={'__name__':'__main__','__file__':'/submission.py','__builtins__':__builtins__}; "
         "exec(compile(s,'/submission.py','exec'),g)")
ENVIRONMENT = {"PATH": "/usr/bin", "LC_ALL": "C.UTF-8"}

libc = ctypes.CDLL(None, use_errno=True)


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def fail(message, code=SETUP_EXIT):
    os.write(2, f"WORK3_SANDBOX_ERROR {message}\n".encode("utf-8", "replace"))
    os._exit(code)


def call(name, *args, code=SETUP_EXIT):
    if getattr(libc, name)(*[ctypes.c_ulong(arg) for arg in args]) != 0:
        fail(f"{name}: {os.strerror(ctypes.get_errno())}", code)


def write(path, text):
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.write(descriptor, text.encode())
    finally:
        os.close(descriptor)


def die_with_parent(parent):
    call("prctl", PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if os.getppid() != parent:  # the parent already exited before the signal was armed
        os._exit(SETUP_EXIT)


def exit_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    return 128 + os.WTERMSIG(status) if os.WIFSIGNALED(status) else SETUP_EXIT


def install_seccomp():
    machine = os.uname().machine
    policy = SECCOMP_POLICY.get(machine)
    if policy is None:
        fail(f"seccomp unsupported on {machine}", SECCOMP_DENIED_EXIT)
    arch, denied = policy
    n = len(denied)
    program = [
        (BPF_LD_W_ABS, 0, 0, 4),          # seccomp_data.arch
        (BPF_JMP_JEQ_K, 0, n + 2, arch),  # mismatch -> EPERM
        (BPF_LD_W_ABS, 0, 0, 0),          # seccomp_data.nr
    ]
    for i, nr in enumerate(denied):
        program.append((BPF_JMP_JEQ_K, n - i, 0, nr))
    program.append((BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    program.append((BPF_RET_K, 0, 0, SECCOMP_RET_EPERM))
    raw = b"".join(struct.pack("<HBBI", *inst) for inst in program)
    buf = ctypes.create_string_buffer(raw, len(raw))
    fprog = SockFprog(len(program), ctypes.addressof(buf))
    if libc.prctl(ctypes.c_ulong(PR_SET_SECCOMP), ctypes.c_ulong(SECCOMP_MODE_FILTER),
                  ctypes.c_ulong(ctypes.addressof(fprog)), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        fail(f"seccomp: {os.strerror(ctypes.get_errno())}", SECCOMP_DENIED_EXIT)


def submission(code_fd, use_seccomp):
    """Last process: rlimits, no new privileges, seccomp-BPF, exec the interpreter."""
    try:
        die_with_parent(1)  # the supervisor is PID 1 of the namespace
        if code_fd != 3:
            os.dup2(code_fd, 3)
        os.set_inheritable(3, True)
        os.closerange(4, max(4096, code_fd + 1))
        for kind, value in LIMITS:
            resource.setrlimit(kind, (value, value))
        call("prctl", PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        if use_seccomp:
            install_seccomp()
        os.execve("/usr/bin/python3", ["python3", "-I", "-B", "-S", "-c", INNER], ENVIRONMENT)
    except BaseException as exc:  # noqa: BLE001 - report every setup error as infrastructure
        fail(f"exec: {exc}")


def supervisor(code_fd, use_seccomp):
    """PID 1 of the private PID namespace. Its exit kills everything left inside."""
    try:
        die_with_parent(0)  # the launcher lives in the parent PID namespace
        child = os.fork()
        if child == 0:
            submission(code_fd, use_seccomp)
        while True:
            pid, status = os.waitpid(-1, 0)
            if pid == child:
                os._exit(exit_code(status))
    except BaseException as exc:  # noqa: BLE001
        fail(f"supervisor: {exc}")


def enter_drop_first(jail_fd, uid, gid, parent):
    if os.geteuid() != 0:
        fail("drop-first mode requires root")
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    call("prctl", PR_SET_DUMPABLE, 1, 0, 0, 0)  # lets the new uid write its own id maps
    die_with_parent(parent)  # credential changes clear the parent-death signal
    outer_uid, outer_gid = os.geteuid(), os.getegid()
    call("unshare", NAMESPACES, code=USERNS_DENIED_EXIT)
    write("/proc/self/setgroups", "deny")
    write("/proc/self/uid_map", f"{INSIDE_ID} {outer_uid} 1\n")
    write("/proc/self/gid_map", f"{INSIDE_ID} {outer_gid} 1\n")
    os.fchdir(jail_fd)
    os.chroot(".")
    os.chdir("/")


def enter_map_helper(jail_fd, uid, gid, parent):
    if os.geteuid() != 0:
        fail("map-helper mode requires root")
    os.setgroups([])
    ready_read, ready_write = os.pipe()
    done_read, done_write = os.pipe()
    launcher = os.getpid()
    helper = os.fork()
    if helper == 0:  # stays root in the parent namespaces and only writes the maps
        try:
            os.close(ready_write)
            os.close(done_read)
            if os.read(ready_read, 1) != b"1":
                os._exit(SETUP_EXIT)
            write(f"/proc/{launcher}/setgroups", "deny")
            write(f"/proc/{launcher}/uid_map", f"{INSIDE_ID} {uid} 1\n")
            write(f"/proc/{launcher}/gid_map", f"{INSIDE_ID} {gid} 1\n")
            os.write(done_write, b"1")
            os._exit(0)
        except BaseException:  # noqa: BLE001
            os._exit(SETUP_EXIT)
    os.close(ready_read)
    os.close(done_write)
    if libc.unshare(ctypes.c_ulong(NAMESPACES)) != 0:
        error = ctypes.get_errno()
        os.close(ready_write)
        os.waitpid(helper, 0)
        fail(f"unshare: {os.strerror(error)}", USERNS_DENIED_EXIT)
    os.write(ready_write, b"1")
    os.close(ready_write)
    mapped = os.read(done_read, 1) == b"1"
    _, status = os.waitpid(helper, 0)
    if not mapped or exit_code(status) != 0:
        fail("helper could not write the id maps")
    os.fchdir(jail_fd)
    os.chroot(".")
    os.chdir("/")
    os.setresgid(INSIDE_ID, INSIDE_ID, INSIDE_ID)
    os.setresuid(INSIDE_ID, INSIDE_ID, INSIDE_ID)  # outside uid becomes UID; capabilities are cleared
    die_with_parent(parent)


def enter_nonroot_ro(jail, parent):
    """Fallback when invoked as a non-root dev user who owns the jail directory."""
    outer_uid, outer_gid = os.geteuid(), os.getegid()
    call("unshare", NAMESPACES | CLONE_NEWNS, code=USERNS_DENIED_EXIT)
    write("/proc/self/setgroups", "deny")
    write("/proc/self/uid_map", f"{INSIDE_ID} {outer_uid} 1\n")
    write("/proc/self/gid_map", f"{INSIDE_ID} {outer_gid} 1\n")
    jail_bytes = os.fsencode(jail)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    if libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None) != 0:
        fail(f"mount private: {os.strerror(ctypes.get_errno())}")
    if libc.mount(jail_bytes, jail_bytes, None, MS_BIND | MS_REC, None) != 0:
        fail(f"mount bind: {os.strerror(ctypes.get_errno())}")
    flags = os.statvfs(jail).f_flag | MS_BIND | MS_REMOUNT | MS_RDONLY
    if libc.mount(None, jail_bytes, None, flags, None) != 0:
        fail(f"mount ro: {os.strerror(ctypes.get_errno())}")
    os.chdir(jail)
    os.chroot(".")
    os.chdir("/")
    die_with_parent(parent)


def main():
    mode, seccomp_flag, jail, uid, gid, wall, code_fd = sys.argv[1:8]
    uid, gid, wall, code_fd = int(uid), int(gid), float(wall), int(code_fd)
    use_seccomp = seccomp_flag == "on"
    if seccomp_flag not in ("on", "off"):
        fail(f"unknown seccomp mode {seccomp_flag}")
    if uid <= 0 or gid <= 0:
        fail("the sandbox uid and gid must be non-zero")
    parent = os.getppid()
    die_with_parent(parent)
    jail_fd = os.open(jail, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if mode == "drop-first":
            enter_drop_first(jail_fd, uid, gid, parent)
        elif mode == "map-helper":
            enter_map_helper(jail_fd, uid, gid, parent)
        elif mode == "nonroot-ro":
            enter_nonroot_ro(jail, parent)
        else:
            fail(f"unknown mode {mode}")
    except OSError as exc:
        fail(f"{mode}: {exc}")
    os.close(jail_fd)
    if os.getuid() != INSIDE_ID or os.geteuid() != INSIDE_ID:
        fail("the submission would not run as the unprivileged inside uid")
    child = os.fork()
    if child == 0:
        supervisor(code_fd, use_seccomp)
    expired = []

    def expire(*_):
        expired.append(True)
        try:
            os.kill(child, signal.SIGKILL)  # PID 1 of the namespace; the kernel kills the rest
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, wall)
    _, status = os.waitpid(child, 0)
    signal.setitimer(signal.ITIMER_REAL, 0)
    os._exit(TIMEOUT_EXIT if expired else exit_code(status))


if __name__ == "__main__":
    main()
