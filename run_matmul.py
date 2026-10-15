#!/usr/bin/env python3
"""
EE542 Lab 6 - Call the CUDA tiled matrix multiplication (libmatmul.so) from Python via ctypes.

Build the library first (on the GPU machine):
    nvcc -O2 -arch=sm_75 -Xcompiler -fPIC -shared matmul_lib.cu -o libmatmul.so
Run:
    python3 run_matmul.py              # N = 512, 1024, 2048
    python3 run_matmul.py 256 4096     # any list of N
"""
import ctypes
import os
import sys
import time

import numpy as np
from numpy.ctypeslib import ndpointer

TRIALS = 5

# ---- Load the shared library from this script's folder ----
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmatmul.so")
lib = ctypes.CDLL(lib_path)

# Declare the C signatures. Without these, ctypes guesses (int arguments, int return)
# and passes wrong values silently instead of raising an error.
#   int gpu_matmul(const float *A, const float *B, float *C, int N, float *timings_ms)
f32_matrix = ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
f32_vector = ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
lib.gpu_matmul.argtypes = [f32_matrix, f32_matrix, f32_matrix, ctypes.c_int, f32_vector]
lib.gpu_matmul.restype = ctypes.c_int
lib.gpu_error_string.argtypes = [ctypes.c_int]
lib.gpu_error_string.restype = ctypes.c_char_p


def gpu_matmul(A: np.ndarray, B: np.ndarray):
    """Return (C, timings) where C = A @ B computed on the GPU and
    timings = [H2D ms, kernel ms, D2H ms]."""
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    N = A.shape[0]
    # The C code reads raw float32, row-major memory: convert if needed (no copy if already so)
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.empty((N, N), dtype=np.float32)
    timings = np.zeros(3, dtype=np.float32)
    code = lib.gpu_matmul(A, B, C, N, timings)
    if code != 0:
        raise RuntimeError(f"gpu_matmul failed: {lib.gpu_error_string(code).decode()}")
    return C, timings


def median(values):
    return float(np.median(values))


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [512, 1024, 2048]
    rng = np.random.default_rng(42)

    # The first CUDA call creates the context (~150 ms on this machine); keep it out of timings
    t = time.perf_counter()
    gpu_matmul(np.ones((16, 16), np.float32), np.ones((16, 16), np.float32))
    print(f"First call (CUDA context creation): {(time.perf_counter() - t) * 1000:.1f} ms\n")

    header = (f"{'N':>6} {'Python call':>12} {'H2D':>8} {'kernel':>9} {'D2H':>8} "
              f"{'call overhead':>14} {'NumPy (CPU)':>12} {'GPU vs NumPy':>13}  check")
    print(header)
    print("-" * len(header))

    for N in sizes:
        A = rng.integers(0, 100, size=(N, N)).astype(np.float32) / 100.0
        B = rng.integers(0, 100, size=(N, N)).astype(np.float32) / 100.0

        # GPU through ctypes: wall-clock time of the whole Python call, plus the C-side breakdown
        wall, h2d, ker, d2h = [], [], [], []
        for trial in range(TRIALS + 1):          # trial 0 is a warm-up
            t = time.perf_counter()
            C, tm = gpu_matmul(A, B)
            dt = (time.perf_counter() - t) * 1000
            if trial > 0:
                wall.append(dt); h2d.append(tm[0]); ker.append(tm[1]); d2h.append(tm[2])

        # CPU reference: NumPy's matmul (calls an optimized multithreaded BLAS)
        cpu = []
        for trial in range(TRIALS + 1):
            t = time.perf_counter()
            C_ref = A @ B
            dt = (time.perf_counter() - t) * 1000
            if trial > 0:
                cpu.append(dt)

        ok = np.allclose(C, C_ref, rtol=1e-4, atol=1e-3)
        max_rel = float(np.max(np.abs(C - C_ref) / (np.abs(C_ref) + 1e-12)))
        w, a, k, d, c = median(wall), median(h2d), median(ker), median(d2h), median(cpu)
        overhead = w - (a + k + d)   # malloc/free, event setup and the ctypes call itself
        print(f"{N:>6} {w:>10.3f}ms {a:>6.3f}ms {k:>7.3f}ms {d:>6.3f}ms "
              f"{overhead:>12.3f}ms {c:>10.3f}ms {c / w:>12.2f}x  "
              f"{'PASS' if ok else 'FAIL'} (max rel err {max_rel:.2e})")

    print("\nAll times are medians of", TRIALS, "runs after one warm-up.")


if __name__ == "__main__":
    main()
