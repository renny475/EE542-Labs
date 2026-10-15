// EE542 Lab6 - CPU matrix multiplication benchmark (data collection in C)
// Build: gcc -O2 -Wall bench_cpu.c -o bench_cpu
// Run:   ./bench_cpu              (default N list, 3 trials, writes results_cpu.csv)
//        ./bench_cpu 5            (5 trials per N)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplyCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Wall-clock time in seconds (unaffected by NTP adjustments)
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

// Time one N, `trials` times; returns median, stores each trial in out[]
static double bench_one(int N, int trials, double* out) {
    size_t n2 = (size_t)N * N;  // avoid int overflow
    float* A = malloc(n2 * sizeof(float));
    float* B = malloc(n2 * sizeof(float));
    float* C = malloc(n2 * sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed for N=%d\n", N);
        exit(1);
    }

    srand(42);  // fixed seed: reproducible input
    for (size_t i = 0; i < n2; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    matrixMultiplyCPU(A, B, C, N);  // warm-up (page faults, cache, CPU freq)

    volatile float sink = 0.0f;  // prevent compiler from removing the work
    for (int t = 0; t < trials; t++) {
        double t0 = now_sec();
        matrixMultiplyCPU(A, B, C, N);
        double t1 = now_sec();
        out[t] = t1 - t0;
        sink += C[0];
    }
    (void)sink;

    free(A);
    free(B);
    free(C);

    double sorted[trials];
    for (int t = 0; t < trials; t++) sorted[t] = out[t];
    qsort(sorted, trials, sizeof(double), cmp_double);
    return sorted[trials / 2];
}

int main(int argc, char** argv) {
    int trials = 3;
    if (argc > 1) {
        char* end;
        long v = strtol(argv[1], &end, 10);
        if (*end != '\0' || v < 1 || v > 100) {
            fprintf(stderr, "usage: %s [trials 1-100]\n", argv[0]);
            return 1;
        }
        trials = (int)v;
    }

    int Ns[] = {128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048};
    int count = sizeof(Ns) / sizeof(Ns[0]);

    FILE* f = fopen("results_cpu.csv", "w");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fprintf(f, "N,trial,time_s\n");

    double times[100];
    for (int i = 0; i < count; i++) {
        int N = Ns[i];
        double med = bench_one(N, trials, times);
        for (int t = 0; t < trials; t++) {
            fprintf(f, "%d,%d,%.6f\n", N, t + 1, times[t]);
        }
        fflush(f);  // keep partial results if interrupted
        double gflops = 2.0 * N * N * N / med / 1e9;
        printf("N=%5d  median=%9.4f s  %6.2f GFLOPS\n", N, med, gflops);
    }

    fclose(f);
    printf("Saved results_cpu.csv\n");
    return 0;
}