#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DIM 3

typedef struct {
    float pos[DIM];
    float vel[DIM];
    float mass;
} Body;

void init_bodies(Body *b, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            b[i].pos[d] = ((float)rand() / RAND_MAX) * 100.0f - 50.0f;
            b[i].vel[d] = 0.0f;
        }
        b[i].mass = ((float)rand() / RAND_MAX) * 10.0f + 1.0f;
    }
}

void compute_forces_omp(Body *b, float *a, int N, float G, float softening) {
    float soft2 = softening * softening;
    int T = omp_get_max_threads();

    // Reset global acceleration
    #pragma omp parallel for
    for (int i = 0; i < N * DIM; i++)
        a[i] = 0.0f;

    // Per-thread private acceleration arrays
    float (*priv)[DIM] = calloc(N * T, sizeof(float[DIM]));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float (*my)[DIM] = priv + tid * N;

        // Parallelize the i-loop, symmetric j>i preserved
        #pragma omp for schedule(guided)
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {

                float dx = b[j].pos[0] - b[i].pos[0];
                float dy = b[j].pos[1] - b[i].pos[1];
                float dz = b[j].pos[2] - b[i].pos[2];

                float dist2 = dx*dx + dy*dy + dz*dz + soft2;
                float invd = 1.0f / sqrtf(dist2);
                float invd3 = invd * invd * invd;

                float F = G * invd3;

                // i gets pull from j
                my[i][0] += dx * F * b[j].mass;
                my[i][1] += dy * F * b[j].mass;
                my[i][2] += dz * F * b[j].mass;

                // j gets opposite pull from i
                my[j][0] -= dx * F * b[i].mass;
                my[j][1] -= dy * F * b[i].mass;
                my[j][2] -= dz * F * b[i].mass;
            }
        }
    }

    // Merge each thread's private forces into the global array
    for (int t = 0; t < T; t++) {
        float (*src)[DIM] = priv + t * N;
        for (int i = 0; i < N; i++) {
            a[i*DIM + 0] += src[i][0];
            a[i*DIM + 1] += src[i][1];
            a[i*DIM + 2] += src[i][2];
        }
    }

    free(priv);
}

void update(Body *b, float *a, int N, float dt) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            b[i].vel[d] += a[i*DIM + d] * dt / b[i].mass;
            b[i].pos[d] += b[i].vel[d] * dt;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int steps = (argc > 2) ? atoi(argv[2]) : 100;

    float G = 1.0f;
    float dt = 0.01f;
    float softening = 1e-9f;

    Body *bodies = malloc(N * sizeof(Body));
    float *acc = malloc(N * DIM * sizeof(float));

    init_bodies(bodies, N);

    double start = omp_get_wtime();

    for (int s = 0; s < steps; s++) {
        compute_forces_omp(bodies, acc, N, G, softening);
        update(bodies, acc, N, dt);
    }

    double end = omp_get_wtime();
    printf("OpenMP time: %.4f s (%d threads)\n", end - start, omp_get_max_threads());

    //print final positions and velocities of first 5 bodies
    for (int i = 0; i < 5 && i < N; i++) {
        printf("Body %d: pos = (%.2f, %.2f, %.2f), vel = (%.2f, %.2f, %.2f)\n",
            i,
            bodies[i].pos[0], bodies[i].pos[1], bodies[i].pos[2],
            bodies[i].vel[0], bodies[i].vel[1], bodies[i].vel[2]);
    }

    free(bodies);
    free(acc);
    return 0;
}