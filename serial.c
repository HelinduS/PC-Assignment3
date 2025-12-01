#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void compute_forces(Body *b, float *a, int N, float G, float softening) {
    float soft2 = softening * softening;

    // Reset acceleration
    for (int i = 0; i < N * DIM; i++) a[i] = 0.0f;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            float dx = b[j].pos[0] - b[i].pos[0];
            float dy = b[j].pos[1] - b[i].pos[1];
            float dz = b[j].pos[2] - b[i].pos[2];

            float dist2 = dx*dx + dy*dy + dz*dz + soft2;
            float inv_dist = 1.0f / sqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float Fi = G * inv_dist3;

            // i ← j’s pull
            a[i*DIM + 0] += dx * Fi * b[j].mass;
            a[i*DIM + 1] += dy * Fi * b[j].mass;
            a[i*DIM + 2] += dz * Fi * b[j].mass;

            // j ← i’s pull (opposite direction)
            a[j*DIM + 0] -= dx * Fi * b[i].mass;
            a[j*DIM + 1] -= dy * Fi * b[i].mass;
            a[j*DIM + 2] -= dz * Fi * b[i].mass;
        }
    }
}

void update(Body *b, float *a, int N, float dt) {
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

    double start = clock();

    for (int s = 0; s < steps; s++) {
        compute_forces(bodies, acc, N, G, softening);
        update(bodies, acc, N, dt);
    }

    double end = clock();
    printf("Serial time: %.4f s\n", (end - start) / CLOCKS_PER_SEC);

    free(bodies);
    free(acc);
    return 0;
}