%%writefile nbody.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define DIM 3

typedef struct {
    float pos[DIM];
    float vel[DIM];
    float mass;
} Body;

__global__
void compute_forces(const Body *b, float *a, int N, float G, float soft2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float ax = 0, ay = 0, az = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;

        float dx = b[j].pos[0] - b[i].pos[0];
        float dy = b[j].pos[1] - b[i].pos[1];
        float dz = b[j].pos[2] - b[i].pos[2];

        float dist2 = dx*dx + dy*dy + dz*dz + soft2;
        float invd = rsqrtf(dist2);
        float invd3 = invd * invd * invd;

        float F = G * invd3;

        ax += dx * F * b[j].mass;
        ay += dy * F * b[j].mass;
        az += dz * F * b[j].mass;
    }

    a[i*DIM + 0] = ax;
    a[i*DIM + 1] = ay;
    a[i*DIM + 2] = az;
}

int main() {
    int N = 1024;
    float G = 1.0f;
    float soft2 = 1e-9f * 1e-9f;
    int steps = 10;

    Body *h_bodies = (Body*)malloc(N * sizeof(Body));
    float *h_acc = (float*)malloc(N * DIM * sizeof(float));

    // Initialize bodies
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            h_bodies[i].pos[d] = ((float)rand()/RAND_MAX)*100.0f - 50.0f;
            h_bodies[i].vel[d] = 0.0f;
        }
        h_bodies[i].mass = ((float)rand()/RAND_MAX)*10.0f + 1.0f;
    }

    // Allocate on device
    Body *d_bodies;
    float *d_acc;
    cudaMalloc(&d_bodies, N * sizeof(Body));
    cudaMalloc(&d_acc, N * DIM * sizeof(float));

    cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    int block = 1024;
    int grid = (N + block - 1) / block;

    // Timing
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < steps; s++) {
        compute_forces<<<grid, block>>>(d_bodies, d_acc, N, G, soft2);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_acc, d_acc, N * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUDA time: %.3f ms\n", ms);

    cudaFree(d_bodies);
    cudaFree(d_acc);
    free(h_bodies);
    free(h_acc);

    return 0;
}