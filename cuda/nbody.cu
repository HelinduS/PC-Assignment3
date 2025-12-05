#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define DIM 3
#define BLOCK_SIZE 256

typedef struct {
    float pos[DIM];
    float vel[DIM];
    float mass;
} Body;

// Initialize bodies (same as your CPU version)
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

// CUDA Kernel: Each thread computes force on one body from all others
__global__ void compute_forces_kernel(Body *bodies, float *acc, int N, float G, float softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float my_pos_x = bodies[i].pos[0];
    float my_pos_y = bodies[i].pos[1];
    float my_pos_z = bodies[i].pos[2];

    for (int j = 0; j < N; j++) {
        if (i == j) continue;

        float dx = bodies[j].pos[0] - my_pos_x;
        float dy = bodies[j].pos[1] - my_pos_y;
        float dz = bodies[j].pos[2] - my_pos_z;

        float dist2 = dx*dx + dy*dy + dz*dz + softening;
        float inv_dist = rsqrtf(dist2);           // 1 / sqrt(dist2)
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float F = G * inv_dist3 * bodies[j].mass;

        ax += dx * F;
        ay += dy * F;
        az += dz * F;
    }

    acc[i * DIM + 0] = ax;
    acc[i * DIM + 1] = ay;
    acc[i * DIM + 2] = az;
}

// Update positions and velocities (can run on CPU or GPU — we do CPU for simplicity)
void update_bodies(Body *bodies, float *acc, int N, float dt) {
    for (int i = 0; i < N; i++) {
        float inv_mass = 1.0f / bodies[i].mass;
        for (int d = 0; d < DIM; d++) {
            int idx = i * DIM + d;
            bodies[i].vel[d] += acc[idx] * dt * inv_mass;
            bodies[i].pos[d] += bodies[i].vel[d] * dt;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;     // Try 4096 or 8192
    int steps = (argc > 2) ? atoi(argv[2]) : 100;
    float G = 1.0f;
    float dt = 0.01f;
    float softening = 1e-9f;

    // Host memory
    Body *h_bodies = (Body*)malloc(N * sizeof(Body));
    float *h_acc = (float*)malloc(N * DIM * sizeof(float));

    // Device memory
    Body *d_bodies;
    float *d_acc;
    cudaMalloc(&d_bodies, N * sizeof(Body));
    cudaMalloc(&d_acc, N * DIM * sizeof(float));

    // Initialize
    init_bodies(h_bodies, N);
    cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Grid and block setup
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int s = 0; s < steps; s++) {
        compute_forces_kernel<<<grid_size, BLOCK_SIZE>>>(d_bodies, d_acc, N, G, softening);
        cudaDeviceSynchronize();  // Wait for kernel

        // Copy acceleration back and update on CPU
        cudaMemcpy(h_acc, d_acc, N * DIM * sizeof(float), cudaMemcpyDeviceToHost);
        update_bodies(h_bodies, h_acc, N, dt);

        // Copy updated positions back to GPU
        cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("CUDA time: %.4f s (N = %d, steps = %d)\n", milliseconds / 1000.0f, N, steps);

    // Print first 5 bodies
    for (int i = 0; i < 5 && i < N; i++) {
        printf("Body %d: pos = (%.2f, %.2f, %.2f), vel = (%.2f, %.2f, %.2f)\n",
               i,
               h_bodies[i].pos[0], h_bodies[i].pos[1], h_bodies[i].pos[2],
               h_bodies[i].vel[0], h_bodies[i].vel[1], h_bodies[i].vel[2]);
    }

    // Cleanup
    free(h_bodies); free(h_acc);
    cudaFree(d_bodies); cudaFree(d_acc);

    return 0;
}