#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void compute_forces_chunk(Body *b, float *acc_chunk, int start, int end, int N, float G, float soft) {
    float soft2 = soft * soft;

    for (int i = start; i < end; i++) {
        acc_chunk[(i-start)*DIM+0] = 0.0f;
        acc_chunk[(i-start)*DIM+1] = 0.0f;
        acc_chunk[(i-start)*DIM+2] = 0.0f;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            float dx = b[j].pos[0] - b[i].pos[0];
            float dy = b[j].pos[1] - b[i].pos[1];
            float dz = b[j].pos[2] - b[i].pos[2];

            float dist2 = dx*dx + dy*dy + dz*dz + soft2;
            float invd  = 1.0f / sqrtf(dist2);
            float invd3 = invd * invd * invd;

            float F = G * invd3;

            acc_chunk[(i-start)*DIM+0] += dx * F * b[j].mass;
            acc_chunk[(i-start)*DIM+1] += dy * F * b[j].mass;
            acc_chunk[(i-start)*DIM+2] += dz * F * b[j].mass;
        }
    }
}

void update(Body *b, float *acc, int N, float dt) {
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < DIM; d++) {
            b[i].vel[d] += acc[i*DIM + d] * dt / b[i].mass;
            b[i].pos[d] += b[i].vel[d] * dt;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int steps = (argc > 2) ? atoi(argv[2]) : 100;

    float G = 1.0f;
    float dt = 0.01f;
    float soft = 1e-9f;

    Body *bodies = malloc(N * sizeof(Body));
    float *acc_global = NULL;

    if (rank == 0) {
        init_bodies(bodies, N);
        acc_global = malloc(N * DIM * sizeof(float));
    }

    // Broadcast initial bodies to all ranks
    MPI_Bcast(bodies, N * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int chunk = N / size;
    int rem = N % size;
    int start, end;
    for (int i = 0; i < size; i++) {
        counts[i] = (i < rem ? chunk + 1 : chunk) * DIM;
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }
    start = 0;
    for (int i = 0; i < rank; i++) {
        start += (i < rem ? chunk + 1 : chunk);
    }
    end = start + (rank < rem ? chunk + 1 : chunk);

    float *acc_chunk = malloc((end-start) * DIM * sizeof(float));

    double start_time = MPI_Wtime();

    for (int s = 0; s < steps; s++) {

        // Compute only my assigned i-range
        compute_forces_chunk(bodies, acc_chunk, start, end, N, G, soft);

        // Gather all partial accelerations into rank 0 using MPI_Gatherv
        MPI_Gatherv(
            acc_chunk, (end-start)*DIM, MPI_FLOAT,
            acc_global, counts, displs, MPI_FLOAT,
            0, MPI_COMM_WORLD
        );

        // Update only in rank 0
        if (rank == 0) {
            update(bodies, acc_global, N, dt);
        }

        // Broadcast updated bodies back to workers
        MPI_Bcast(bodies, N * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("MPI time: %.4f s (%d processes)\n", end_time - start_time, size);
        //print final positions and velocities of first 5 bodies
        for (int i = 0; i < 5 && i < N; i++) {
            printf("Body %d: pos = (%.2f, %.2f, %.2f), vel = (%.2f, %.2f, %.2f)\n",
                i,
                bodies[i].pos[0], bodies[i].pos[1], bodies[i].pos[2],
                bodies[i].vel[0], bodies[i].vel[1], bodies[i].vel[2]);
        }
    }

    free(bodies);
    free(acc_chunk);
    if (rank == 0) free(acc_global);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}