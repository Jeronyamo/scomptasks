#define _UCRT

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <omp.h>



//  Constants  \\

#define MY_PI 3.14159265358979323846
//#define PRINT_U

#ifndef TIME_UTC
#define TIME_UTC 1
#endif

//typedef float fptype;
typedef double fptype;

const fptype a2 = 1. / 4;
const unsigned N1 = 128, N2 = 256, K = 20;
const fptype tau = 1e-3, tau2 = 1e-6;


struct Vec3 {
    fptype x, y, z;
} const L1 = { 1., 1., 1. }, LP = { MY_PI, MY_PI, MY_PI };


//  Utilities  \\

unsigned long ind(unsigned i, unsigned j, unsigned k, const unsigned N) {
    return i * (N + 1) * (N + 1) + j * (N + 1) + k;
}

double timediff(const struct timespec t1, const struct timespec t2) {
    return t1.tv_sec - t2.tv_sec + (double) (t1.tv_nsec - t2.tv_nsec) * 1e-9;
}

fptype sqr(const fptype _v) { return _v * _v; };
fptype max(fptype v1, fptype v2) { return  v1 >= v2 ? v1 : v2; }


//  U analytical  \\

fptype u_an(fptype x, fptype y, fptype z, fptype t, const struct Vec3 L) {
    return sin((    MY_PI / L.x) * x) *
           sin((4 * MY_PI / L.y) * y) *
           sin((9 * MY_PI / L.z) * z) *
           cos((MY_PI / 2) * sqrt( 1 / sqr(L.x) + 4 / sqr(L.y) + 9 / sqr(L.z) ) * t);
}

// Phi = U analytical with t = 0
fptype phi(fptype x, fptype y, fptype z, const struct Vec3 L) {
    return sin((    MY_PI / L.x) * x) *
           sin((4 * MY_PI / L.y) * y) *
           sin((9 * MY_PI / L.z) * z);
}


//  Code  \\

fptype deltah(unsigned i, unsigned j, unsigned k, fptype *u_n, const unsigned N, const struct Vec3 h2) {
    if (!i || !k || i == N || k == N) return 0;
    return (u_n[ind(i, (j ? j - 1 : N - 1), k, N)] + u_n[ind(i, (j < N ? j + 1 : 1), k, N)] +
            u_n[ind(i - 1, j, k, N)] + u_n[ind(i + 1, j, k, N)] +
            u_n[ind(i, j, k - 1, N)] + u_n[ind(i, j, k + 1, N)] - 6 * u_n[ind(i, j, k, N)]) / sqrt(h2.x) * 2;
}


void comp_all(int nMaj, int nMin, const struct Vec3 L, const unsigned N, const unsigned NThreads) {
    if (nMin == 1) printf("\n");
    printf("\n%d.%d -- L = %f, N = %d, NThreads = %d\n", nMaj, nMin, L.x, N, NThreads);

    const unsigned num_chunks = 3u;
    struct timespec start, finish;
    const unsigned N3 = (N + 1) * (N + 1) * (N + 1);
    fptype *ref = malloc(num_chunks * N3 * sizeof(fptype));
    fptype *ref0, *ref1, *ref2;

    if (ref == NULL) {
        printf("Memory allocation error");
        return;
    }
    memset(ref, 0, num_chunks * N3 * sizeof(fptype));
    ref0 = ref;
    ref1 = ref + N3;
    ref2 = ref + 2 * N3;

    const struct Vec3 h  = { L.x / N, L.y / N, L.z / N };
    const struct Vec3 h2 = { sqr(h.x), sqr(h.y), sqr(h.z) };
    fptype maxv = 0.;


    // U0
    timespec_get(&start, TIME_UTC);
    #pragma omp parallel for collapse(3) num_threads(NThreads)
    for (unsigned i = 0u; i <= N; ++i)
        for (unsigned j = 0u; j <= N; ++j)
            for (unsigned k = 0u; k <= N; ++k)
                ref0[ind(i, j, k, N)] = phi(h.x * i, h.y * j, h.z * k, L);


    // U1
    #pragma omp parallel for collapse(3) num_threads(NThreads)
    for (unsigned i = 0u; i <= N; ++i)
        for (unsigned j = 0u; j <= N; ++j)
            for (unsigned k = 0u; k <= N; ++k)
                ref1[ind(i, j, k, N)] = ref0[ind(i, j, k, N)] + tau2 / 2 * deltah(i, j, k, ref0, N, h2);

#ifdef PRINT_U
    char buf[16];
    memset(buf, 0, 16);
    sprintf(buf, "u_%c_%d.txt", L.x == 1 ? '1' : 'P', N);
    FILE *f = fopen(buf, "w");

    fprintf(f, "t=0\n");
    printf("t=0\r");
    for (unsigned i = 0u; i <= N; ++i)
        for (unsigned j = 0u; j <= N; ++j)
            for (unsigned k = 0u; k <= N; ++k)
                fprintf(f, "%f\n", ref0[ind(i, j, k, N)]);

    fprintf(f, "\nt=1\n");
    printf("t=1\r");
    for (unsigned i = 0u; i <= N; ++i)
        for (unsigned j = 0u; j <= N; ++j)
            for (unsigned k = 0u; k <= N; ++k)
                fprintf(f, "%f\n", ref1[ind(i, j, k, N)]);
#endif

    // t = 2, ..., K
    for (unsigned t = 2u; t < K; ++t) {
        ref0 = ref + ((t - 2) % 3) * N3;
        ref1 = ref + ((t - 1) % 3) * N3;
        ref2 = ref + ((t    ) % 3) * N3;

        #pragma omp parallel for collapse(3) num_threads(NThreads)
        for (unsigned i = 0u; i <= N; ++i)
            for (unsigned j = 0u; j <= N; ++j)
                for (unsigned k = 0u; k <= N; ++k)
                    ref2[ind(i, j, k, N)] = 2 * ref1[ind(i, j, k, N)]  -  ref0[ind(i, j, k, N)] +
                                            tau2 * a2 * deltah(i, j, k, ref1, N, h2);

        #pragma omp parallel for reduction(max : maxv) num_threads(NThreads)
        for (unsigned i = 0u; i <= N; ++i)
            for (unsigned j = 0u; j <= N; ++j)
                for (unsigned k = 0u; k <= N; ++k) {
                    fptype _val = ref2[ind(i, j, k, N)] - u_an(h.x * i, h.y * j, h.z * k, t * tau, L);
                    if (_val > maxv) maxv = _val;
                }

#ifdef PRINT_U
        fprintf(f, "\nt=%d\n", t);
        printf("t=%d\r", t);
        for (unsigned i = 0u; i <= N; ++i)
            for (unsigned j = 0u; j <= N; ++j)
                for (unsigned k = 0u; k <= N; ++k)
                    fprintf(f, "%f\n", ref2[ind(i, j, k, N)]);
#endif
    }

    timespec_get(&finish, TIME_UTC);
    printf("DONE, results:\n");
    printf("Max error = %f\n", maxv);
    printf("overall = %f\n", timediff(finish, start));

    free(ref);
    return;
}


int main(void) {
#ifdef PRINT_U
    FILE *f = fopen("./u_an_1_128.txt", "w");
    for (unsigned t = 0; t < K; ++t)
        for (unsigned i = 0; i < 128; ++i)
            for (unsigned j = 0; j < 128; ++j)
                for (unsigned k = 0; k < 128; ++k)
                    fprintf(f, "%f\n", u_an((1. / 128) * i, (1. / 128) * j, (1. / 128) * k, t * tau, L1));
    fclose(f);

    f = fopen("./u_an_P_128.txt", "w");
    for (unsigned t = 0; t < K; ++t)
        for (unsigned i = 0; i < 128; ++i)
            for (unsigned j = 0; j < 128; ++j)
                for (unsigned k = 0; k < 128; ++k)
                    fprintf(f, "%f\n", u_an((MY_PI / 128) * i, (MY_PI / 128) * j, (MY_PI / 128) * k, t * tau, LP));
    fclose(f);

    f = fopen("./u_an_1_256.txt", "w");
    for (unsigned t = 0; t < K; ++t)
        for (unsigned i = 0; i < 256; ++i)
            for (unsigned j = 0; j < 256; ++j)
                for (unsigned k = 0; k < 256; ++k)
                    fprintf(f, "%f\n", u_an((1. / 256) * i, (1. / 256) * j, (1. / 256) * k, t * tau, L1));
    fclose(f);

    f = fopen("./u_an_P_256.txt", "w");
    for (unsigned t = 0; t < K; ++t)
        for (unsigned i = 0; i < 256; ++i)
            for (unsigned j = 0; j < 256; ++j)
                for (unsigned k = 0; k < 256; ++k)
                    fprintf(f, "%f\n", u_an((MY_PI / 256) * i, (MY_PI / 256) * j, (MY_PI / 256) * k, t * tau, LP));
    fclose(f);
    comp_all(1, 4, L1, N1, 16);
    comp_all(2, 4, LP, N1, 16);
    comp_all(3, 3, L1, N2, 16);
    comp_all(4, 3, LP, N2, 16);
#else

    comp_all(0, 0, L1, N1, 2);

    comp_all(1, 1, L1, N1, 2);
    comp_all(1, 2, L1, N1, 4);
    comp_all(1, 3, L1, N1, 8);
    comp_all(1, 4, L1, N1, 16);

    comp_all(2, 1, LP, N1, 2);
    comp_all(2, 2, LP, N1, 4);
    comp_all(2, 3, LP, N1, 8);
    comp_all(2, 4, LP, N1, 16);

    comp_all(3, 1, L1, N2, 4);
    comp_all(3, 2, L1, N2, 8);
    comp_all(3, 3, L1, N2, 16);
    comp_all(3, 4, L1, N2, 32);

    comp_all(4, 1, LP, N2, 4);
    comp_all(4, 2, LP, N2, 8);
    comp_all(4, 3, LP, N2, 16);
    comp_all(4, 4, LP, N2, 32);
#endif
    return 0;
}