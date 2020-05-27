#include <iostream>
#include <pmmintrin.h>
#include <cstdio>
#include <algorithm>
#include <windows.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <omp.h>
#include <math.h>

using namespace std;

const int maxN = 500;
const int testRound = 10;

int n;
float a[maxN][maxN], b[maxN][maxN], c[maxN][maxN];
float d[maxN], x[maxN];

long long head, tail, freq; // timers

void init(int u)
{
    srand(time(NULL));
    for(int i = 0; i < u; i++)
    {
        for(int j = 0; j < u; j++)
        {
            a[i][j] = rand() + 1;
            b[i][j] = rand() + 1;
        }
        d[i] = rand() + 1;
    }
}

void gauss_sse_omp(int thread_num)
{
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < n; k++)
    {
        float a_kk[4] = {a[k][k], a[k][k], a[k][k], a[k][k]};
        t1 = _mm_loadu_ps(a_kk);
        for(int j = n - 4; j >= k; j -= 4)
        {
            t2 = _mm_loadu_ps(a[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(a[k] + j, t3);
        }
        for(int j = k; j % 4 != (n % 4); j++)a[k][j] /= a[k][k];

        #pragma omp parallel for num_threads(thread_num) private(t1, t2, t3, t4)
        for(int i = k + 1; i < n; i++)
        {
            float a_ik[4] = {a[i][k], a[i][k], a[i][k], a[i][k]};
            t1 = _mm_loadu_ps(a_ik);
            for(int j = n - 4; j > k; j -= 4)
            {
                t2 = _mm_loadu_ps(a[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_loadu_ps(a[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_storeu_ps(a[i] + j, t4);
            }
            for(int j = k + 1; j % 4 != (n % 4); j++)a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}

void gauss_omp(int thread_num = 4)
{
    n = maxN;
    init(n);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j = 0; j < testRound; j++)
    {
        gauss_sse_omp(thread_num);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << n << ":\t" << (tail - head) * 1000.0 / freq << "\tms" << endl;
}

void gauss_sse(int n, float a[][maxN], float d[maxN], float x[maxN])
{
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < n; k++)
    {
        float a_kk[4] = {a[k][k], a[k][k], a[k][k], a[k][k]};
        t1 = _mm_loadu_ps(a_kk);
        for(int j = n - 4; j >= k; j -= 4)
        {
            t2 = _mm_loadu_ps(a[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(a[k] + j, t3);
        }
        for(int j = k; j % 4 != (n % 4); j++)a[k][j] /= a[k][k];
        for(int i = k + 1; i < n; i++)
        {
            float a_ik[4] = {a[i][k], a[i][k], a[i][k], a[i][k]};
            t1 = _mm_loadu_ps(a_ik);
            for(int j = n - 4; j > k; j -= 4)
            {
                t2 = _mm_loadu_ps(a[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_loadu_ps(a[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_storeu_ps(a[i] + j, t4);
            }
            for(int j = k + 1; j % 4 != (n % 4); j++)a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}

void gauss(void (*func)(int n, float a[][maxN], float d[maxN], float x[maxN]))
{
    int r = maxN;
    init(r);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j = 0; j < testRound; j++)
    {
        func(r, a, d, x);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << r << ":\t" << (tail - head) * 1000.0 / freq << "\tms" << endl;
}

int main()
{
    gauss(gauss_sse);
    gauss_omp(10);
    return 0;
}
