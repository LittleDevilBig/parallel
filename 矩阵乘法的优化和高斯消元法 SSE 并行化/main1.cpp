#include <iostream>
#include <stdio.h>
#include <pmmintrin.h>
#include <stdlib.h>
#include <algorithm>
#include <windows.h>

using namespace std;
const int maxN =1024; // magnitude of matrix
const int T = 64; // tile size
int n;
float a[maxN][maxN];
float b[maxN][maxN];
float c[maxN][maxN];
long long head, tail, freq; //timers

void mul() {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void trans_mul(){
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void sse_mul(){
    __m128 t1, t2, sum;
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            c[i][j] = 0.0;
            sum = _mm_setzero_ps();
            for (int k = n - 4; k >= 0; k -= 4){ // sum every 4 elements
                t1 = _mm_loadu_ps(a[i] + k);
                t2 = _mm_loadu_ps(b[j] + k);
                t1 = _mm_mul_ps(t1, t2);
                sum = _mm_add_ps(sum, t1);
            }
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            _mm_store_ss(c[i] + j, sum);
            for (int k = (n % 4) - 1; k >= 0; --k){
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void sse_tile()
{
    __m128 t1, t2, sum;
    float t;
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
    for (int r = 0; r < n / T; ++r) for (int q = 0; q < n / T; ++q) {
        for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) c[r * T + i][q * T +j] = 0.0;
        for (int p = 0; p < n / T; ++p) {
            for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) {
                sum = _mm_setzero_ps();
                for (int k = 0; k < T; k += 4){ //sum every 4th elements
                    t1 = _mm_loadu_ps(a[r * T + i] + p * T + k);
                    t2 = _mm_loadu_ps(b[q * T + j] + p * T + k);
                    t1 = _mm_mul_ps(t1, t2);
                    sum = _mm_add_ps(sum, t1);
                }
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                _mm_store_ss(&t, sum);
                c[r * T + i][q * T + j] += t;
            }
        }
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

int main()
{
    for(n=10;n<1010;n+=90){
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                a[i][j]=rand();
                b[i][j]=rand();
            }
        }
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_tile();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout <<n<<" "<< (tail - head) * 1000.0 / freq << "ms" << endl;
    }
}
