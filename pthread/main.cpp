#include <iostream>
#include <algorithm>
#include <vector>
#include <time.h>
#include <immintrin.h>
#include <windows.h>
#include <pthread.h>
using namespace std;
/*
typedef struct{
int threadId;
} threadParm_t;
const int ARR_NUM = 10000;
const int ARR_LEN = 10000;
const int THREAD_NUM = 4;
const int seg = ARR_NUM / 8;
vector<int> arr[ARR_NUM];
pthread_mutex_t mutex;
long long head, freq; // timers
int next_arr = 0;
pthread_mutex_t mutex_task;
void init(void)
{
srand(unsigned(time(nullptr)));
for (int i = 0; i < ARR_NUM; i++) {
arr[i].resize(ARR_LEN);
for (int j = 0; j < ARR_LEN; j++)
arr[i][j] = rand();
}
}
void *arr_sort(void *parm)
{
threadParm_t *p = (threadParm_t *) parm;
int r = p->threadId;
int task = 0;
long long tail;
while (1) {
pthread_mutex_lock(&mutex_task);
task = next_arr++;
pthread_mutex_unlock(&mutex_task);
if (task >= ARR_NUM) break;
stable_sort(arr[task].begin(), arr[task].end());
}
pthread_mutex_lock(&mutex);
QueryPerformanceCounter((LARGE_INTEGER *)&tail);
printf("Thread %d: %lfms.\n", r, (tail - head) * 1000.0 / freq);
pthread_mutex_unlock(&mutex);
pthread_exit(nullptr);
}
int main(int argc, char *argv[])
{
QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
init();
mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t thread[THREAD_NUM];
threadParm_t threadParm[THREAD_NUM];
QueryPerformanceCounter((LARGE_INTEGER *)&head);
for (int i = 0; i < THREAD_NUM; i++)
{
threadParm[i].threadId = i;
pthread_create(&thread[i], nullptr, arr_sort, (void *)&threadParm[i]);
}
for (int i = 0; i < THREAD_NUM; i++)
{
pthread_join(thread[i], nullptr);
}
pthread_mutex_destroy(&mutex);
}
*/

const int maxN = 3000;
const int testRound = 10;

int n;
float a[maxN][maxN], b[maxN][maxN], c[maxN][maxN];
float d[maxN], x[maxN];

int THREAD_NUM = 4;
int next_arr;
int seg;
int glo_k;

pthread_mutex_t mutex;
pthread_barrier_t barrier;

long long head, tail, freq; // timers

typedef struct{
    int threadId;
    int st;
    int ed;
}threadParm_t;


void init(int u)
{
    srand(time(NULL));
    for(int i = 0; i < u; i++){
        for(int j = 0; j < u; j++){
            a[i][j] = rand() + 1;
            b[i][j] = rand() + 1;
        }
        d[i] = rand() + 1;
    }
}

void *gause_sse(void *parm)
{
    threadParm_t *p = (threadParm_t *) parm;
    int r = p->threadId;
    int task = p->st;
    int last = p->ed;
    int k = task - 1;
    while (1)
    {
        pthread_mutex_lock(&mutex);
        task = next_arr++;
        pthread_mutex_unlock(&mutex);
        if (task >= last){
            break;
        }
        for(int j = k + 1; j < n; j++)a[task][j] -= a[task][k] * a[k][j];
        a[task][k] = 0;
    }

    return NULL;
}

void *gause_sse_block(void *parm)
{
    threadParm_t *p = (threadParm_t *) parm;
    int r = p->threadId;
    if(r<n-1)for(int h=r*seg;h<(r+1)*seg;h++){
        for(int j = glo_k + 1; j < n; j++)a[h][j] -= a[h][glo_k] * a[glo_k][j];
        a[h][glo_k] = 0;
    }
    else{
        for(int h=r*seg;h<n;h++){
            for(int j = glo_k + 1; j < n; j++)a[h][j] -= a[h][glo_k] * a[glo_k][j];
            a[h][glo_k] = 0;
        }
    }
    return NULL;
}

void gauss_sse_elimination_back_pthread(int thread_num)
{
    THREAD_NUM = thread_num;

    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < n; k++){
        float a_kk[4] = {a[k][k], a[k][k], a[k][k], a[k][k]};
        t1 = _mm_loadu_ps(a_kk);
        for(int j = n - 4; j >= k; j -= 4){
            t2 = _mm_loadu_ps(a[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(a[k] + j, t3);
        }
        for(int j = k; j % 4 != (n % 4); j++)a[k][j] /= a[k][k];
        next_arr = k + 1;
        for(int i = 0; i < THREAD_NUM; i++){
            threadParm[i].threadId = i;
            threadParm[i].st = next_arr;
            threadParm[i].ed = n;
            pthread_create(&thread[i], NULL, gause_sse, (void*)&threadParm[i]);
        }
        for(int i = 0; i < THREAD_NUM; i++)pthread_join(thread[i], NULL);

    }
    for(int k = n - 1; k >= 0; k--){
        float d_k[4] = {d[k], d[k], d[k], d[k]};
        t4 = _mm_loadu_ps(d_k);
        for(int i = n - 4; i > k; i -= 4){
            t1 = _mm_loadu_ps(x + i);
            t2 = _mm_loadu_ps(a[k] + i);
            t3 = _mm_mul_ps(t1, t2);
            t4 = _mm_sub_ps(t4, t3);
        }
        _mm_storeu_ps(d_k, t4);
        x[k] = d_k[0];
        for(int i = k + 1; i % 4 != (n % 4); i++)x[k] -= a[k][i] * x[i];
        x[k] /= a[k][k];
    }

}

void gauss_sse_elimination_back_pthread_block(int thread_num)
{
    THREAD_NUM = thread_num;

    pthread_t thread[THREAD_NUM];
    threadParm_t threadParm[THREAD_NUM];
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < n; k++){
        float a_kk[4] = {a[k][k], a[k][k], a[k][k], a[k][k]};
        t1 = _mm_loadu_ps(a_kk);
        for(int j = n - 4; j >= k; j -= 4){
            t2 = _mm_loadu_ps(a[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(a[k] + j, t3);
        }
        for(int j = k; j % 4 != (n % 4); j++)a[k][j] /= a[k][k];
        seg = (n - k - 1) / THREAD_NUM;
        glo_k = k;
        for(int i = 0; i < THREAD_NUM; i++){
            threadParm[i].threadId = i;
            pthread_create(&thread[i], NULL, gause_sse_block, (void*)&threadParm[i]);
        }
        for(int i = 0; i < THREAD_NUM; i++)pthread_join(thread[i], NULL);

    }
    for(int k = n - 1; k >= 0; k--){
        float d_k[4] = {d[k], d[k], d[k], d[k]};
        t4 = _mm_loadu_ps(d_k);
        for(int i = n - 4; i > k; i -= 4){
            t1 = _mm_loadu_ps(x + i);
            t2 = _mm_loadu_ps(a[k] + i);
            t3 = _mm_mul_ps(t1, t2);
            t4 = _mm_sub_ps(t4, t3);
        }
        _mm_storeu_ps(d_k, t4);
        x[k] = d_k[0];
        for(int i = k + 1; i % 4 != (n % 4); i++)x[k] -= a[k][i] * x[i];
        x[k] /= a[k][k];
    }

}

void SSE_gaosi(int n, float a[][maxN], float d[maxN], float x[maxN])
{
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < n; k++){
        float a_kk[4] = {a[k][k], a[k][k], a[k][k], a[k][k]};
        t1 = _mm_loadu_ps(a_kk);
        for(int j = n - 4; j >= k; j -= 4){
            t2 = _mm_loadu_ps(a[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(a[k] + j, t3);
        }
        for(int j = k; j % 4 != (n % 4); j++)a[k][j] /= a[k][k];
        for(int i = k + 1; i < n; i++){
            float a_ik[4] = {a[i][k], a[i][k], a[i][k], a[i][k]};
            t1 = _mm_loadu_ps(a_ik);
            for(int j = n - 4; j > k; j -= 4){
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
    for(int k = n - 1; k >= 0; k--){
        float d_k[4] = {d[k], d[k], d[k], d[k]};
        t4 = _mm_loadu_ps(d_k);
        for(int i = n - 4; i > k; i -= 4){
            t1 = _mm_loadu_ps(x + i);
            t2 = _mm_loadu_ps(a[k] + i);
            t3 = _mm_mul_ps(t1, t2);
            t4 = _mm_sub_ps(t4, t3);
        }
        _mm_storeu_ps(d_k, t4);
        x[k] = d_k[0];
        for(int i = k + 1; i % 4 != (n % 4); i++)x[k] -= a[k][i] * x[i];
        x[k] /= a[k][k];
    }
}


void gauss_pthread(int thread_num = 4)
{
    n = maxN;
    init(n);
    mutex = PTHREAD_MUTEX_INITIALIZER;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j = 0; j < testRound; j++){
        gauss_sse_elimination_back_pthread(thread_num);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << n << ":\t" << (tail - head) * 1000.0 / freq << "\tms" << endl;
    pthread_mutex_destroy(&mutex);
}

void pthread_block(int thread_num = 4)
{
    n = maxN;
    init(n);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j = 0; j < testRound; j++){
        gauss_sse_elimination_back_pthread_block(thread_num);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << n << ":\t" << (tail - head) * 1000.0 / freq << "\tms" << endl;
}

void gauss(void (*func)(int n, float a[][maxN], float d[maxN], float x[maxN]))
{
    int r = maxN;
    init(r);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int j = 0; j < testRound; j++){
        func(r, a, d, x);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << r << ":\t" << (tail - head) * 1000.0 / freq << "\tms" << endl;
}


int main()
{
    gauss(SSE_gaosi);
    pthread_block(10);
    gauss_pthread(10);
    return 0;
}
