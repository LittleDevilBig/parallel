#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <pthread.h>
pthread_mutex_t mutex;
typedef struct{
    int threadId;
    double a, b;
    int n;
    int thread_count;
    double *global_result;
} threadParm_t;
double f(double x){
    return cos(x);
}
void Trap( double a, double b, int n, double* global_result_p ) {
double h, x, my_result ;
double local_a , local_b ;
int i, local_n ;
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
h = (b-a)/n;
local_n = n/thread_count ;
local_a = a + my_rank*local_n*h;
local_b = local_a + local_n*h;
my_result = (f(local_a) + f(local_b))/2.0;
for(i = 1; i <= local_n; i++) {
x = local_a + i*h;
my_result += f(x);
}
my_result = my_result*h;
# pragma omp critical
*global_result_p += my_result ;
}
void* cal(void *parm){
    double h, x, local_a, local_b, my_result;
    int r, local_n, i;
    threadParm_t *p = (threadParm_t*)parm;
    r = p->threadId;
    h = (p->b - p->a) / p->n;
    local_n = p->n / p->thread_count;
    local_a = p->a + r * local_n * h;
    local_b = local_a +  local_n * h;
    my_result = (f(local_a) + f(local_b)) / 2.0;
    for(i = 1; i <= local_n; i++)
        my_result += f(local_a + i * h);
    my_result *= h;
    pthread_mutex_lock(&mutex);
    *p->global_result += my_result;
    pthread_mutex_unlock(&mutex);
    return NULL;
}
void pthread(double a, double b, int n, int thread_count){
    printf("Testing Pthread.\n");
    mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t thread[thread_count];
    threadParm_t threadParm[thread_count];
    double ans = 0;
    for(int i = 0; i < thread_count; i++){
        threadParm[i].threadId = i;
        threadParm[i].a = a;
        threadParm[i].b = b;
        threadParm[i].n = n;
        threadParm[i].global_result = &ans;
        threadParm[i].thread_count = thread_count;
        pthread_create(&thread[i], NULL, cal, (void*)&threadParm[i]);
    }
    for(int i = 0; i < thread_count; i++)
        pthread_join(thread[i], NULL);
    pthread_mutex_destroy(&mutex);
    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.14e\n", a, b, ans);
}
int main(int argc, char* argv[]) {
double global_result = 0.0;
double a, b;
int n;
int thread_count ;
thread_count = strtol(argv[1], NULL, 10) ;
printf("Enter a, b, and n\n");
scanf("%lf %lf %d", &a, &b, &n);
# pragma omp parallel num_threads(thread_count)
Trap(a, b, n, &global_result);
printf("With n = %d trapezoids, our estimate\n", n);
printf("of the integral from %f to %f = %.14e\n", a, b, global_result);
pthread(a, b, n, thread_count);
return 0;
}
