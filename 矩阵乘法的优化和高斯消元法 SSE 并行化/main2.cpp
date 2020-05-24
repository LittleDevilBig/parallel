#include<pmmintrin.h>
#include<time.h>
#include<xmmintrin.h>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <windows.h>

using namespace std;
int N;
long long head, tail, freq; //timers

void normal_gaosi(float **matrix) //没加SSE串行的高斯消去法
{
    float d[N],x[N];
    for (int k = 0; k < N; k++){
        float tmp =matrix[k][k];
        for (int j = k; j < N; j++){
            matrix[k][j] = matrix[k][j] / tmp;
        }
        for (int i = k + 1; i < N; i++){
            float tmp2 = matrix[i][k];
            for (int j = k + 1; j < N; j++){
                matrix[i][j] = matrix[i][j] - tmp2 * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    for(int k = N - 1; k >= 0; k--){
        x[k] = d[k];
        for(int i = k + 1; i < N; i++)x[k] -= matrix[k][i] * x[i];
        x[k] /= matrix[k][k];
    }
}
void SSE_gaosi_elimination(float **matrix)
{
    float d[N],x[N];
    __m128 t1, t2, t3, t4;
    for(int k = 0; k < N; k++){
        float matrix_kk[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(matrix_kk);
        for(int j = N - 4; j >= k; j -= 4){
            t2 = _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        for(int j = k; j % 4 != (N % 4); j++)matrix[k][j] /= matrix[k][k];
        for(int i = k + 1; i < N; i++){
            float matrix_ik[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm_loadu_ps(matrix_ik);
            for(int j = N - 4; j > k; j -= 4){
                t2 = _mm_loadu_ps(matrix[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_loadu_ps(matrix[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_storeu_ps(matrix[i] + j, t4);
            }
            for(int j = k + 1; j % 4 != (N % 4); j++)matrix[i][j] -= matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
    for(int k = N - 1; k >= 0; k--){
        x[k] = d[k];
        for(int i = k + 1; i < N; i++)x[k] -= matrix[k][i] * x[i];
        x[k] /= matrix[k][k];
    }
}
void SSE_gaosi(float **matrix){
    float d[N],x[N];
    __m128 t1, t2, t3, t4;
    for (int k = 0; k < N; k++){
        float tmp[4] = { matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k] };
        t1 = _mm_loadu_ps(tmp);
        for (int j = N - 4; j >=k; j -= 4) //从后向前每次取四个
        {
            t2 = _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);//除法
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        if (k % 4 != (N % 4)) //处理不能被4整除的元素
        {
            for (int j = k; j % 4 != ( N% 4); j++){
                matrix[k][j] = matrix[k][j] / tmp[0];
            }
        }
        for (int j = (N % 4) - 1; j>= 0; j--){
            matrix[k][j] = matrix[k][j] / tmp[0];
        }
        for (int i = k + 1; i < N; i++){
            float tmp[4] = { matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k] };
            t1 = _mm_loadu_ps(tmp);
            for (int j = N - 4; j >k;j -= 4){
                t2 = _mm_loadu_ps(matrix[i] + j);
                t3 = _mm_loadu_ps(matrix[k] + j);
                t4 = _mm_sub_ps(t2,_mm_mul_ps(t1, t3)); //减法
                _mm_storeu_ps(matrix[i] + j, t4);
            }
            for (int j = k + 1; j % 4 !=(N % 4); j++){
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    for(int k = N - 1; k >= 0; k--){
        float d_k[4] = {d[k], d[k], d[k], d[k]};
        t4 = _mm_loadu_ps(d_k);
        for(int i = N - 4; i > k; i -= 4){
            t1 = _mm_loadu_ps(x + i);
            t2 = _mm_loadu_ps(matrix[k] + i);
            t3 = _mm_mul_ps(t1, t2);
            t4 = _mm_sub_ps(t4, t3);
        }
        _mm_storeu_ps(d_k, t4);
        x[k] = d_k[0];
        for(int i = k + 1; i % 4 != (N % 4); i++)x[k] -= matrix[k][i] * x[i];
        x[k] /= matrix[k][k];
    }
}

int main(){
    for(N=8;N<4100;N*=2){
        srand((unsigned)time(NULL));
        float **matrix = new float*[N];
        float **matrix2 = new float*[N];
        for (int i = 0; i<N; i++){
            matrix[i] = new float[N];
            matrix2[i] = matrix[i];
        }
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                matrix[i][j] = rand() % 100;
            }
        }
        cout <<endl<<endl<<endl<<"不使用SSE串行的高斯消去法" << endl;
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
        normal_gaosi(matrix);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout <<N<< "总共耗时： " << (tail - head) * 1000.0 / freq << "ms" << endl;
        cout <<endl<<endl<<endl<< "使用SSE消元并行的高斯消去法" << endl;
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
        SSE_gaosi_elimination(matrix2);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout <<N<< "总共耗时： " <<  (tail - head) * 1000.0 / freq<< "ms" << endl;
        cout <<endl<<endl<<endl<< "使用SSE并行的高斯消去法" << endl;
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
        SSE_gaosi(matrix2);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        cout <<N<< "总共耗时： " <<  (tail - head) * 1000.0 / freq<< "ms" << endl;
    }
    system("pause");
    return 0;
}
