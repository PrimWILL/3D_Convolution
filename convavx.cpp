#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define nprocs 32

using namespace std;

void print256_num(__m256 var);
int validation(float* output, float* check_output, int omat_x, int omat_y, int omat_z);
void conv_single();
void* conv_multi(void *arg);

int imat_x, imat_y, imat_z;
int omat_x, omat_y, omat_z;
int kernel_size;
float *input, *kernel, *output;
float *single_output;
float *multi_output;
float **multi_temp;

struct MultipleArg
{
    int id;
	int start;
	int end;
};

int main(int argc, char **argv)
{
    srand(time(NULL));
    timespec start, end;
    float diff_time;

    ifstream inputFile;
    ifstream kernelFile;
    ifstream outputFile;

    int i = 0;

    /* get input file */
    inputFile.open(argv[1], ifstream::in);

    if (inputFile.is_open() == false) {
        cout << "The "<< argv[1] << " file can not be opend" << endl;
        return 1;
    }

    inputFile >> imat_z;
    inputFile >> imat_y;
    inputFile >> imat_x;

    input = (float*)aligned_alloc(32, sizeof(float) * imat_z * imat_y * imat_x);
    i = 0;
    while (!inputFile.eof()) {
        inputFile >> input[i++];
    }
    inputFile.close();


    /* get kernel file */
    kernelFile.open(argv[2], ifstream::in);

    if (kernelFile.is_open() == false) {
        cout << "The "<< argv[2] << " file can not be opend" << endl;
        return 1;
    }

    kernelFile >> kernel_size;

    kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size * kernel_size);
    i = 0;
    while (!kernelFile.eof()) {
        kernelFile >> kernel[i++];
    }
    kernelFile.close();


    /* get output file */
    outputFile.open(argv[3], ifstream::in);

    if (outputFile.is_open() == false) {
        cout << "The "<< argv[3] << " file can not be opend" << endl;
        return 1;
    }

    outputFile >> omat_z;
    outputFile >> omat_y;
    outputFile >> omat_x;

    output = (float*)malloc(sizeof(float) * omat_z * omat_y * omat_x);
    i = 0;
    while (!outputFile.eof()) {
        outputFile >> output[i++];
    }
    outputFile.close();

    /* single-thread (AVX) */
    clock_gettime(CLOCK_REALTIME, &start);
    single_output = (float*)aligned_alloc(32, sizeof(float) * imat_z * imat_y * imat_x);
    conv_single();
    clock_gettime(CLOCK_REALTIME, &end);
    diff_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/(float)1000000000);
    printf("%.6f second(s) elapsed.\n", diff_time);

    /* validation with single-thread and output file */
    if (validation(output, single_output, omat_x, omat_y, omat_z)) printf("single-thread valid!\n");
    else printf("single-thread invalid! Please check one more time.\n");

    /* multi-thread (with AVX) */
    multi_output = (float*)aligned_alloc(32, sizeof(float) * imat_z * imat_y * imat_x);
    multi_temp = (float**)aligned_alloc(32, sizeof(float*) * kernel_size);

    for (int i = 0; i < kernel_size; i++)
    {
        multi_temp[i] = (float*)aligned_alloc(32, sizeof(float) * imat_z * imat_y * imat_x);
        memset(multi_temp[i], 0, sizeof(float)*imat_z * imat_y * imat_x);
    }

    int quotient = imat_z / nprocs;
    int remainder = imat_z % nprocs;

    int point = 0;
    MultipleArg multiple_arg[nprocs];

    for (int i = 0; i < nprocs; i++) {
      multiple_arg[i].id = i;
      multiple_arg[i].start = point;
      if (remainder > 0) {
        remainder--;
        point++;
      }
      point += quotient;
      multiple_arg[i].end = point;
    }

    pthread_t threads[nprocs];

    clock_gettime(CLOCK_REALTIME, &start);
    for (int i = 0; i < nprocs; i++) {
      pthread_create(&threads[i], NULL, conv_multi, (void*)&multiple_arg[i]);
    }

    for (int i = 0; i < nprocs; i++) {
      pthread_join(threads[i], NULL);
    }

    int idx = 0;
    __m256 temp, res;
    for (int i = 0; i < imat_z; i++) {
        for (int j = 0; j < imat_y; j++) {
            for (int k = 0; k < imat_x; k += 8) {
                idx = i * imat_x * imat_y + j * imat_x + k;
                res = _mm256_setzero_ps();
                for (int p = 0; p < kernel_size; p++) {
                    temp = _mm256_load_ps(&multi_temp[p][idx]);
                    res = _mm256_add_ps(res, temp);
                }
                memcpy(&multi_output[idx], &res, sizeof(__m256));
            }
        }
    }

    clock_gettime(CLOCK_REALTIME, &end);
    diff_time = (end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec)/(float)1000000000);
    printf("%.6f second(s) elapsed.\n", diff_time);

    /* validation with multi-thread and output file */
    if (validation(output, multi_output, omat_x, omat_y, omat_z)) printf("multi-thread valid!\n");
    else printf("multi-thread invalid! Please check one more time.\n");

    /* 동적할당한 배열들 free */
    free(input);
    free(kernel);
    free(output);
    free(multi_output);

    for (int i = 0; i < kernel_size; i++) 
        free(multi_temp[i]);
    free(multi_temp);
    return 0;
}

/* __m256 자료형을 가진 변수의 값을 출력하기 위한 print 함수 */
void print256_num(__m256 var)
{
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %f %f %f %f %f %f %f %f \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

/* single thread convolution */
void conv_single()
{
    __m256 ker_arr[kernel_size * kernel_size * kernel_size];
    __m256 result[kernel_size * kernel_size * kernel_size] = { 0, };
    float* line = new float[kernel_size * kernel_size * kernel_size * imat_x]();

    __m256 num, temp, temp2, res;
    int padding = (kernel_size / 2);
    int dist = 0, updown = 0;

    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        ker_arr[i] = _mm256_set_ps(kernel[i], kernel[i], kernel[i], kernel[i], 
                                   kernel[i], kernel[i], kernel[i], kernel[i]);
    }

    for (int i = 0; i < imat_z; i++) {              // z row
        for (int j = 0; j < imat_y; j++) {          // y row
            for (int k = 0; k < imat_x; k += 8) {   // x row
                /* input에서 x축 기준 8개의 elments를 load한다 */
                num = _mm256_load_ps(&input[i * imat_x * imat_y + j * imat_x + k]);

                /* load한 원소와 kernel의 각 원소와 연산한 후, 연산한 값을 imat_x의 크기를 가진 배열에 맞는 위치에 넣어준다 */
                for (int p = 0; p < kernel_size * kernel_size * kernel_size; p++) {
                    result[p] = _mm256_mul_ps(ker_arr[p], num);
                    memcpy(&line[p * imat_x + k], &result[p], sizeof(float) * 8);
                }
            }

            /* imat_x의 크기를 가진 배열에서 kernel의 element의 위치에 따라 왼쪽 혹은 오른쪽으로 shift 한다 */
            for (int p = (kernel_size / 2); p < kernel_size * kernel_size * kernel_size; p += kernel_size) {
                for (int t = 1; t <= padding; t++) {
                    memmove(&line[(p - t)*imat_x + t], &line[(p - t)*imat_x], sizeof(float) * (imat_x - t));
                    memmove(&line[(p + t)*imat_x], &line[(p + t)*imat_x + t], sizeof(float) * (imat_x - t));
                    for (int s = 0; s < t; s++) {
                        line[(p - t)*imat_x + s] = 0;
                        line[(p + t)*imat_x + imat_x - 1 - s] = 0;
                    }
                }
            }
            
            /* output에 값을 더한다 */
            for (int k = 0; k < imat_x; k += 8) {
                for (int p = 0; p < kernel_size * kernel_size * kernel_size; p++) {
                    /* output에 넣을 위치가 matrix 범위를 넘어가는지 계산하기 위한 변수 dist, updown */
                    dist = kernel_size / 2 - p / (kernel_size * kernel_size);
                    updown = kernel_size / 2 - (p % (kernel_size * kernel_size)) / kernel_size;

                    /* output에 넣을 위치가 matrix 범위를 넘어간다면 continue */
                    if (imat_y - j <= padding && updown > 0 && updown >= (imat_y - j)) { continue; }
                    if (i + dist < 0 || i + dist >= imat_z || j + updown < 0 || j + updown > imat_y) continue;

                    temp = _mm256_load_ps(&line[p * imat_x + k]);

                    /* 들어가야 할 위치에 kernel과 input과 연산한 값을 더하여 넣어준다 */
                    temp2 = _mm256_load_ps(&single_output[(i+dist)*imat_x*imat_y + (j+updown)*imat_x + k]);
                    res = _mm256_add_ps(temp, temp2);
                    memcpy(&single_output[(i+dist)*imat_x*imat_y + (j+updown)*imat_x + k], &res, sizeof(__m256));
                }
            }
        }
    }
}

/* multi thread convolution */
void* conv_multi(void *arg)
{
    MultipleArg *args = (MultipleArg*) arg;

    int temp_idx = args->id % kernel_size;

    __m256 ker_arr[kernel_size * kernel_size * kernel_size];
    __m256 result[kernel_size * kernel_size * kernel_size] = { 0, };
    float* line = new float[kernel_size * kernel_size * kernel_size * imat_x];

    __m256 num, temp, temp2, res;
    int padding = (kernel_size / 2);
    int dist = 0, updown = 0, idx = 0;

    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        ker_arr[i] = _mm256_set_ps(kernel[i], kernel[i], kernel[i], kernel[i], 
                                   kernel[i], kernel[i], kernel[i], kernel[i]);
    }

    for (int i = args->start; i < args->end; i++) {              // z row
        for (int j = 0; j < imat_y; j++) {          // y row
            for (int k = 0; k < imat_x; k += 8) {   // x row
                /* input에서 x축 기준 8개의 elments를 load한다 */
                num = _mm256_load_ps(&input[i * imat_x * imat_y + j * imat_x + k]);

                /* load한 원소와 kernel의 각 원소와 연산한 후, 연산한 값을 imat_x의 크기를 가진 배열에 맞는 위치에 넣어준다 */
                for (int p = 0; p < kernel_size * kernel_size * kernel_size; p++) {
                    result[p] = _mm256_mul_ps(ker_arr[p], num);
                    memcpy(&line[p * imat_x + k], &result[p], sizeof(float) * 8);
                }
            }

            /* imat_x의 크기를 가진 배열에서 kernel의 element의 위치에 따라 왼쪽 혹은 오른쪽으로 shift 한다 */
            for (int p = (kernel_size / 2); p < kernel_size * kernel_size * kernel_size; p += kernel_size) {
                for (int t = 1; t <= padding; t++) {
                    memmove(&line[(p - t)*imat_x + t], &line[(p - t)*imat_x], sizeof(float) * (imat_x - t));
                    memmove(&line[(p + t)*imat_x], &line[(p + t)*imat_x + t], sizeof(float) * (imat_x - t));
                    for (int s = 0; s < t; s++) {
                        line[(p - t)*imat_x + s] = 0;
                        line[(p + t)*imat_x + imat_x - 1 - s] = 0;
                    }
                }
            }
            
            /* output에 값을 더한다 */
            for (int k = 0; k < imat_x; k += 8) {
                for (int position = 0; position < kernel_size * kernel_size * kernel_size; position++) {
                    /* output에 넣을 위치가 matrix 범위를 넘어가는지 계산하기 위한 변수 dist, updown */
                    dist = kernel_size / 2 - position / (kernel_size * kernel_size);
                    updown = kernel_size / 2 - (position % (kernel_size * kernel_size)) / kernel_size;

                    /* output에 넣을 위치가 matrix 범위를 넘어간다면 continue */
                    if (imat_y - j <= padding && updown > 0 && updown >= (imat_y - j)) { continue; }
                    if (i + dist < 0 || i + dist >= imat_z || j + updown < 0 || j + updown > imat_y) continue;

                    /* 들어가야 할 위치에 kernel과 input과 연산한 값을 더하여 넣어준다 */
                    idx = (i+dist)*imat_x*imat_y + (j+updown)*imat_x + k;
                    temp = _mm256_load_ps(&line[position * imat_x + k]);
                    temp2 = _mm256_load_ps(&multi_temp[temp_idx][idx]);
                    res = _mm256_add_ps(temp, temp2);
                    memcpy(&multi_temp[temp_idx][idx], &res, sizeof(__m256));
                }
            }
            usleep(1); 
        }
    }
}

int validation(float* output, float* check_output, int omat_x, int omat_y, int omat_z)
{
    /* check the validation */ 
    for (int i = 0; i < omat_x * omat_y * omat_z; i++) {
        if (abs(output[i] - check_output[i]) >= 0.001f) {
            return 0;
        }
    }
    return 1;
}