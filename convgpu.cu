#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string.h>
using namespace std;

#define KERNEL_SIZE 5   // 크게 설정해두고 입력 kernel size만큼 제한해서 사용
#define TILE_SIZE 4     // 한 block 당 최대 thread수는 1024. TILE_SIZE가 8이상이면 BLOCK_SIZE^3>=1024이 된다.
#define BLOCK_SIZE (TILE_SIZE + (KERNEL_SIZE-1))


/* validation with results and output file */
int validation(const float *result, const float *output, int Rows, int Columns, int Depth, int kernel_size)
{
    /* check the validation */ 

   bool equal = true;

   for (int i = 0; i < Depth * Rows * Columns && equal; i++) {
      if (abs(result[i] - output[i]) >= 0.001f) {
         equal = false;
            break;
      }
   }
   if (equal) {
      printf("Results are equal!\n");
        return true;
   }
   else {
      printf("Results are NOT equal!\n");
        return false;
   }
}

/* print matrix one depth */
void print_matrix(float* matrix, int width, int height, int depth){
    printf("width*height = %d\n", width*height);
    for(int i = 0; i<width*height; i++){
        printf("%f ", matrix[i]);
        if((i+1)%width == 0) printf("\n");
        if((i+1)%(width*height) == 0) printf("\n");
    }
}



__constant__ float Mc[KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE]; // kernel size는 입력 kernel size중 최대인 5로 잡았으며, 코드 내에서 입력 kernel size로 제한해서 사용

__global__ void convolution(float *image_d, float *kernel_d, float *output_d, int imat_x, int imat_y, int imat_z, int kernel_size){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_o = blockIdx.y * TILE_SIZE + ty;    
    int col_o = blockIdx.x * TILE_SIZE + tx;    
    int depth_o = blockIdx.z * TILE_SIZE + tz;

    int row_i = row_o - (kernel_size-1)/2;      // 최대 KERNEL_SIZE말고 입력받은 kernel의 size                      
    int col_i = col_o - (kernel_size-1)/2;
    int depth_i = depth_o - (kernel_size-1)/2;

    float output = 0.0f;

    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];    // tile -> size는 그대로

    if((depth_i>=0) && (depth_i < imat_z) && (row_i >= 0) && (row_i < imat_y) && (col_i >= 0) && (col_i < imat_x))
        Ns[tz][ty][tx] = image_d[(depth_i*imat_x*imat_y)+row_i*imat_x+col_i];
    else 
        Ns[tz][ty][tx] = 0.0f;

    __syncthreads();

    if(tz < TILE_SIZE && ty < TILE_SIZE && tx < TILE_SIZE){
        for(int k = 0; k<kernel_size; k++)
            for(int i = 0; i<kernel_size; i++)
                for(int j = 0; j<kernel_size; j++)
                    output += Mc[k*kernel_size*kernel_size+i*kernel_size+j]*Ns[k+tz][i+ty][j+tx];

        if(depth_o < imat_z && row_o < imat_y && col_o < imat_x)
            output_d[depth_o*imat_x*imat_y + row_o*imat_x+col_o] = output;
            
    }
    __syncthreads();

}

void GPU(int imat_x, int imat_y, int imat_z, float* input, int kernel_size, float* kernel, float* output){

    float *image_h, *image_d;
    float *kernel_h, *kernel_d;
    float *output_h, *output_d;

    image_h = input;
    kernel_h = kernel;
    output_h = (float*)malloc(sizeof(float)*imat_x*imat_y*imat_z);

    cudaError_t err1 = cudaMalloc((void**)&image_d, sizeof(float)*imat_x*imat_y*imat_z);
    cudaError_t err2 = cudaMalloc((void**)&kernel_d, sizeof(float)*kernel_size*kernel_size*kernel_size);
    cudaError_t err3 = cudaMalloc((void**)&output_d, sizeof(float)*imat_x*imat_y*imat_z);

    cudaError_t err4 = cudaMemcpy(image_d, image_h, sizeof(float)*imat_x*imat_y*imat_z, cudaMemcpyHostToDevice);
    cudaError_t err5 = cudaMemcpy(kernel_d, kernel_h, sizeof(float)*kernel_size*kernel_size*kernel_size, cudaMemcpyHostToDevice);
    cudaError_t err6 = cudaMemcpy(output_d, output_h, sizeof(float)*imat_x*imat_y*imat_z, cudaMemcpyHostToDevice);

    cudaError_t err7 = cudaMemcpyToSymbol(Mc, kernel, sizeof(float)*KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE); 
    

    dim3 dimGrid(ceil((float)imat_x/TILE_SIZE), ceil((float)imat_y/TILE_SIZE), ceil((float)imat_z/TILE_SIZE));  // device 함수에서는 TILE_SIZE단위로 접근하기에 여기에서 BLOCK_SIZE로 접근하면 처리되지 못하는 element 발생
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    convolution<<<dimGrid, dimBlock>>>(image_d, kernel_d, output_d, imat_x, imat_y, imat_z, kernel_size);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        return;
    }

    cudaMemcpy(output_h, output_d, sizeof(float)*imat_x*imat_y*imat_z, cudaMemcpyDeviceToHost);
    //print_matrix(output_h, imat_z, imat_y, imat_x);

    /* validation with GPU and output file */
    //validation(output_h, output, imat_x, imat_y, imat_z, kernel_size);
    if (validation(output_h, output, imat_x, imat_y, imat_z, kernel_size)) printf("GPU programming valid!\n");
    else printf("GPU programming invalid! Please check one more time.\n");


    cudaFree(image_d);
    cudaFree(kernel_d);
    cudaFree(output_d);
    free(output_h);
}


int main(int argc, char **argv)
{
    ifstream inputFile;
    ifstream kernelFile;
    ifstream outputFile;

    int imat_x, imat_y, imat_z;
    int omat_x, omat_y, omat_z;
    int kernel_size;
    float *input, *kernel, *output;
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

    input = (float*)malloc(sizeof(float) * imat_z * imat_y * imat_x);
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

    /* multi-thread (with AVX) */

    /* GPU */ 
    GPU(imat_x, imat_y, imat_z, input, kernel_size, kernel, output);


    /* validation with single-thread and output file */
    //if (validation()) printf("single-thread valid!\n");
    //else printf("single-thread invalid! Please check one more time.\n");

    /* validation with multi-thread and output file */
    //if () printf("multi-thread valid!\n");
    //else printf("multi-thread invalid! Please check one more time.\n");

    /* validation with GPU and output file */
    //if (validation(result, output, imat_x, imat_y, imat_z, kernel_size)) printf("GPU programming valid!\n");
    //else printf("GPU programming invalid! Please check one more time.\n");

    free(input);
    free(kernel);
    free(output);

    return 0;
}