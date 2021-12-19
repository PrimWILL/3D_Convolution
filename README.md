# 3D Convoltuion using AVX and CUDA
AVX연산과 multi-threading을 사용하여 3D convolution을 수행하는 convavx.cpp와,
GPU연산을 사용하여 3D convolution을 수행하는 convgpu.cu 파일이다.

## 🔨 How To Build the Server with Makefile
[🇰🇷]  
Makefile을 사용했기 때문에 편하게 다음 명령어 하나로 convgpu파일과 convavx파일을 build할 수 있다.  

```$ make```

이를 통해 생성된 파일들로 3D convolution을 할 수 있다.
 
[Eng]  
Because of using Makefile, you can build convgpu.cu and convavx.cpp easily by entering command ```$ make```
<br/>

---

## ⚙ How To Execute testone
[🇰🇷]  
3D convolution을 실행하기 위한 명령어는 다음과 같다.  

1. AVX 연산을 test하고 싶은 경우
```$ ./convavx <test folder>/input.txt <test folder>/kernel.txt <test folder>/output.txt```  
ex) test1 폴더의 file로 3D convolution 연산을 하고 싶다면 ``` $ ./convavx test1/input.txt test1/kernel.txt test1/output.txt ``` 이라고 입력해야 한다.

2. GPU 연산을 test하고 싶은 경우  
```$ ./convgpu <test folder>/input.txt <test folder>/kernel.txt <test folder>/output.txt```  
ex) test1 폴더의 file로 3D convolution 연산을 하고 싶다면 ``` $ ./convgpu test1/input.txt test1/kernel.txt test1/output.txt ``` 이라고 입력해야 한다.

[Eng]  
You can test using 2 options (AVX with multi-threading, GPU(CUDA))   

1. AVX operation  
```$ ./convavx <test folder>/input.txt <test folder>/kernel.txt <test folder>/output.txt```  

2. GPU operation  
```$ ./convgpu <test folder>/input.txt <test folder>/kernel.txt <test folder>/output.txt```
