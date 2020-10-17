# Simple VGG16 from Scratch

This is a simple implementation of the VGG16 net from scratch written in C++.

## Build & Run

```
$ make clean; make; ./vgg16
$ make clean; make BENCHMARK=1; ./vgg16
```

## Unit Test

```
$ make clean; make test; ./vgg16_test
```

## Output

```shell
$ make clean; make BENCHMARK=1; ./vgg16
rm -rf objs *~ vgg16 vgg16_test
mkdir -p objs/
clang++ main.cpp -Iobjs/ -O3 -std=c++17 -Wall -fopenmp  -DBENCHMARK -c -o objs/main.o
clang++ -Iobjs/ -O3 -std=c++17 -Wall -fopenmp  -DBENCHMARK -o vgg16 objs/main.o
The VGG16 Net
-----------------------------
NAME:   MEM     PARAM   MAC
-----------------------------
Conv:   25690112,       1792,   86704128
Conv:   25690112,       36928,  1849688064
MaxPool:        6422528,        0,      0
Conv:   12845056,       73856,  924844032
Conv:   12845056,       147584, 1849688064
MaxPool:        3211264,        0,      0
Conv:   6422528,        295168, 924844032
Conv:   6422528,        590080, 1849688064
MaxPool:        1605632,        0,      0
Conv:   3211264,        1180160,        924844032
Conv:   3211264,        2359808,        1849688064
MaxPool:        802816, 0,      0
Conv:   802816, 2359808,        462422016
Conv:   802816, 2359808,        462422016
MaxPool:        200704, 0,      0
FC:     32768,  102764544,      102760448
FC:     32768,  16781312,       16777216
FC:     8000,   4097000,        4096000
Total:  110Mb,  133M,   11308M
-----------------------------
@@ result @@
0.0413532 0.0416434 0.0412649 0.0419855 0.0412341 ... so many ... 0.0415309 0.0406799 0.0418257 0.0416706 0.0413013
```