#pragma once
#include "Util.hpp"
#include "Tensor.hpp"
#include "Operand.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

#ifdef BENCHMARK
extern long long int MEM, PARAM, MAC, MEMALL, PARAMALL, MACALL;
#endif

namespace sv
{
    template <typename dtype>
    class Layer
    {
    private:
        std::string name;

    public:
        virtual ~Layer(){};

        virtual void print() const = 0;
#ifdef BENCHMARK
        virtual void printBenchmark() const = 0;
#endif

        virtual std::string getName() const = 0;
        virtual void forward(sv::Tensor<dtype> &input, sv::Tensor<dtype> &out) const = 0;
    };

    template <typename dtype>
    class ConvLayer : public Layer<dtype>
    {
    private:
        std::string name = "Conv";
        int fmapSize;
        int kernelSize;
        int channelSize;
        sv::Tensor<dtype> weight;
        sv::Tensor<dtype> bias;

    public:
        ConvLayer(int fmapSize, int kernelSize, int channelSize)
            : fmapSize{fmapSize}, kernelSize{kernelSize}, channelSize{channelSize}
        {
            STATIC_ASSERT_FLOAT_TYPE(dtype);
            weight = sv::Tensor<dtype>(kernelSize, kernelSize, fmapSize, channelSize);
            weight.randam();
            bias = sv::Tensor<dtype>(channelSize);
            bias.randam();
        }
        ~ConvLayer() = default;

        void print() const override
        {
            std::cout << "*weight*" << std::endl;
            std::cout << "shape:" << this->weight.shapeStr() << std::endl;
            std::cout << this->weight << std::endl;
            std::cout << "*Bias*" << std::endl;
            std::cout << "shape:" << this->bias.shapeStr() << std::endl;
            std::cout << this->bias << std::endl;
        }

#ifdef BENCHMARK
        virtual void printBenchmark() const override
        {
            std::cout << name << ":\t" << MEM << ",\t" << PARAM << ",\t" << MAC << std::endl;
            MEMALL += MEM;
            PARAMALL += PARAM;
            MACALL += MAC;
        };
#endif

        std::string getName() const override
        {
            return name;
        };

        void forward(sv::Tensor<dtype> &input, sv::Tensor<dtype> &out) const override
        {
            auto inputShape = input.shape();
            int x = inputShape[0], y = inputShape[1], z = inputShape[2];

            sv::Tensor<dtype> newTensor(x + 2, y + 2, z);
            auto newShape = newTensor.shape();
            int width = newShape[0], height = newShape[1];

            for (int i = 0; i < z; i++)
            {
                for (int k = 0; k < y; k++)
                {
                    for (int j = 0; j < x; j++)
                    {

                        int newId = sv::to1D(i, k + 1, j + 1, width, height);
                        int oldId = sv::to1D(i, k, j, x, y);
                        newTensor[newId] = input[oldId];
                    }
                }
            }

            sv::conv2d<dtype>(newTensor, out, weight, bias);
#ifdef BENCHMARK
            printBenchmark();
#endif
        }
    };

    template <typename dtype>
    class MaxPoolLayer : public Layer<dtype>
    {
    private:
        std::string name = "MaxPool";
        int poolSize;
        int stride;

    public:
        MaxPoolLayer(int poolSize, int stride)
            : poolSize{poolSize}, stride{stride}
        {
            STATIC_ASSERT_FLOAT_TYPE(dtype);
        }
        ~MaxPoolLayer() = default;

        void print() const override
        {
            std::cout << "poolSize: " << poolSize << std::endl;
        }

#ifdef BENCHMARK
        virtual void printBenchmark() const override
        {
            std::cout << name << ":\t" << MEM << ",\t" << PARAM << ",\t" << MAC << std::endl;
            MEMALL += MEM;
            PARAMALL += PARAM;
            MACALL += MAC;
        };
#endif

        std::string getName() const override
        {
            return name;
        };

        void forward(sv::Tensor<dtype> &input, sv::Tensor<dtype> &out) const override
        {
            sv::maxpool<dtype>(input, out, poolSize, stride);
#ifdef BENCHMARK
            printBenchmark();
#endif
        }
    };

    template <typename dtype>
    class FCLayer : public Layer<dtype>
    {
    private:
        std::string name = "FC";
        sv::Tensor<dtype> weight;
        sv::Tensor<dtype> bias;
        int inputSize;
        int outputSize;

    public:
        FCLayer(int inputSize, int outputSize)
            : inputSize{inputSize}, outputSize{outputSize}
        {
            STATIC_ASSERT_FLOAT_TYPE(dtype);
            weight = sv::Tensor<dtype>(inputSize, outputSize);
            weight.randam();
            bias = sv::Tensor<dtype>(outputSize);
            bias.randam();
        }
        ~FCLayer() = default;

        void print() const override
        {
            std::cout << "*weight*" << std::endl;
            std::cout << "shape:" << this->weight.shapeStr() << std::endl;
            std::cout << this->weight << std::endl;
            std::cout << "*Bias*" << std::endl;
            std::cout << "shape:" << this->bias.shapeStr() << std::endl;
            std::cout << this->bias << std::endl;
        }

#ifdef BENCHMARK
        virtual void printBenchmark() const override
        {
            std::cout << name << ":\t" << MEM << ",\t" << PARAM << ",\t" << MAC << std::endl;
            MEMALL += MEM;
            PARAMALL += PARAM;
            MACALL += MAC;
        };
#endif

        std::string getName() const override
        {
            return name;
        };

        void forward(sv::Tensor<dtype> &input, sv::Tensor<dtype> &out) const override
        {
            sv::fc<dtype>(input, out, weight, bias);
#ifdef BENCHMARK
            printBenchmark();
#endif
        }
    };
} // namespace sv