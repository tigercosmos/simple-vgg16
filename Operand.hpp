#pragma once
#include <cassert>
#include "Activation.hpp"
#include <omp.h>
#include <limits>

#ifdef BENCHMARK
extern long long int MEM, PARAM, MAC;
#endif

namespace sv
{
    template <typename dtype>
    void conv2d(sv::Tensor<dtype> &input, sv::Tensor<dtype> &output,
                sv::Tensor<dtype> const &weight, sv::Tensor<dtype> const &bias)
    {
#ifdef BENCHMARK
        MEM = 0;
        PARAM = 0;
        MAC = 0;
#endif
        auto inputShape = input.shape();
        int i_w = inputShape[0];
        int i_h = inputShape[1];
        int i_f = inputShape[2]; // fmap
        auto weightShape = weight.shape();
        int w_w = weightShape[0];
        int w_h = weightShape[1];
        int w_f = weightShape[2]; // fmap
        int w_c = weightShape[3]; // channel

        int o_w = i_w - w_w + 1;
        int o_h = i_h - w_h + 1;
        int o_c = w_c;
        output = sv::Tensor<dtype>(o_w, o_h, o_c);
        auto outputShape = output.shape();

        assert(outputShape.size() == 3);
        assert(weightShape.size() == 4);
        assert(i_f == w_f);
        assert(o_c == bias.shape()[0]);

        int n, m, x, y, i, j;

#pragma omp parallel for collapse(3)
        for (n = 0; n < o_c; n++) // output channel
        {
            for (y = 0; y < o_h; y++) // output y
            {
                for (x = 0; x < o_w; x++) // output x
                {
                    dtype sum = 0;
#pragma omp parallel for collapse(3)
                    for (m = 0; m < w_f; m++) // kernel fmap
                    {
                        for (j = 0; j < w_h; j++) // kernel y
                        {
                            for (i = 0; i < w_w; i++) // kernel x
                            {
                                dtype inputWeight = input[sv::to1D(m, (y + j), (x + i), i_w, i_h)];
                                dtype kernelWeight = weight[sv::to1D(n, m, j, i, w_w, w_h, w_f)];
#pragma omp atomic
                                sum += inputWeight * kernelWeight;
                            }
                        }
                    }
#pragma omp critical
                    {
                        sum += bias[n];
                        output[sv::to1D(n, y, x, o_w, o_h)] += sv::ReLU(sum);
                    }
                }
            }
        }

#ifdef BENCHMARK
        MAC = w_f * w_h * w_w * o_c * o_w * o_h;
        PARAM = weight.data().size() + bias.data().size();
        MEM = output.data().size() * sizeof(dtype);
#endif
    }

    template <typename dtype>
    void maxpool(sv::Tensor<dtype> &input, sv::Tensor<dtype> &output, int const &poolSize, int const &stride)
    {

#ifdef BENCHMARK
        MEM = 0;
        PARAM = 0;
        MAC = 0;
#endif
        auto inputShape = input.shape();
        assert(inputShape.size() == 3);

        int width = inputShape[1];
        int height = inputShape[0];
        int outSize = (height - poolSize) / stride + 1;
        sv::Tensor<dtype> out(outSize, outSize, inputShape[2]);

        int n, x, y, i, j, idx = 0;

#pragma omp parallel for collapse(3)
        for (n = 0; n < inputShape[2]; n++)
        {
            for (y = 0; y < height; y += stride)
            {
                for (x = 0; x < width; x += stride)
                {
                    dtype max = std::numeric_limits<dtype>::min();
                    for (i = 0; i < poolSize; i++)
                    {
                        for (j = 0; j < poolSize; j++)
                        {
                            if (input[sv::to1D(n, (y + j), (x + i), width, height)] > max)
                            {

                                max = input[sv::to1D(n, (y + j), (x + i), width, height)];
                            }
                        }
                    }

                    out[idx++] = max;
                }
            }
        }

        output = out;

#ifdef BENCHMARK
        MAC = 0;
        PARAM = 0;
        MEM = output.data().size() * sizeof(dtype);
#endif
    }

    template <typename dtype>
    void fc(sv::Tensor<dtype> &input, sv::Tensor<dtype> &output,
            sv::Tensor<dtype> const &weight, sv::Tensor<dtype> const &bias)
    {
#ifdef BENCHMARK
        MEM = 0;
        PARAM = 0;
        MAC = 0;
#endif

        auto inputShape = input.shape();
        auto inputLength = input.data().size();
        auto weightShape = weight.shape();

        int outputSize = weightShape[1];
        output = sv::Tensor<dtype>(outputSize);

#pragma omp parallel for
        for (int i = 0; i < outputSize; i++)
        {
            output[i] = bias[i];

            // input data stores as flattened, so no need to flatten
            for (int k = 0; k < inputLength; k++)
            {
                output[i] += input[k] * weight[i + outputSize * k];
            }
        }

#ifdef BENCHMARK
        auto weightLength = weight.data().size();

        MAC = inputLength * outputSize;
        PARAM = weightLength + outputSize;
        MEM = output.data().size() * sizeof(dtype);
#endif
    }

} // namespace sv
