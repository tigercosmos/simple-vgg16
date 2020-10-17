#pragma once
#include <cassert>
#include "Activation.hpp"

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
        // int i_h = inputShape[1];
        // int i_f = inputShape[2]; // fmap
        auto weightShape = weight.shape();
        int w_w = weightShape[0];
        int w_h = weightShape[1];
        int w_f = weightShape[2]; // fmap
        int w_c = weightShape[3]; // channel

        int outSize = i_w - w_w + 1;
        output = sv::Tensor<dtype>(outSize, outSize, w_c);
        auto outputShape = output.shape();
        int o_w = outputShape[0];
        int o_h = outputShape[1];
        int o_c = outputShape[2];

        assert(outputShape.size() == 3);
        assert(weightShape.size() == 4);

        int n, m, x, y, i, j, k;

        for (m = 0; m < w_f; m++) // kernel fmap
        {
            for (n = 0; n < o_c; n++) // kernel channel
            {
                for (x = 0; x < o_w; x++) //  output x
                {
                    for (y = 0; y < o_h; y++) // output y
                    {
                        dtype sum = 0;
                        for (k = 0; k < o_c; k++) // output channel
                        {
                            for (i = 0; i < w_w; i++) // kernel x
                            {
                                for (j = 0; j < w_h; j++) // kernel y
                                {
                                    dtype inputWeight = input[sv::to1D(k, (y + j), (x + i), o_w, o_h)];
                                    dtype kernelWeight = weight[sv::to1D(m, n, j, i, w_w, w_h, w_f)];
                                    sum += inputWeight * kernelWeight;
                                }
                            }
                        }

                        sum += bias[n];
                        output[sv::to1D(n, y, x, o_w, o_h)] = sv::ReLU(sum);
                    }
                }
            }
        }

#ifdef BENCHMARK
        MAC = w_c * w_f * w_h * w_w * o_c * o_w * o_h;
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

        int n, x, y, i, j, u, v, idx = 0;
        dtype max;
        for (n = 0; n < inputShape[2]; n++)
        {
            for (y = 0, v = 0; y < height; y += stride)
            {
                for (x = 0, u = 0; x < width; x += stride)
                {
                    max = -100000000000;

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
