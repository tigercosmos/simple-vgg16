#pragma once
#include <cassert>
#include "Activation.hpp"
namespace sd
{
    template <typename dtype>
    void conv2d(sd::Tensor<dtype> &input, sd::Tensor<dtype> &output,
                sd::Tensor<dtype> const &kernels, sd::Tensor<dtype> const &bias)
    {
        auto inputShape = input.shape();
        auto kernelsShape = kernels.shape();

        int outSize = inputShape[0] - kernelsShape[0] + 1;
        output = sd::Tensor<dtype>(outSize, outSize, kernelsShape[2]);
        auto outputShape = output.shape();

        assert(outputShape.size() == 3);
        assert(kernelsShape.size() == 3);

        int n, x, y, i, j, k;
        int R = inputShape[2], S = kernelsShape[0], C = kernelsShape[1],
            N = outputShape[2], F = outputShape[0], E = outputShape[1];

        for (n = 0; n < N; n++)
        {
            for (x = 0; x < F; x++)
            {
                for (y = 0; y < E; y++)
                {
                    dtype sum = 0;
                    for (k = 0; k < R; k++)
                    {
                        for (i = 0; i < S; i++)
                        {
                            for (j = 0; j < C; j++)
                            {
                                dtype inputWeight = input[sd::to1D(k, (y + j), (x + i), F, E)];
                                dtype kernelWeight = kernels[sd::to1D(n, j, i, S, C)];
                                sum += inputWeight * kernelWeight;
                            }
                        }
                    }

                    sum += bias[n];
                    output[sd::to1D(n, y, x, F, E)] = sd::ReLU(sum);
                }
            }
        }
    }

    template <typename dtype>
    void maxpool(sd::Tensor<dtype> &input, sd::Tensor<dtype> &output, int const &poolSize, int const &stride)
    {
        auto inputShape = input.shape();
        assert(inputShape.size() == 3);

        int width = inputShape[1];
        int height = inputShape[0];
        int outSize = (height - poolSize) / stride + 1;
        sd::Tensor<dtype> out(outSize, outSize, inputShape[2]);

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
                            if (input[sd::to1D(n, (y + j), (x + i), width, height)] > max)
                            {
                                max = input[sd::to1D(n, (y + j), (x + i), width, height)];
                            }
                        }
                    }

                    out[idx++] = max;
                }
            }
        }

        output = out;
    }

    template <typename dtype>
    void fc(sd::Tensor<dtype> &input, sd::Tensor<dtype> &output,
            sd::Tensor<dtype> const &weight, sd::Tensor<dtype> const &bias)
    {
        auto inputShape = input.shape();
        auto weightShape = weight.shape();

        int outputSize = weightShape[1];
        output = sd::Tensor<dtype>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            output[i] = bias[i];

            // input data stores as flattened, so no need to flatten
            for (int k = 0; k < input.data().size(); k++)
            {
                output[i] += input[k] * weight[i + outputSize * k];
            }
        }
    }

} // namespace sd
