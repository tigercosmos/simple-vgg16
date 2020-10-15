#pragma once
#include "Tensor.hpp"

namespace sv
{
    template <typename dtype>
    dtype ReLU(dtype input)
    {
        return input < 0 ? 0 : input;
    }
} // namespace sv
