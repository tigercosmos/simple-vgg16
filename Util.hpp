#pragma once
#include <type_traits>

#define STATIC_ASSERT_FLOAT_TYPE(T)     \
    static_assert(is_floating_point<T>, \
                  "Can only be used with float types")

template <typename T>
constexpr bool is_floating_point = std::is_floating_point<T>::value;

namespace sv
{
    int to1D(int z, int y, int x, int xSize, int ySize)
    {
        return (ySize * xSize * z) + (xSize * y) + x;
    }

    int to1D(int f, int z, int y, int x, int xSize, int ySize, int zSize)
    {
        return (ySize * zSize * xSize * f) + (ySize * xSize * z) + (xSize * y) + x;
    }
} // namespace sv
