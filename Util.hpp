#pragma once
#include <type_traits>

#define STATIC_ASSERT_FLOAT_TYPE(T)     \
    static_assert(is_floating_point<T>, \
                  "Can only be used with float types")

template <typename T>
constexpr bool is_floating_point = std::is_floating_point<T>::value;

namespace sv
{
    int to1D(int z, int y, int x, int ySize, int zSize)
    {
        return (ySize * zSize * z) + (ySize * y) + x;
    }

    int to1D(int f, int z, int y, int x, int ySize, int zSize, int fSize)
    {
        return (ySize * zSize * fSize * f) + (ySize * zSize * z) + (ySize * y) + x;
    }
} // namespace sv
