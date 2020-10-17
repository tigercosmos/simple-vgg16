#pragma once
#include <type_traits>

#define STATIC_ASSERT_FLOAT_TYPE(T)     \
    static_assert(is_floating_point<T>, \
                  "Can only be used with float types")

template <typename T>
constexpr bool is_floating_point = std::is_floating_point<T>::value;

namespace sv
{
    int to1D(int z, int y, int x, int width, int height)
    {
        return (width * height * z) + (width * y) + x;
    }

    int to1D(int f, int z, int y, int x, int width, int height, int fsize)
    {
        return (width * height * fsize * f) + (width * height * z) + (width * y) + x;
    }
} // namespace sv
