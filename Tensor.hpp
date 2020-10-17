#pragma once
#include <vector>
#include <iostream>
#include <sstream>

namespace sv
{

    template <typename dtype>
    class Tensor
    {
    private:
        std::vector<dtype> mData;
        std::vector<int> mShape;

    public:
        Tensor()
        {
        }
        Tensor(Tensor &other) : mData{other.mData}, mShape{other.mShape} {}
        Tensor(Tensor &&other) : mData{std::move(other.mData)}, mShape{std::move(other.mShape)} {}
        Tensor(int a)
        {
            mData.resize(a);
            mShape.push_back(a);
        }
        Tensor(int a, int b)
        {
            mData.resize(a * b);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
        }
        Tensor(int a, int b, int c)
        {
            mData.resize(a * b * c);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
            this->mShape.push_back(c);
        }
        Tensor(int a, int b, int c, int d)
        {
            mData.resize(a * b * c * d);
            this->mShape.push_back(a);
            this->mShape.push_back(b);
            this->mShape.push_back(c);
            this->mShape.push_back(d);
        }

        Tensor &operator=(Tensor<dtype> &tensor)
        {
            mShape = std::move(tensor.mShape);
            mData = std::move(tensor.mData);
            return *this;
        }
        Tensor &operator=(Tensor<dtype> &&tensor)
        {
            mShape = std::move(tensor.mShape);
            mData = std::move(tensor.mData);
            return *this;
        }

        dtype &operator[](int idx)
        {
            return mData[idx];
        }

        dtype const &operator[](int idx) const
        {
            return mData[idx];
        }

        ~Tensor() {}

        std::vector<dtype> &data()
        {
            return this->mData;
        }

        std::vector<dtype> data() const
        {
            return this->mData;
        }

        std::vector<int> shape()
        {
            return this->mShape;
        }

        std::vector<int> shape() const
        {
            return this->mShape;
        }

        std::string shapeStr() const
        {
            std::stringstream ss;
            for (int i = 0; i < this->mShape.size(); i++)
            {
                ss << mShape[i] << " ";
            }
            return ss.str();
        }

        std::string shapeStr()
        {
            std::stringstream ss;
            for (int i = 0; i < this->mShape.size(); i++)
            {
                ss << mShape[i] << " ";
            }
            return ss.str();
        }

        std::string str() const
        {
            if (mShape.size() == 4)
                return strND(4);
            if (mShape.size() == 3)
                return strND(3);
            if (mShape.size() == 2)
                return strND(2);
            return strND(1);
        }

        std::string strND(int N) const
        {
            std::stringstream ss;
            for (int i = 0; i < mData.size(); i++)
            {
                if (N == 4)
                {
                    if (i % (mShape[0] * mShape[1] * mShape[2]) == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                if (N >= 3)
                {
                    if (i % (mShape[0] * mShape[1]) == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                if (N >= 2)
                {
                    if (i % mShape[0] == 0 && i != 0)
                    {
                        ss << std::endl;
                    }
                }
                ss << mData[i] << " ";
            }
            return ss.str();
        }

        friend std::ostream &operator<<(std::ostream &inOStream, Tensor const &tensor)
        {
            inOStream << tensor.str();
            return inOStream;
        }

        void randam()
        {
            for (int i = 0; i < mData.size(); i++)
            {
                mData[i] = (double)std::rand() / RAND_MAX / 1000;
            }
        }
    };
} // namespace sv
