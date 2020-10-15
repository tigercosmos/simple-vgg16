#pragma once
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Util.hpp"
#include <vector>
namespace sv
{
    template <typename dtype>
    class Network
    {
    private:
        std::vector<sv::Layer<dtype> *> layers;

    public:
        Network()
        {
            STATIC_ASSERT_FLOAT_TYPE(dtype);
        };
        ~Network() = default;

        void addLayer(sv::Layer<dtype> *layer)
        {
            this->layers.push_back(layer);
        }

        void printLayers()
        {
            std::cout << "=== Network ===" << std::endl;
            for (auto layer : layers)
            {
                std::cout << "--- " << layer->getName() << " ---" << std::endl;
                layer->print();
            }
            std::cout << "===============" << std::endl;
        }

        void predict(Tensor<dtype> &input, Tensor<dtype> &output)
        {
            for (auto layer : layers)
            {
                layer->forward(input, output);
                if (layer != layers.back())
                {
                    input = output;
                }
            }
        }
    };
} // namespace sv