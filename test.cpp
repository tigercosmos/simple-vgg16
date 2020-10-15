#include <iostream>
#include <memory>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Network.hpp"

void testSimpleNet()
{
    sd::Tensor<double> output;

    std::cout << "Test Simple Net" << std::endl;

    auto t1 = sd::Tensor<double>(6, 6, 3); // 6 x 6 x 3
    t1.randam();

    auto network = sd::Network<double>();

    auto *layer2 = new sd::ConvLayer<double>(3, 6); // 6 x 6 x 6
    network.addLayer(layer2);

    auto *layer3 = new sd::MaxPoolLayer<double>(2, 2); // 3 x 3 x 3
    network.addLayer(layer3);

    auto *layer4 = new sd::FCLayer<double>(27, 6); // 1 x 1 x 6
    network.addLayer(layer4);

    network.printLayers();
    network.predict(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testMaxPool()
{
    sd::Tensor<double> output;

    std::cout << "Test MaxPool" << std::endl;
    auto t1 = sd::Tensor<double>(4, 4, 1);
    std::cout << "origin input" << std::endl;
    for (int i = 0; i < t1.data().size(); i++)
    {
        t1.data()[i] = i;
    }
    std::cout << t1 << std::endl;

    auto *layer = new sd::MaxPoolLayer<double>(2, 2);
    layer->print();
    layer->forward(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testFC()
{
    sd::Tensor<double> output;

    std::cout << "Test ConvPool" << std::endl;
    auto t1 = sd::Tensor<double>(10, 10, 1);
    std::cout << "origin input" << std::endl;
    for (int i = 3; i < 6; i++)
    {
        for (int j = 3; j < 6; j++)
        {
            t1.data()[i * 10 + j] = 1;
        }
    }
    std::cout << t1 << std::endl;

    auto *layer = new sd::FCLayer<double>(10 * 10 * 1, 7);
    // layer->print();
    layer->forward(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testConv()
{
    sd::Tensor<double> output;

    std::cout << "Test FC" << std::endl;
    auto t1 = sd::Tensor<double>(4, 4, 1);
    std::cout << "origin input" << std::endl;
    for (int i = 0; i < t1.data().size(); i++)
    {
        t1.data()[i] = i;
    }
    std::cout << t1 << std::endl;

    auto *layer = new sd::MaxPoolLayer<double>(2, 2);
    layer->print();
    layer->forward(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

int main()
{
    testMaxPool();
    std::cout << std::endl;

    testConv();
    std::cout << std::endl;

    testFC();
    std::cout << std::endl;

    testSimpleNet();
    std::cout << std::endl;

    return 0;
}