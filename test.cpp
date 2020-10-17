#include <iostream>
#include <memory>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Network.hpp"

void testSimpleNet()
{
    sv::Tensor<double> output;

    std::cout << "Test Simple Net" << std::endl;

    auto t1 = sv::Tensor<double>(6, 6, 3); // 6 x 6 x 3
    t1.randam();

    auto network = sv::Network<double>();

    auto *layer2 = new sv::ConvLayer<double>(3, 3, 6); // 6 x 6 x 6
    network.addLayer(layer2);

    auto *layer3 = new sv::MaxPoolLayer<double>(2, 2); // 3 x 3 x 6
    network.addLayer(layer3);

    auto *layer4 = new sv::FCLayer<double>(3 * 3 * 6, 7); // 1 x 1 x 7
    network.addLayer(layer4);

    network.printLayers();
    network.predict(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testFC()
{
    sv::Tensor<double> output;

    std::cout << "Test FC" << std::endl;
    auto t1 = sv::Tensor<double>(4, 4, 1);
    std::cout << "origin input" << std::endl;
    for (int i = 0; i < t1.data().size(); i++)
    {
        t1.data()[i] = i;
    }
    std::cout << t1 << std::endl;

    auto *layer = new sv::FCLayer<double>(4 * 4 * 1, 2);
    layer->print();
    layer->forward(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testConv()
{
    sv::Tensor<double> output;

    std::cout << "Test Conv" << std::endl;
    auto t1 = sv::Tensor<double>(3, 3, 2); // 5 x 5 x 2
    std::cout << "origin input" << std::endl;
    t1.randam();
    std::cout << t1 << std::endl;

    auto *layer = new sv::ConvLayer<double>(2, 3, 2); // 5 x 5 x 2
    layer->print();
    layer->forward(t1, output);

    std::cout << "@@ result @@" << std::endl;
    std::cout << output << std::endl;
}

void testMaxPool()
{
    sv::Tensor<double> output;

    std::cout << "Test MaxPool" << std::endl;
    auto t1 = sv::Tensor<double>(4, 4, 1);
    std::cout << "origin input" << std::endl;
    for (int i = 0; i < t1.data().size(); i++)
    {
        t1.data()[i] = i;
    }
    std::cout << t1 << std::endl;

    auto *layer = new sv::MaxPoolLayer<double>(2, 2);
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