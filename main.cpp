#include <iostream>
#include <memory>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Network.hpp"

int main()
{
    std::cout << "The VGG16 Net" << std::endl;

    // prepare the input data
    auto input = sv::Tensor<double>(224, 224, 3); // 224 x 224 x 3
    input.randam();

    // prepare the network
    auto network = sv::Network<double>();

    // add layers into the network
    auto *layer1_1 = new sv::ConvLayer<double>(3, 64); //224 x 224 x 64
    network.addLayer(layer1_1);
    auto *layer1_2 = new sv::ConvLayer<double>(3, 64); //224 x 224 x 64
    network.addLayer(layer1_2);
    auto *layer1_3 = new sv::MaxPoolLayer<double>(2, 2); // 112 x 112 x 64
    network.addLayer(layer1_3);

    auto *layer2_1 = new sv::ConvLayer<double>(3, 128); // 112 x 112 x 128
    network.addLayer(layer2_1);
    auto *layer2_2 = new sv::ConvLayer<double>(3, 128); // 112 x 112 x 128
    network.addLayer(layer2_2);
    auto *layer2_3 = new sv::MaxPoolLayer<double>(2, 2); // 56 x 56 x 64
    network.addLayer(layer2_3);

    auto *layer3_1 = new sv::ConvLayer<double>(3, 256); // 56 x 56 x 256
    network.addLayer(layer3_1);
    auto *layer3_2 = new sv::ConvLayer<double>(3, 256); // 56 x 56 x 256
    network.addLayer(layer3_2);
    auto *layer3_3 = new sv::MaxPoolLayer<double>(2, 2); // 28 x 28 x 256
    network.addLayer(layer3_3);

    auto *layer4_1 = new sv::ConvLayer<double>(3, 512); // 28 x 28 x 512
    network.addLayer(layer4_1);
    auto *layer4_2 = new sv::ConvLayer<double>(3, 512); // 28 x 28 x 512
    network.addLayer(layer4_2);
    auto *layer4_3 = new sv::MaxPoolLayer<double>(2, 2); // 14 x 14 x 512
    network.addLayer(layer4_3);

    auto *layer5_1 = new sv::ConvLayer<double>(3, 512); // 14 x 14 x 512
    network.addLayer(layer5_1);
    auto *layer5_2 = new sv::ConvLayer<double>(3, 512); // 14 x 14 x 512
    network.addLayer(layer5_2);
    auto *layer5_3 = new sv::MaxPoolLayer<double>(2, 2); // 7 x 7 x 512
    network.addLayer(layer5_3);

    auto *layer6_1 = new sv::FCLayer<double>(7 * 7 * 512, 4096); // 1 x 1 x 4096
    network.addLayer(layer6_1);
    auto *layer6_2 = new sv::FCLayer<double>(4096, 4096); // 1 x 1 x 4096
    network.addLayer(layer6_2);
    auto *layer6_3 = new sv::FCLayer<double>(4096, 1000); // 1 x 1 x 1000
    network.addLayer(layer6_3);

    // predict the result by forwarding
    sv::Tensor<double> output;
    network.predict(input, output);

    // show the result
    std::cout << "@@ result @@" << std::endl;
    for (int i = 0; i < 5; i++)
        std::cout << output[i] << " ";
    std::cout << "... so many ... ";
    for (int i = 995; i < 1000; i++)
        std::cout << output[i] << " ";
    std::cout << std::endl;

    return 0;
}