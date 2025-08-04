#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <functional>

#include "common.hpp"
#include "layer.hpp"
#include "loss.hpp"


class Model {
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss;
    std::function<void(int epoch, double loss)> on_epoch_end_callback;
    double lr;
    int epochs;

public:
    Model(
        std::unique_ptr<Loss> loss,
        double lr,
        int epochs,
        std::function<void(int, double)> on_epoch_end_callback = nullptr
    ) : loss(std::move(loss)), lr(lr), epochs(epochs), on_epoch_end_callback(on_epoch_end_callback) {
        layers = std::vector<std::unique_ptr<Layer>>();
    }

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }
    
    void train(xt::xarray<double> inputs, xt::xarray<double> truths) {
        double err;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.shape()[0]; i++) {
                xt::xarray<double> curr = xt::row(inputs, i);
                for (auto& layer: layers) {
                    curr = layer->forward(curr);
                }
                xt::xarray<double> truth = xt::row(truths, i);
                err = loss->forward(curr, truth);
                
                xt::xarray<double> grad = loss->backward(curr, truth);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    grad = layers[j]->backward(grad, lr);
                }
            }
            if (on_epoch_end_callback) {
                on_epoch_end_callback(epoch, err);
            }
        }
    }
};

void test_callback(int epoch, double loss) {
    std::cout << "epoch: " << epoch << ", loss: " << loss << std::endl;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    xt::random::seed(time(NULL));

    xt::xarray<double> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    xt::xarray<double> truths = {0.0, 1.0, 1.0, 0.0};
    truths.reshape({4, 1});
    
    double lr = 1e-2;
    double err;
    int epochs = 1000000;

    Model model(std::make_unique<MSELoss>(), lr, epochs, test_callback);

    model.addLayer(std::make_unique<DenseLayer>(2, 2));
    model.addLayer(std::make_unique<ReLULayer>());

    model.addLayer(std::make_unique<DenseLayer>(2, 1));
    model.addLayer(std::make_unique<ReLULayer>());
    model.train(inputs, truths);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
    return 0;
}