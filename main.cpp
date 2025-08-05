#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

#include "common.hpp"
#include "layer.hpp"
#include "loss.hpp"


class Model {
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss;
    float lr;
    int epochs;
    int batch_size;
    std::function<void(int epoch, float loss)> on_epoch_end_callback;

public:
    Model(
        std::unique_ptr<Loss> loss,
        float lr,
        int batch_size,
        int epochs,
        std::function<void(int, float)> on_epoch_end_callback = nullptr
    ) : loss(std::move(loss)), lr(lr),
        batch_size(batch_size), epochs(epochs),
        on_epoch_end_callback(on_epoch_end_callback)
    {
        layers = std::vector<std::unique_ptr<Layer>>();
    }

    void addLayer(std::unique_ptr<Layer> p_layer) {
        layers.push_back(std::move(p_layer));
    }
    
    void train(xt::xarray<float> inputs, xt::xarray<float> truths) {
        float err;
        auto n_batches = inputs.shape()[0] / batch_size;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < n_batches; batch++) {
                xt::xarray<float> curr = xt::view(inputs, xt::range(batch * batch_size, (batch + 1) * batch_size));
                for (auto& layer: layers) {
                    curr = layer->forward(curr);
                }   
                xt::xarray<float> truth = xt::view(truths, xt::range(batch * batch_size, (batch + 1) * batch_size));
                auto batch_err = loss->forward(curr, truth);
                err += batch_err;

                xt::xarray<float> grad = loss->backward(curr, truth);

                for (int j = layers.size() - 1; j >= 0; j--) {
                    grad = layers[j]->backward(grad, lr);
                }
            }
            err /= inputs.shape()[0];
            if (on_epoch_end_callback) {
                on_epoch_end_callback(epoch, err);
            }
            err = 0.0;
        }
    }
};

void test_callback(int epoch, float loss) {
    if (epoch % 10000 != 0) return;
    std::cout << "epoch: " << epoch << ", loss: " << loss << std::endl;
}

int main() {
    auto config = load_json("../config.json");
    if (config.is_null()) {
        std::cerr << "Failed to load config.json" << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    xt::random::seed(time(NULL));

    xt::xarray<float> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    xt::xarray<float> truths = {0, 1, 1, 0};
    truths.reshape({4, 1});
    
    float lr = config.value("learning_rate", 1e-4);
    int batch_size = config.value("batch_size", 4);
    int epochs = config.value("epochs", 1000);

    int max_batch_size = inputs.shape()[0];
    if (batch_size > max_batch_size) {
        std::cout << "\e[1;33mBatch size cannot be greater than the number of inputs.\e[0m" << std::endl;
        std::cout << "Using batch size: " << max_batch_size << std::endl;
        batch_size = max_batch_size;
    }

    Model model(std::make_unique<MSELoss>(), lr, batch_size, epochs, test_callback);

    model.addLayer(std::make_unique<DenseLayer>(2, 2));
    model.addLayer(std::make_unique<SigmoidLayer>());
    model.addLayer(std::make_unique<DenseLayer>(2, 1));
    model.addLayer(std::make_unique<SigmoidLayer>());
    model.train(inputs, truths);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
    return 0;
}