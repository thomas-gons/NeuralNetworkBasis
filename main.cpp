#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <functional>

#include "common.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "dataloader/dataset.hpp"


class Model {
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<loss::Loss> loss;
    float lr;
    int epochs;
    int batch_size;
    std::function<void(int epoch, float loss)> on_epoch_end_callback;

public:
    Model(
        std::unique_ptr<loss::Loss> loss,
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
    
    MNISTDataset dataset(
        "../data/train-images-idx3-ubyte",
        "../data/train-labels-idx1-ubyte",
        "../data/t10k-images-idx3-ubyte",
        "../data/t10k-labels-idx1-ubyte"
    );

    auto start = std::chrono::high_resolution_clock::now();
    xt::random::seed(time(NULL));

    
    float lr = config.value("learning_rate", 1e-4);
    int batch_size = config.value("batch_size", 4);
    int epochs = config.value("epochs", 1000);

    int max_batch_size = dataset.training_images.shape()[0];
    if (batch_size > max_batch_size) {
        std::cout << "\e[1;33mBatch size cannot be greater than the number of inputs.\e[0m" << std::endl;
        std::cout << "Using batch size: " << max_batch_size << std::endl;
        batch_size = max_batch_size;
    }
    
    auto image_size = dataset.training_images.shape()[1];
    Model model(std::make_unique<loss::MSE>(), lr, batch_size, epochs, test_callback);
    model.addLayer(std::make_unique<DenseLayer>(image_size * image_size, 128));
    model.addLayer(std::make_unique<activation::Sigmoid>());
    model.addLayer(std::make_unique<DenseLayer>(128, 10));
    model.addLayer(std::make_unique<activation::Softmax>());

    return 0;
    model.train(dataset.training_images, dataset.training_labels);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
    return 0;
}