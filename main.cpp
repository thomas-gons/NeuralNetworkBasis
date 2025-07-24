#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include <memory>

#include "layer.hpp"
#include "loss.hpp"

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss;
    double learning_rate;
    int epochs;

public:
    Model(double p_learning_rate, int p_epochs) : learning_rate(p_learning_rate), epochs(p_epochs) {}

    void addLayer(std::unique_ptr<Layer> p_layer) {
        layers.push_back(std::move(p_layer));
    }

    template<typename... Layers>
    void addLayers(Layers&&... p_layers) {
        (addLayer(std::move(p_layers)), ...);
    }

    void addLoss(std::unique_ptr<Loss> p_loss) {
        loss = std::move(p_loss);
    }

    double computeLoss(Tensor pred, Tensor truth) {
        return loss->forward(pred, truth);
    } 

    Tensor predict(const Tensor& input) {
        Tensor current_output = input;
        for (auto& layer : layers) {
            current_output = layer->forward(current_output);
        }
        return current_output;
    }

    void train(const Tensor& input, const Tensor truth) {
        Tensor pred = predict(input);

        Tensor grad = loss->backward(pred, truth);

        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad, learning_rate);
        }
    }

    ~Model() = default;
};


int main() {
    std::vector<Eigen::Tensor<double, 1> x_data = {
        (Eigen::Tensor(2) << 0, 0).finished(),
        (Eigen::Tensor(2) << 1, 0).finished(),
        (Eigen::Tensor(2) << 0, 1).finished(),
        (Eigen::Tensor(2) << 1, 1).finished()
    };
    std::vector<Eigen::Tensor> y_data = {
        (Eigen::Tensor(1) << 0).finished(),
        (Eigen::Tensor(1) << 1).finished(),
        (Eigen::Tensor(1) << 1).finished(),
        (Eigen::Tensor(1) << 0).finished()
    };

    double lr = 0.1;
    int epochs = 100;
    Model model(lr, epochs);

    model.addLayers(
        std::make_unique<DenseLayer>(2, 3),
        std::make_unique<SigmoidLayer>(),
        std::make_unique<DenseLayer>(3, 1),
        std::make_unique<SigmoidLayer>()
    );

    model.addLoss(std::make_unique<MSELoss>());

    for (int epoch = 0; epoch < 20000; epoch++) {
        double total_loss = 0.0;
        for (size_t i = 0; i < x_data.size(); i++) {
            model.train(x_data[i], y_data[i]);
            total_loss += model.computeLoss(model.predict(x_data[i]), y_data[i]);
        }

        if (epoch > 0 && epoch % 2000 == 0) {
            std::cout << "Epoque " << epoch << ", Perte moyenne: " << total_loss / x_data.size() << std::endl;
        }
    }

    std::cout << "\n--- Test apres entrainement ---\n";
    for (size_t i = 0; i < x_data.size(); i++) {
        Tensor pred = model.predict(x_data[i]);
        std::cout << "Entree: (" << x_data[i](0) << ", " << x_data[i](1)
                  << ") -> Prediction: " << pred(0) << " (Cible: " << y_data[i](0) << ")" << std::endl;
    }

    return 0;
}