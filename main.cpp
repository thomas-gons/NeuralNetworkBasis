#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <Eigen/Dense>

#include "layer.hpp"
#include "loss.hpp"
#include <memory>

class Model {
public:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss;
    double lr;

    public:
    Model(double lr) : lr(lr) {}

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    template<typename... Layers>
    void addLayers(Layers&&... ls) {
        (addLayer(std::move(ls)), ...);
    }

    void addLoss(std::unique_ptr<Loss> l) {
        loss = std::move(l);
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        Eigen::VectorXd current_output = input;
         for (auto& layer : layers) {
            current_output = (*layer).forward(current_output);
        }
        return current_output;
    }

    void train(const Eigen::VectorXd& input, const Eigen::VectorXd truth) {
        Eigen::VectorXd pred = predict(input);

        Eigen::VectorXd grad = (*layers[layers.size() - 1]).backward(pred, lr); 

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer& current_layer = *layers[i];
            grad = current_layer.backward(grad, lr);
        }
    }

    ~Model() = default;
};


int main() {
    // Utilisation de conteneurs C++ modernes pour les données
    std::vector<Eigen::VectorXd> x_data = {
        (Eigen::VectorXd(2) << 0, 0).finished(),
        (Eigen::VectorXd(2) << 1, 0).finished(),
        (Eigen::VectorXd(2) << 0, 1).finished(),
        (Eigen::VectorXd(2) << 1, 1).finished()
    };
    std::vector<Eigen::VectorXd> y_data = {
        (Eigen::VectorXd(1) << 0).finished(),
        (Eigen::VectorXd(1) << 1).finished(),
        (Eigen::VectorXd(1) << 1).finished(),
        (Eigen::VectorXd(1) << 0).finished()
    };

    double lr = 0.1;
    Model model(lr);
    
    model.addLayers(
        std::make_unique<DenseLayer>(2, 3),
        std::make_unique<SigmoidLayer>(),
        std::make_unique<DenseLayer>(3, 1),
        std::make_unique<SigmoidLayer>()
    );

    for (int epoch = 0; epoch < 20000; epoch++) {
        double total_loss = 0.0;
        for (size_t i = 0; i < x_data.size(); i++) {
            model.train(x_data[i], y_data[i]);
            // Calculer la perte pour l'affichage
            total_loss += (*model.loss).forward(model.predict(x_data[i]), y_data[i]);
        }

        if (epoch > 0 && epoch % 2000 == 0) {
            std::cout << "Epoque " << epoch << ", Perte moyenne: " << total_loss / x_data.size() << std::endl;
        }
    }

    std::cout << "\n--- Test apres entrainement ---\n";
    for (size_t i = 0; i < x_data.size(); i++) {
        Eigen::VectorXd pred = model.predict(x_data[i]);
        std::cout << "Entree: (" << x_data[i](0) << ", " << x_data[i](1)
                  << ") -> Prediction: " << pred(0) << " (Cible: " << y_data[i](0) << ")" << std::endl;
    }

    return 0;
}