#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <functional>

#include "common.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "utils/dataset.hpp"
#include "utils/dataloader.hpp"


class Model {
	std::vector<std::unique_ptr<Layer>> layers;
	std::unique_ptr<loss::Loss> loss;
	float lr;
	int epochs;
	std::function<void(int epoch, float loss)> on_epoch_end_callback;

public:
	Model(std::unique_ptr<loss::Loss> loss, float lr, int epochs, std::function<void(int, float)> on_epoch_end_callback = nullptr)
		: loss(std::move(loss)), lr(lr),
		  epochs(epochs),
		  on_epoch_end_callback(on_epoch_end_callback)
	{
		layers = std::vector<std::unique_ptr<Layer>>();
	}

	void addLayer(std::unique_ptr<Layer> p_layer) {
		layers.push_back(std::move(p_layer));
	}

	void train(Dataloader& train_dataloader) {
		float err = 0.0f;
		int batch_done;

		auto softmax_layer = dynamic_cast<activation::Softmax*>(layers.back().get());
		auto cross_entropy_loss = dynamic_cast<loss::CrossEntropy*>(loss.get());

		for (int epoch = 0; epoch < epochs;  epoch++) {
			batch_done = 0;
			for (auto [inputs, truths] : train_dataloader) {
				auto curr = inputs;
				for (auto& layer : layers) {
					curr = layer->forward(curr);
				}
				
				auto batch_err = loss->forward(curr, truths);
				err += batch_err;
				
				xt::xarray<float> grad;

				if (softmax_layer && cross_entropy_loss) {
					grad = cross_entropy_loss->backward_fused(curr, truths);

					for (int j = layers.size() - 2; j >= 0; j--) {
						grad = layers[j]->backward(grad, lr);
					}
				} else {
					grad = loss->backward(curr, truths);

					for (int j = layers.size() - 1; j >= 0; j--) {
						grad = layers[j]->backward(grad, lr);
					}
				}

				batch_done++;
				std::cout << "Batch " << batch_done << "/" << train_dataloader.n_batches << std::endl;
			}
			err /= train_dataloader.n_batches;
			if (on_epoch_end_callback) {
				on_epoch_end_callback(epoch, err);
			}
			err = 0.0;
		}

	}
};

void test_callback(int epoch, float loss) {
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

	auto train_dataloader = Dataloader(
		std::move(dataset.training_images),
		std::move(dataset.training_labels),
		config.value("batch_size", 4)
	);

	auto image_size = dataset.training_images.shape()[1];

	auto loss = std::make_unique<loss::CrossEntropy>();
	Model model(std::move(loss), lr, epochs, test_callback);
	model.addLayer(std::make_unique<DenseLayer>(image_size, 128));
	model.addLayer(std::make_unique<activation::Sigmoid>());
	model.addLayer(std::make_unique<DenseLayer>(128, 10));
	model.addLayer(std::make_unique<activation::Softmax>());

	model.train(train_dataloader);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
	return 0;
}