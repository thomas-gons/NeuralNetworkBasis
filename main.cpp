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
	std::function<void(int epoch, float train_loss, float val_loss)> on_epoch_end_callback;

public:
	Model(std::unique_ptr<loss::Loss> loss, float lr, int epochs, std::function<void(int, float, float)> on_epoch_end_callback = nullptr)
		: loss(std::move(loss)), lr(lr),
		  epochs(epochs),
		  on_epoch_end_callback(on_epoch_end_callback)
	{
		layers = std::vector<std::unique_ptr<Layer>>();
	}

	void addLayer(std::unique_ptr<Layer> p_layer) {
		layers.push_back(std::move(p_layer));
	}

	void train(Dataloader& train_dataloader, Dataloader& val_dataloader) {
		float train_err = 0.0f;
		float val_err = 0.0f;

		auto softmax_layer = dynamic_cast<activation::Softmax*>(layers.back().get());
		auto cross_entropy_loss = dynamic_cast<loss::CrossEntropy*>(loss.get());

		for (int epoch = 0; epoch < epochs; epoch++) {
			train_err = 0.0f;
			val_err = 0.0f;
			
			int total_batches = train_dataloader.n_batches;
			int bar_width = 70;
			auto start_time = std::chrono::high_resolution_clock::now();

			int batch_done = 0;
			for (auto [inputs, truths] : train_dataloader) {
				auto curr = inputs;
				for (auto& layer : layers) {
					curr = layer->forward(curr);
				}
				
				auto batch_err = loss->forward(curr, truths);
				train_err += batch_err;
				
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
				
				float progress = (float)batch_done / total_batches;
				int pos = static_cast<int>(bar_width * progress);

				auto current_time = std::chrono::high_resolution_clock::now();
				auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
				double elapsed_time_s_double = static_cast<double>(elapsed_time_ms.count()) / 1000.0;
				double batches_per_sec = (batch_done > 0) ? (double)batch_done / elapsed_time_s_double : 0.0;
				double time_per_batch = elapsed_time_s_double / batch_done;
				double remaining_time_s_double = time_per_batch * (total_batches - batch_done);

				// Conversion des secondes en format MM:SS pour le temps écoulé
				int elapsed_minutes = static_cast<int>(elapsed_time_s_double) / 60;
				int elapsed_seconds = static_cast<int>(elapsed_time_s_double) % 60;

				// Conversion des secondes en format MM:SS pour le temps restant
				int remaining_minutes = static_cast<int>(remaining_time_s_double) / 60;
				int remaining_seconds = static_cast<int>(remaining_time_s_double) % 60;

				std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " | "
						<< "[" << std::string(pos, '=') << std::string(bar_width - pos, ' ') << "] "
						<< " | " << batch_done << "/" << total_batches
						<< " [" << std::setfill('0') << std::setw(2) << elapsed_minutes << ":" 
						<< std::setfill('0') << std::setw(2) << elapsed_seconds << "<" 
						<< std::setfill('0') << std::setw(2) << remaining_minutes << ":" 
						<< std::setfill('0') << std::setw(2) << remaining_seconds
						<< ", " << std::fixed << std::setprecision(2) << batches_per_sec << " it/s"
						<< "]";
				std::cout.flush();
			}

			std::cout << std::endl;
			
			train_err /= total_batches;
			
			// Boucle de validation (inchangée)
			for (auto [inputs, truths] : val_dataloader) {
				auto curr = inputs;
				for (auto& layer : layers) {
					curr = layer->forward(curr);
				}
				
				auto batch_err = loss->forward(curr, truths);
				val_err += batch_err;
			}
			val_err /= val_dataloader.n_batches;

			if (on_epoch_end_callback) {
				on_epoch_end_callback(epoch, train_err, val_err);
			}
		}
	}
};

void test_callback(int epoch, float train_loss, float val_loss) {
	std::cout << "epoch: " << epoch 
			  << ", train_loss: " << train_loss 
			  << ", val_loss: " << val_loss 
			  << std::endl;
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



	float validation_split = config.value("validation_split", 0.2);

	// Split the dataset into training and validation sets
	auto n_samples = dataset.training_images.shape()[0];
	auto n_val_samples = static_cast<int>(n_samples * validation_split);
	auto n_train_samples = n_samples - n_val_samples;
	auto training_images = xt::view(dataset.training_images, xt::range(0, n_train_samples), xt::all());
	auto training_labels = xt::view(dataset.training_labels, xt::range(0, n_train_samples), xt::all());
	auto validation_images = xt::view(dataset.training_images, xt::range(n_train_samples, n_samples), xt::all());
	auto validation_labels = xt::view(dataset.training_labels, xt::range(n_train_samples, n_samples), xt::all());

	auto train_dataloader = Dataloader(
		std::move(training_images),
		std::move(training_labels),
		config.value("train_batch_size", 4)
	);

	auto val_dataloader = Dataloader(
		std::move(validation_images),
		std::move(validation_labels),
		config.value("val_batch_size", 4)
	);

	auto image_size = dataset.training_images.shape()[1];

	float lr = config.value("learning_rate", 1e-4);
	int batch_size = config.value("batch_size", 4);
	int epochs = config.value("epochs", 1000);

	auto loss = std::make_unique<loss::CrossEntropy>();
	Model model(std::move(loss), lr, epochs, test_callback);
	model.addLayer(std::make_unique<DenseLayer>(image_size, 128));
	model.addLayer(std::make_unique<activation::Sigmoid>());
	model.addLayer(std::make_unique<DenseLayer>(128, 10));
	model.addLayer(std::make_unique<activation::Softmax>());

	model.train(train_dataloader, val_dataloader);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
	return 0;
}