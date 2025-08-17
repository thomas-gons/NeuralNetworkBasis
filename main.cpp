#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <functional>

#include "model.hpp"



void test_callback(const EpochResult& result) {
	std::cout << "epoch: " << result.epoch 
			  << ", train_loss: " << result.train_loss 
			  << ", val_loss: " << result.val_loss
			  << ", val_accuracy: " << result.val_accuracy
			  << std::endl;
}

int main() {
	auto config = load_json("../config.json");
	if (config.is_null()) {
		std::cerr << "Failed to load config.json" << std::endl;
		return 1;
	}

	IrisDataset dataset(
		"../data/Iris/iris.data"
	);

	auto start = std::chrono::high_resolution_clock::now();
	xt::random::seed(time(NULL));


	float validation_split = config.value("validation_split", 0.2);

	// Split the dataset into training and validation sets
	auto splits = dataset.split_dataset(validation_split, 0.1f);

	auto train_dataloader = Dataloader(
		std::move(splits.train),
		config.value("train_batch_size", 4),
		true
	);

	auto val_dataloader = Dataloader(
		std::move(splits.val),
		config.value("val_batch_size", 4)
	);

	float lr = config.value("learning_rate", 1e-4);
	float weight_decay = config.value("weight_decay", 1e-4);
	int batch_size = config.value("batch_size", 4);
	int epochs = config.value("epochs", 1000);

	auto loss = std::make_unique<loss::CrossEntropy>();
	Model model(std::move(loss), lr, weight_decay, epochs, test_callback);
	model.addLayer(std::make_unique<DenseLayer>(4, 16));
	model.addLayer(std::make_unique<activation::ReLU>());
	model.addLayer(std::make_unique<DenseLayer>(16, 3));
	model.addLayer(std::make_unique<activation::Softmax>());

	model.train(train_dataloader, val_dataloader);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	std::cout << "\nTemps total d'exécution : " << duration.count() << " secondes" << std::endl;
	return 0;
}