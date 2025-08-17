#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include "common.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "utils/dataloader.hpp"


struct EpochResult {
    int epoch;
    float train_loss;
    float val_loss;
    float val_accuracy;
};

using EpochEndCallback = std::function<void(const EpochResult&)>;


class Model {
	std::vector<std::unique_ptr<Layer>> layers;
	std::unique_ptr<loss::Loss> loss;
	float lr;
	float weight_decay;
	int epochs;
	EpochEndCallback on_epoch_end_callback;
    bool softmax_cross_entropy = false;

public:
	Model(std::unique_ptr<loss::Loss> loss, float lr, float weight_decay, int epochs, EpochEndCallback on_epoch_end_callback = nullptr)
		: loss(std::move(loss)), lr(lr),
		  weight_decay(weight_decay),
		  epochs(epochs),
		  on_epoch_end_callback(on_epoch_end_callback)
	{
		layers = std::vector<std::unique_ptr<Layer>>();
	}

	void addLayer(std::unique_ptr<Layer> p_layer) {
		layers.push_back(std::move(p_layer));
        auto softmax_layer = dynamic_cast<activation::Softmax*>(layers.back().get());
        auto cross_entropy_loss = dynamic_cast<loss::CrossEntropy*>(loss.get());
        softmax_cross_entropy = (softmax_layer && cross_entropy_loss);
	}

	void train(Dataloader& train_dataloader, Dataloader& val_dataloader);

private:
    float train_step(xt::xarray<float>& inputs, xt::xarray<float>& truths, float dynamic_lr);
    std::tuple<float, unsigned int> validation_step(xt::xarray<float>& inputs, xt::xarray<float>& truths);
};

class ProgressBar {
    int epochs;
    int total_batches;
    int bar_width = 50;
    int batch_done = 0;
    float elapsed_time_s = 0.0f;
    float remaining_time_s = 0.0f;

public:
    ProgressBar(int epochs, int total_batches)
        : epochs(epochs), total_batches(total_batches) {};


    void update(int epoch, float batch_err, double elapsed_time_s_double);
    void clear();
};

#endif