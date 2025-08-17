#include "model.hpp"


void Model::train(Dataloader& train_dataloader, Dataloader& val_dataloader) {
    float train_err = 0.0f;
    float val_err = 0.0f;

    float dynamic_lr = lr;

    for (int epoch = 0; epoch < epochs; epoch++) {
        train_err = 0.0f;
        val_err = 0.0f;
        
        int total_batches = train_dataloader.n_batches;
        int bar_width = 50;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ProgressBar progress_bar(epochs, total_batches);
        for (auto [inputs, truths] : train_dataloader) {
            
            auto batch_err = train_step(inputs, truths, dynamic_lr);
            train_err += batch_err;

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            double elapsed_time_s = static_cast<double>(elapsed_time_ms.count()) / 1000.0;
            progress_bar.update(epoch, batch_err, elapsed_time_s);
        }
        progress_bar.clear();

        dynamic_lr = lr * std::exp(-weight_decay * epoch);
        train_err /= (float) total_batches;
        
        unsigned int correct_predictions = 0;
        float val_err = 0.0f;
        for (auto [inputs, truths] : val_dataloader) {
            auto [batch_val_err, correct_prediction] = validation_step(inputs, truths);
            val_err += batch_val_err;
            correct_predictions += correct_prediction;
        }

        val_err /= (float) val_dataloader.n_batches;
        float val_accuracy = (float) correct_predictions / (float) val_dataloader.total_samples;

        auto result = EpochResult{
            .epoch = epoch,
            .train_loss = train_err,
            .val_loss = val_err,
            .val_accuracy = val_accuracy,
        };

        if (on_epoch_end_callback) {
            on_epoch_end_callback(result);
        }
    }
}

float Model::train_step(xt::xarray<float>& inputs, xt::xarray<float>& truths, float dynamic_lr) {
    auto curr = inputs;
    for (auto& layer : layers) {
        curr = layer->forward(curr);
    }
    
    xt::xarray<float> predicted = xt::argmax(curr, {1});
    auto batch_err = loss->forward(predicted, truths);
    
    xt::xarray<float> grad;
    if (softmax_cross_entropy) {
        auto cross_entropy_loss = dynamic_cast<loss::CrossEntropy*>(loss.get());
        grad = cross_entropy_loss->backward_fused(curr, truths);

        for (int j = layers.size() - 2; j >= 0; j--) {
            grad = layers[j]->backward(grad, dynamic_lr);
        }
    } else {
        grad = loss->backward(curr, truths);

        for (int j = layers.size() - 1; j >= 0; j--) {
            grad = layers[j]->backward(grad, dynamic_lr);
        }
    }
    return batch_err;
}

std::tuple<float, uint> Model::validation_step(xt::xarray<float>& inputs, xt::xarray<float>& truths) {
    auto curr = inputs;
    for (auto& layer : layers) {
        curr = layer->forward(curr);
    }
    
    xt::xarray<size_t> predicted = xt::argmax(curr, {1});
    auto batch_err = loss->forward(predicted, truths);

    auto matches = xt::equal(predicted, truths);
    auto correct_predictions = xt::sum(matches)();
    return {batch_err, correct_predictions};
}

void ProgressBar::update(int epoch, float batch_err, double elapsed_time_s_double) {
    batch_done++;
            
    float progress = (float)batch_done / total_batches;
    int pos = static_cast<int>(bar_width * progress);

    
    double batches_per_sec = (batch_done > 0) ? (double)batch_done / elapsed_time_s_double : 0.0;
    double time_per_batch = elapsed_time_s_double / batch_done;
    double remaining_time_s_double = time_per_batch * (total_batches - batch_done);

    int elapsed_minutes = static_cast<int>(elapsed_time_s_double) / 60;
    int elapsed_seconds = static_cast<int>(elapsed_time_s_double) % 60;

    int remaining_minutes = static_cast<int>(remaining_time_s_double) / 60;
    int remaining_seconds = static_cast<int>(remaining_time_s_double) % 60;

    std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " | "
              << "[" << std::string(pos, '=') << std::string(bar_width - pos, ' ') << "] "
              << batch_done << "/" << total_batches
              << " [" << std::setfill('0') << std::setw(2) << elapsed_minutes << ":" 
              << std::setfill('0') << std::setw(2) << elapsed_seconds << "<" 
              << std::setfill('0') << std::setw(2) << remaining_minutes << ":" 
              << std::setfill('0') << std::setw(2) << remaining_seconds
              << ", " << std::fixed << std::setprecision(2) << batches_per_sec << " it/s] "
              << "Train Loss: " << batch_err;
    std::cout.flush();
}

void ProgressBar::clear() {
    std::cout << std::endl;
    batch_done = 0;
    elapsed_time_s = 0.0f;
    remaining_time_s = 0.0f;
}