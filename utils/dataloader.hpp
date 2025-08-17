#ifndef __DATALOADER_HPP__
#define __DATALOADER_HPP__

#include "../common.hpp"
#include "dataset.hpp"
#include <utility>

class Dataloader {
    xt::xarray<float> x_data;
    xt::xarray<float> y_data;
    
public:
    unsigned int batch_size;
    unsigned int n_batches;
    unsigned int total_samples;
    
    Dataloader(Subset subset, unsigned int batch_size, bool shuffle = false)
        : batch_size(batch_size),
          x_data(std::move(subset.data)),
          y_data(std::move(subset.labels)) 
    {
        if (this->x_data.shape()[0] != this->y_data.shape()[0]) {
            throw std::runtime_error("Input and output data must have the same number of samples.");
        }
        
        if (this->x_data.shape()[0] < batch_size) {
            throw std::runtime_error("Batch size must be <= number of samples.");
        }

        if (shuffle) {
            auto& gen = xt::random::get_default_random_engine();
            xt::random::shuffle(this->x_data, gen);
            xt::random::shuffle(this->y_data, gen);
        }

        n_batches = this->x_data.shape()[0] / batch_size;
        total_samples = this->x_data.shape()[0];
    }

    class iterator {
        const xt::xarray<float>& x_data;
        const xt::xarray<float>& y_data;
        unsigned int batch_size;
        unsigned int current_index;

    public:
        iterator(const xt::xarray<float>& x_data,
                 const xt::xarray<float>& y_data,
                 unsigned int batch_size,
                 unsigned int current_index = 0)
            : x_data(x_data), y_data(y_data), batch_size(batch_size), current_index(current_index) {}

        bool operator!=(const iterator& other) const {
            return current_index != other.current_index;
        }

        iterator& operator++() {
            current_index += batch_size;
            return *this;
        }

        std::pair<xt::xarray<float>, xt::xarray<float>> operator*() const {
            auto start = current_index;
            auto end = std::min(current_index + batch_size, (unsigned int)x_data.shape(0));

            auto x_batch = xt::view(x_data, xt::range(start, end), xt::all());
            auto y_batch = xt::view(y_data, xt::range(start, end), xt::all());

            return {x_batch, y_batch};
        }
    };

    iterator begin() {
        return iterator(x_data, y_data, batch_size, 0);
    }

    iterator end() {
        unsigned int last_batch_start = (x_data.shape(0) / batch_size) * batch_size;
        return iterator(x_data, y_data, batch_size, last_batch_start);
    }
};

#endif
