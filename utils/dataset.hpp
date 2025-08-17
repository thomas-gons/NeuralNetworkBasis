#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include "../common.hpp"

struct Subset {
    xt::xarray<float> data;
    xt::xarray<uint> labels;
};
struct DatasetSplit {
    Subset train;
    Subset val;
    Subset test;
};

class Dataset {
public:
    xt::xarray<float> data;
    xt::xarray<uint> labels;
    
    virtual ~Dataset() = default;
    DatasetSplit split_dataset(float val_ratio = 0.2f, float test_ratio = 0.1f) {
        std::size_t total_samples = data.shape()[0];
        std::size_t val_size = static_cast<std::size_t>(total_samples * val_ratio);
        std::size_t test_size = static_cast<std::size_t>(total_samples * test_ratio);
        std::size_t train_size = total_samples - val_size - test_size;
        std::cout << "Splitting dataset: "
                  << "Train: " << train_size 
                  << ", Val: " << val_size 
                  << ", Test: " << test_size 
                  << std::endl;
                  
        Subset train_subset;
        train_subset.data = xt::view(data, xt::range(0, train_size), xt::all());
        train_subset.labels = xt::view(labels, xt::range(0, train_size), xt::all());

        Subset val_subset;
        val_subset.data = xt::view(data, xt::range(train_size, train_size + val_size), xt::all());
        val_subset.labels = xt::view(labels, xt::range(train_size, train_size + val_size), xt::all());

        Subset test_subset;
        test_subset.data = xt::view(data, xt::range(train_size + val_size, total_samples), xt::all());
        test_subset.labels = xt::view(labels, xt::range(train_size + val_size, total_samples), xt::all());

        return DatasetSplit{
            .train = train_subset,
            .val = val_subset,
            .test = test_subset
        };
    }
};

class MNISTDataset: public Dataset {
public:
    MNISTDataset(
        const std::string& training_images_path,
        const std::string& training_labels_path,
        const std::string& test_images_path,
        const std::string& test_labels_path
    );

};

class IrisDataset: public Dataset {
public:
    IrisDataset(const std::string& file_path);
};

std::tuple<xt::xarray<float>, xt::xarray<unsigned char>> load_idx_data(const std::string& image_path, const std::string& label_path);
xt::xarray<uint> remap_labels(const xt::xarray<unsigned char>& labels);

#endif