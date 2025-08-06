#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include "../common.hpp"

class Dataset {
public:
    virtual ~Dataset() = default;

};

class MNISTDataset {

public:
    xt::xarray<unsigned char> training_images;
    xt::xarray<unsigned char> training_labels;
    xt::xarray<unsigned char> test_images;
    xt::xarray<unsigned char> test_labels;

    MNISTDataset(
        const std::string& training_images_path,
        const std::string& training_labels_path,
        const std::string& test_images_path,
        const std::string& test_labels_path
    );

};



std::tuple<xt::xarray<uint32_t>, xt::xarray<unsigned char>> load_idx_data(const std::string& image_path, const std::string& label_path);

#endif