#include "dataset.hpp"
#include <xtensor/views/xmasked_view.hpp>
#include <fstream>

uint32_t bswap_32(uint32_t x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}


std::vector<uint32_t> get_idx_dimension(std::ifstream& file) {
    // magic number ==> first 2 bytes 00, third byte indicate the type of data and the fourth byte indicate the number of dimensions
    // size in dimension 0 
    // size in dimension 1 
    // size in dimension 2 
    // ..... 
    // size in dimension N 
    // data
    // Read the IDX file header byte by byte
    unsigned char zeros[2];
    unsigned char data_type;
    unsigned char n_dim_byte;

    file.read(reinterpret_cast<char*>(zeros), 2);
    file.read(reinterpret_cast<char*>(&data_type), 1);
    file.read(reinterpret_cast<char*>(&n_dim_byte), 1);

    int n_dim = static_cast<int>(n_dim_byte);

    // Utiliser un vecteur pour stocker les tailles de dimensions de manière générique
    std::vector<uint32_t> dimensions;
    for (int i = 0; i < n_dim; ++i) {
        uint32_t dim_be;
        file.read(reinterpret_cast<char*>(&dim_be), sizeof(dim_be));
        uint32_t dim = bswap_32(dim_be);
        dimensions.push_back(dim);
    }

    return dimensions;
}

std::tuple<xt::xarray<float>, xt::xarray<unsigned char>> load_idx_data(const std::string& image_path, const std::string& label_path) {

    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        std::cerr << "Error: Unable to open image file " << image_path << std::endl;
        return {};
    }
    std::vector<uint32_t> image_dims = get_idx_dimension(image_file);
    if (image_dims.size() != 3) {
        std::cerr << "Error: Expected 3 dimensions for images, but got " << image_dims.size() << std::endl;
        return {};
    }

    size_t total_image_pixels = image_dims[0] * image_dims[1] * image_dims[2];
    xt::xarray<unsigned char> images_uchar(xt::adapt(std::vector<unsigned char>(total_image_pixels), image_dims));
    image_file.read(reinterpret_cast<char*>(images_uchar.data()), total_image_pixels);
    
    // Merge the rows and cols of the images 
    images_uchar = xt::reshape_view(images_uchar, {image_dims[0], image_dims[1] * image_dims[2]});

    // --- NORMALIZATION STEP ---
    xt::xarray<float> images = xt::cast<float>(images_uchar);
    images /= 255.0f;
    // -------------------------

    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Error: Unable to open label file " << label_path << std::endl;
        return {};
    }
    std::vector<uint32_t> label_dims = get_idx_dimension(label_file);
    if (label_dims.size() != 1) {
        std::cerr << "Error: Expected 1 dimension for labels, but got " << label_dims.size() << std::endl;
        return {};
    }
    size_t num_labels = label_dims[0];
    xt::xarray<unsigned char> labels(xt::adapt(std::vector<unsigned char>(num_labels), label_dims));
    label_file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return std::make_tuple(images, labels);
}


MNISTDataset::MNISTDataset(
    const std::string& training_images_path,
    const std::string& training_labels_path,
    const std::string& test_images_path,
    const std::string& test_labels_path
) {

    auto training_data = load_idx_data(training_images_path, training_labels_path);
    if (std::tuple_size<decltype(training_data)>::value != 2) {
        std::cerr << "Failed to load training data." << std::endl;
        return;
    }
    auto test_data = load_idx_data(test_images_path, test_labels_path);
    if (std::tuple_size<decltype(test_data)>::value != 2) {
        std::cerr << "Failed to load test data." << std::endl;
        return;
    }
    auto training_images = std::get<0>(training_data);
    auto training_labels = std::get<1>(training_data);
    

    auto test_images = std::get<0>(test_data);
    auto test_labels = std::get<1>(test_data);
    
    this->data = xt::concatenate(xt::xtuple(training_images, test_images), 0);
    auto raw_labels = xt::concatenate(xt::xtuple(training_labels, test_labels), 0);
    this->labels = remap_labels(raw_labels);

    std::cout << "MNIST data loaded successfully." << std::endl;
}

IrisDataset::IrisDataset(const std::string &file_path) {
std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open IRIS dataset file " << file_path << std::endl;
        return;
    }

    std::vector<std::vector<float>> features_vec;
    std::vector<unsigned int> labels_vec;

    std::unordered_map<std::string, unsigned int> label_map;
    unsigned int next_id = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::vector<float> feature_row;
        float value;
        char comma;

        for (int i = 0; i < 4; ++i) {
            if (!(ss >> value)) {
                std::cerr << "Error: Invalid data format in IRIS dataset." << std::endl;
                return;
            }
            feature_row.push_back(value);
            ss >> comma;
        }

        std::string label_str;
        if (!(ss >> label_str)) {
            std::cerr << "Error: Missing label in IRIS dataset." << std::endl;
            return;
        }

        if (label_map.find(label_str) == label_map.end()) {
            label_map[label_str] = next_id++;
        }

        features_vec.push_back(std::move(feature_row));
        labels_vec.push_back(label_map[label_str]);
    }

    std::size_t n = features_vec.size();
    std::size_t d = 4;

    // Flatten features into a contiguous buffer
    std::vector<float> flat_features;
    flat_features.reserve(n * d);
    for (const auto& row : features_vec) {
        flat_features.insert(flat_features.end(), row.begin(), row.end());
    }

    // Assign to xtensor members
    this->data   = xt::adapt(flat_features, std::array<std::size_t, 2>{n, d});
    this->labels = xt::adapt(labels_vec,    std::array<std::size_t, 1>{n});
    auto& gen = xt::random::get_default_random_engine();
    xt::random::shuffle(this->data, gen);
    xt::random::shuffle(this->labels, gen);
    std::cout << "IRIS dataset loaded successfully. "
              << "Samples: " << n << ", Features: " << d << std::endl;
}


xt::xarray<uint> remap_labels(const xt::xarray<unsigned char>& labels) {
    auto unique_labels = xt::unique(labels);

    xt::xarray<uint> remapped = xt::zeros<uint>(labels.shape());

    for (uint i = 0; i < unique_labels.size(); ++i) {
        auto mask = xt::equal(labels, unique_labels(i));
        xt::masked_view(remapped, mask) = i;
    }

    std::cout << "Remapped labels: " << remapped << std::endl;
    return remapped;
}