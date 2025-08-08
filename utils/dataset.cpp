#include "dataset.hpp"
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

std::tuple<xt::xarray<uint32_t>, xt::xarray<unsigned char>> load_idx_data(const std::string& image_path, const std::string& label_path) {

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
    xt::xarray<unsigned char> images(xt::adapt(std::vector<unsigned char>(total_image_pixels), image_dims));
    image_file.read(reinterpret_cast<char*>(images.data()), total_image_pixels);
    
    // Merge the rows and cols of the images 
    images = xt::reshape_view(images, {image_dims[0], image_dims[1] * image_dims[2]});

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
    training_images = std::get<0>(training_data);
    training_labels = std::get<1>(training_data);
    

    test_images = std::get<0>(test_data);
    test_labels = std::get<1>(test_data);
    std::cout << "MNIST data loaded successfully." << std::endl;
}