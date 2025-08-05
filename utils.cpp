#include "common.hpp"
#include <fstream>
#include <iostream>

std::string xarray_shape(const xt::xarray<float>& arr) {

    std::string shape = "(";
    for (size_t i = 0; i < arr.shape().size(); i++) {
        shape += std::to_string(arr.shape()[i]);
        if (i < arr.shape().size() - 1) {
            shape += ", ";
        }
    }
    shape += ")";
    return shape;
}

nlohmann::json load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Erreur: Impossible d'ouvrir le fichier " << path << std::endl;
        return nullptr; // Retourne un objet JSON nul en cas d'erreur
    }
    
    // Créez un objet JSON à partir du flux de fichier
    nlohmann::json data;
    try {
        file >> data;
    } catch (nlohmann::json::parse_error& e) {
        std::cerr << "Erreur de parsing JSON: " << e.what() << std::endl;
        return nullptr;
    }

    return data;
}
