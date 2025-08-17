#include "common.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>


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

std::string print_xarray(const xt::xarray<float>& arr) {
    std::stringstream ss;
    ss << arr; 
    return ss.str();
}

nlohmann::json load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Erreur: Impossible d'ouvrir le fichier " << path << std::endl;
        return nullptr;
    }
    
    nlohmann::json data;
    try {
        file >> data;
    } catch (nlohmann::json::parse_error& e) {
        std::cerr << "Erreur de parsing JSON: " << e.what() << std::endl;
        return nullptr;
    }

    return data;
}
