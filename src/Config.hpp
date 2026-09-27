#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <cstdlib>
#include <iostream>
#include <string>

struct Config {
    std::string image_file = "data/barbara.png";
    std::string kernel_file = "kernels/lpa_kernels_m_1_0_d_16_h12_1_2_4_8_16_24_32_symmetric.txt";
    std::string output_file; // empty: <image dir>/<image name>_<version>.hdr
    float sigma_noise = 20.0f / 255.0f;
    unsigned int noise_seed = 20250910u;
    float ici_gamma = 2.0f;
};

// directory of a path, with the trailing slash
inline std::string dir_of(const std::string &path) {
    size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "" : path.substr(0, slash + 1);
}

// file placed in the same directory as path
inline std::string sibling_path(const std::string &path, const std::string &name) {
    return dir_of(path) + name;
}

inline void print_usage(const char *prog) {
    Config d;
    std::cout << "Usage: " << prog << " [options]\n"
              << "  -i, --image <file>     input image (default " << d.image_file << ")\n"
              << "  -k, --kernels <file>   LPA kernel file (default " << d.kernel_file << ")\n"
              << "  -o, --output <file>    denoised .hdr output (default <image dir>/<image name>_<version>.hdr)\n"
              << "  -s, --sigma <value>    noise standard deviation on the 0-255 scale (default 20)\n"
              << "  -g, --gamma <value>    ICI threshold (default " << d.ici_gamma << ")\n"
              << "      --seed <value>     noise seed (default " << d.noise_seed << ")\n"
              << "  -h, --help             show this message\n";
}

inline Config parse_args(int argc, char **argv, const std::string &version) {
    Config cfg;

    auto fail = [&](const std::string &msg) {
        std::cerr << "Error: " << msg << "\n\n";
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    };

    auto parse_float = [&](const std::string &opt, const char *s) {
        char *end;
        float v = std::strtof(s, &end);
        if (*s == '\0' || *end != '\0' || !(v > 0.0f)) fail(opt + " needs a positive number, got '" + s + "'");
        return v;
    };

    for (int i = 1; i < argc; i++) {
        std::string opt = argv[i];

        if (opt == "-h" || opt == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) fail("unknown option or missing value for '" + opt + "'");
        const char *val = argv[++i];

        if (opt == "-i" || opt == "--image") cfg.image_file = val;
        else if (opt == "-k" || opt == "--kernels") cfg.kernel_file = val;
        else if (opt == "-o" || opt == "--output") cfg.output_file = val;
        else if (opt == "-s" || opt == "--sigma") cfg.sigma_noise = parse_float(opt, val) / 255.0f;
        else if (opt == "-g" || opt == "--gamma") cfg.ici_gamma = parse_float(opt, val);
        else if (opt == "--seed") {
            char *end;
            unsigned long v = std::strtoul(val, &end, 10);
            if (*val == '\0' || *end != '\0') fail("--seed needs an unsigned integer, got '" + std::string(val) + "'");
            cfg.noise_seed = static_cast<unsigned int>(v);
        }
        else fail("unknown option '" + opt + "'");
    }

    if (cfg.output_file.empty()) {
        std::string name = cfg.image_file.substr(dir_of(cfg.image_file).size());
        name = name.substr(0, name.find_last_of('.'));
        cfg.output_file = dir_of(cfg.image_file) + name + "_" + version + ".hdr";
    }

    std::cout << "Image: " << cfg.image_file << "\n"
              << "Kernels: " << cfg.kernel_file << "\n"
              << "Output: " << cfg.output_file << "\n"
              << "Sigma: " << cfg.sigma_noise * 255.0f << "/255, gamma: " << cfg.ici_gamma
              << ", seed: " << cfg.noise_seed << "\n";

    return cfg;
}

#endif // CONFIG_HPP
