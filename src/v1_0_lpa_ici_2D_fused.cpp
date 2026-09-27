#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>

 #include "Timer.hpp"
 #include "Config.hpp"

// This macro tells the header to actually compile the implementation code
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image/stb_image_write.h"


// --- Function Prototypes ---

/**
  * @brief Loads the LPA kernels from a given text file.
  *
  * This function opens the specified file, reads the number of directions
  * and scales, and then loads the kernel matrices into memory.
  *
  * @param filename The path to the text file containing the kernel data.
  * @param host_kernel_weights Reference to the vector that will store the flattened 1D
  * array of all kernel values.
  * @param host_offsets Reference to the vector that will store the starting
  * index of each kernel in host_kernel_weights.
  * @param host_rows Reference to the vector that will store the number of rows
  * for each kernel.
  * @param host_cols Reference to the vector that will store the number of
  * columns for each kernel.
  * @param num_dirs Number of kernel directions
  * @param num_scales Number of kernel scales 
  */
void load_LPA_kernels(const std::string &filename,
					std::vector<float> &host_kernel_weights,
					std::vector<int> &host_offsets, std::vector<int> &host_rows,
					std::vector<int> &host_cols,
                    int &num_dirs, int &num_scales);

/**
  * @brief Loads the LPA kernels from a given text file trims the oversized
  * directions to the standard form
  *
  * This function opens the specified file, reads the number of directions
  * and scales, and then loads the kernel matrices into memory.
  * Since the matlab toolkit exports angled kernels with padding but we are using
  * the cake shaped ones, we can get rid of the padding.
  * The first direction is used as reference, if it is not the smallest one
  * function should be modified accordingly
  *
  * @param filename The path to the text file containing the kernel data.
  * @param host_kernel_weights Reference to the vector that will store the flattened 1D
  * array of all kernel values.
  * @param host_offsets Reference to the vector that will store the starting
  * index of each kernel in host_kernel_weights.
  * @param host_rows Reference to the vector that will store the number of rows
  * for each kernel.
  * @param host_cols Reference to the vector that will store the number of
  * columns for each kernel.
  * @param num_dirs Number of kernel directions
  * @param num_scales Number of kernel scales 
  */
void load_LPA_kernels_standardized(const std::string &filename,
								std::vector<float> &host_kernel_weights,
								std::vector<int> &host_offsets,
								std::vector<int> &host_rows,
								std::vector<int> &host_cols,
                                int &num_dirs, int &num_scales);

/**
 * @brief Flips all loaded kernels 180 degrees in-place.
 * 
 * This flip is necessary to match python's scipy.signal.convolve2d, which 
 * performs a 180-degree rotation of the kernel before sliding it.
 * Required to test against python.
 * Note that the convolution function will not perform any flip
 */
void flip_kernels(std::vector<float>& host_kernel_weights,
                  const std::vector<int>& host_offsets,
                  const std::vector<int>& host_rows,
                  const std::vector<int>& host_cols,
                  int num_dirs, int num_scales);

/**
  * @brief Displays the LPA kernels in a form similar to the input file
  *
  * This function prints the kernels like the input file except the first lines,
  * and only if kernels have not been trimmed.
  *
  * @param host_kernel_weights Reference to the vector that will store the flattened 1D
  * array of all kernel values.
  * @param host_offsets Reference to the vector that will store the starting
  * index of each kernel in host_kernel_weights.
  * @param host_rows Reference to the vector that will store the number of rows
  * for each kernel.
  * @param host_cols Reference to the vector that will store the number of
  * columns for each kernel.
  */
void display_LPA_kernels(std::vector<float> &host_kernel_weights,
						std::vector<int> &host_offsets,
						std::vector<int> &host_rows, std::vector<int> &host_cols,
						int num_dirs, int num_scales);

/**
  * @brief Displays the LPA kernels in a visual way similar to spy(matrix) in matlab
  *
  * @param host_kernel_weights Reference to the vector that will store the flattened 1D
  * array of all kernel values.
  * @param host_offsets Reference to the vector that will store the starting
  * index of each kernel in host_kernel_weights.
  * @param host_rows Reference to the vector that will store the number of rows
  * for each kernel.
  * @param host_cols Reference to the vector that will store the number of
  * columns for each kernel.
  */
void display_LPA_kernels_visual(std::vector<float> &host_kernel_weights,
							std::vector<int> &host_offsets,
							std::vector<int> &host_rows,
							std::vector<int> &host_cols);

/**
  * @brief Displays the LPA kernels in a visual way merging for one size to check for gaps
  * 
  */
void display_merged_LPA_cakes(const std::vector<float> &host_kernel_weights,
    const std::vector<int> &host_offsets,
    const std::vector<int> &host_rows,
    const std::vector<int> &host_cols,
    int num_dirs, int num_scales);

/**
  * @brief Pre computes the variance values for each kernel to be passed to the LPA algorithm functions
  *
  */   
void compute_kernel_variances(std::vector<float> &host_kernel_weights,
							std::vector<int> &host_offsets,
							std::vector<int> &host_rows,
							std::vector<int> &host_cols,
                            std::vector<float> &kernel_variances,
                            float sigma_noise);
								
/**
  * @brief Manually converts an RGB image to grayscale (simple average).
  *
  * Note: This was the original manual conversion.
  * If using this, stbi_load should be called with desired_channels = 0.
  *
  * @param img Pointer to the original image data.
  * @param width Image width.
  * @param height Image height.
  * @param channels Number of channels in the original image (e.g., 3 or 4).
  */
void manual_RGB_to_gray(float *img, int width, int height, int channels);

/**
  * @brief Generates a noisy image from a clean one
  *
  * This function adds white additive gaussian noise to an image from a clean reference
  *
  * @param img_clean Pointer to the original image data.
  * @param img_noisy Pointer to a clean copy of the original image data, this will be corrupted by noise.
  * @param img_size width * height * channels(1).
  * @param sigma_noise The desired standard deviation of the noise.
  * @param noise_seed Seed of the PRNG.
  */
void create_noisy_image(float* img_clean, float* img_noisy, int img_size, float sigma_noise, unsigned int noise_seed);

/**
  * @brief Calculates PSNR
  *
  * This function calculates the PSNR between two images
  *
  * @param img_one Pointer to the clean image.
  * @param img_two Pointer to the estimate or noisy image.
  * @param img_size width * height * channels(1).
  */
double calculate_PSNR(float* img_one, float* img_two, int img_size);


/**
  * @brief Performs 2D convolution and immediately updates ICI bounds
  *
  * This function calculates the convolution for each pixel and instantly 
  * evaluates the Intersection of Confidence Intervals (ICI) rule. It updates 
  * the global arrays for the current direction in-place, eliminating 
  * the need for an intermediate full-image buffer.
  *
  * @param img_noisy Pointer to the input noisy image data.
  * @param width Image width in pixels.
  * @param height Image height in pixels.
  * @param channels Number of channels (assumes 1 for grayscale).
  * @param curr_kernel Pointer to the weights of the current filter.
  * @param kernel_rows Number of rows in the kernel.
  * @param kernel_cols Number of columns in the kernel.
  * @param var_yhat_h The precomputed variance of the estimate for this scale.
  * @param ici_gamma The threshold parameter for the confidence intervals.
  * @param lower_bound Pointer to the lower bounds array for the current direction.
  * @param upper_bound Pointer to the upper bounds array for the current direction.
  * @param img_denoised_d Pointer to the valid estimates array for the current direction.
  * @param var_d Pointer to the valid variances array for the current direction.
  */
void process_scale_fused(float* img_noisy, 
                         int width, int height, int channels,
                         float* kernel, int kernel_rows, int kernel_cols,
                         float kernel_variance, float threshold,
                         float* lower_bound, float* upper_bound,
                         float* img_denoised_d,float* var_d);

/**
  * @brief Applies anisotropic LPA-ICI filtering on the CPU
  *
  * This function runs the directional LPA-ICI algorithm using flat raw pointers.
  * It loops over all directions and scales sequentially
  * immediately updates the confidence intervals avoiding unecessary 
  *
  * @param img_noisy Pointer to the input noisy image data.
  * @param img_denoised Pointer to the allocated array for the final denoised image initialized at 0.0f.
  * @param width Image width.
  * @param height Image height.
  * @param channels Number of channels (only =1 implemented here).
  * @param weights Pointer to an allocated workspace array for global normalization weights.
  * @param num_dirs Total number of directions.
  * @param num_scales Total number of scales.
  * @param host_kernel_weights Pointer to the flattened 1D array of all kernel values.
  * @param host_offsets Pointer to the starting index of each kernel in host_kernel_weights.
  * @param host_rows Pointer to the number of rows for each kernel.
  * @param host_cols Pointer to the number of columns for each kernel.
  * @param sigma_noise The standard deviation of the noise.
  * @param ici_gamma The ICI threshold parameter.
  */
void anisotropic_lpa_ici_cpu_fused(float* img_noisy, float* img_denoised,
                            int width, int height, int channels,
                            float* global_weights_buffer,
                            int num_dirs, int num_scales,
                            float* host_kernel_weights,
                            int* host_offsets,
                            int* host_rows,
                            int* host_cols,
                            float* kernel_variances,
                            float ici_gamma);






// --- MAIN ----

int main(int argc, char** argv) {

    Config cfg = parse_args(argc, argv, "v1_0_lpa_ici_2D_fused_cpu");

    // --- IMAGE LOADING and RGB->WB CONVERSION ---

    int width, height, channels;

    // stb_image internally converts the image to grayscale using the standard
    // perceptual luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    float *img_gray = stbi_loadf(cfg.image_file.c_str(), &width, &height, &channels, 1);
    // reassign correct channel since we forced 1
    channels = 1;
    if (img_gray == NULL) {
        std::cout << "Error in loading the image\n";
        return 0;
    }
    std::cout << "Loaded image with a width of " << width << " px, a height of "
                        << height << " px and (imported) " << channels << " channels\n";

	//write the imported image for checking. hdr is the only format that supports writing floats via stbi
	std::string gray_image_file = sibling_path(cfg.output_file, "gray_image.hdr");
	std::cout << "Saving gray image as " << gray_image_file << "\n";
    stbi_write_hdr(gray_image_file.c_str(), width, height, channels, img_gray);

    // if you want to load RGB and convert to gray manually uncomment below
    // float *img_rgb = stbi_loadf("data/Lena512rgb.png", &width, &height, &channels, 0);
    // manual_RGB_to_gray(img_rgb, width, height, channels);
    // stbi_image_free(img_rgb);
    // float *img_gray_manual = stbi_loadf("data/gray_image.png", &width, &height, &channels, 0);



    // --- CREATE NOISY IMAGE ---
    
    
    //allocate the new image (cast unecessary, just for clarification). Uses range constructor, pointer to first and last value
    int img_size = width * height * channels;
    std::vector<float> img_gray_noisy(img_gray, static_cast<float*>(img_gray + img_size));
    //or allocate in C style via (better for shared memory if cuda)
    //float *img_gray_noisy = (float*)malloc(img_size * sizeof(float));
    //or
    //float *img_gray_noisy = new float[img_size];

    create_noisy_image(img_gray, img_gray_noisy.data(), img_size, cfg.sigma_noise, cfg.noise_seed);
    //save the noisy image for checking
    std::string gray_image_noisy_file = sibling_path(cfg.output_file, "gray_image_noisy.hdr");
    std::cout << "Saving noisy gray image as " << gray_image_noisy_file << "\n";
    stbi_write_hdr(gray_image_noisy_file.c_str(), width, height, channels, img_gray_noisy.data());


    // -- CALCULATE PSNR ---
    
    //allocate a vector for the differences
    double psnr_noisy = calculate_PSNR(img_gray, img_gray_noisy.data(), img_size);
    std::cout << "Calculated PSNR of img_gray_noisy: " << psnr_noisy << "\n";



    // --- KERNEL IMPORT ---

    // variables for number of directions and number of scales, detected from the
    // file.
    int num_dirs, num_scales;
    // global vectors to store the kernel data in Host RAM
    std::vector<float> host_kernel_weights; // the actual filter values including zeros
    std::vector<int> host_offsets;   // index of matrices
    std::vector<int> host_rows; //filter dimention
    std::vector<int> host_cols; //filter dimention

    // load the kernels in memory
    // load_LPA_kernels(kernel_file,
    //                 host_kernel_weights,
    //                 host_offsets,
    //                 host_rows, host_cols,
    //                 num_dirs, num_scales);
    load_LPA_kernels_standardized(cfg.kernel_file,
                                host_kernel_weights,
                                host_offsets,
                                host_rows, host_cols,
                                num_dirs, num_scales);
    //flip kernels 180 to match python scipy.signal.convolve2d
    // flip_kernels(host_kernel_weights,
    //             host_offsets,
    //             host_rows, host_cols,
    //             num_dirs,
    //             num_scales);

    //calculate the variances for the kernels
    //allocate a vector
    int num_kernels = num_dirs * num_scales;
    std::vector<float> kernel_variances(num_kernels,0.0f); //filter variances for LPA confidence intervals
    
    compute_kernel_variances(host_kernel_weights,
                                    host_offsets,
                                    host_rows,
                                    host_cols,
                                    kernel_variances,
                                    cfg.sigma_noise);

    
    


    // --- KERNEL DISPLAY ---
    // display the kernels for export or visual check
    // display_LPA_kernels(host_kernel_weights, host_offsets, host_rows, host_cols,
    // num_dirs, num_scales); 
    // display_LPA_kernels_visual(host_kernel_weights,
    //                            host_offsets,
    //                            host_rows, host_cols);
    // check coverage of whole circle with this visual debug function
    // display_merged_LPA_cakes(host_kernel_weights,
    //                          host_offsets,
    //                          host_rows,
    //                          host_cols,
    //                          num_dirs, num_scales);




    // --- LPA ICI ----
    

    //initialize the empty estimate and weights
    std::vector<float> img_denoised(img_size,0.0f);
    std::vector<float> global_weights_buffer(img_size,0.0f);


    Timer t;

    anisotropic_lpa_ici_cpu_fused(img_gray_noisy.data(),img_denoised.data(),
                            width, height, channels,
                            global_weights_buffer.data(),
                            num_dirs, num_scales,
                            host_kernel_weights.data(),
                            host_offsets.data(),
                            host_rows.data(),
                            host_cols.data(),
                            kernel_variances.data(),
                            cfg.ici_gamma);

    std::cout << "Time taken for entire lpa function execution: " << t.elapsed() << " seconds\n";



    // --- CALCULATE PSNR ---
    double psnr_denoised = calculate_PSNR(img_gray, img_denoised.data(), img_size);
    std::cout << "Calculated PSNR of img_denoised: " << psnr_denoised << "\n";
                            
    // --- SAVE RESULTS ---
    std::cout << "Saving denoised image as "<< cfg.output_file.c_str() << "\n";
    stbi_write_hdr(cfg.output_file.c_str(), width, height, channels, img_denoised.data());
    



    //teardown                        
    stbi_image_free(img_gray); // Free the original image data before exiting
    return 0;
}

void load_LPA_kernels(const std::string &filename,
                                        std::vector<float> &host_kernel_weights,
                                        std::vector<int> &host_offsets, std::vector<int> &host_rows,
                                        std::vector<int> &host_cols,
                                        int &num_dirs, int &num_scales) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    file >> num_dirs >> num_scales;

    // keep track of offset and store in host_offset
    int current_offset = 0;

    for (int d = 0; d < num_dirs; ++d) {
        for (int s = 0; s < num_scales; ++s) {
            // keep track of offset and store in host_offsets
            host_offsets.push_back(current_offset);

            // read kernel row and col
            int kernel_rows, kernel_cols;
            file >> kernel_rows >> kernel_cols;
            // push to memory
            host_rows.push_back(kernel_rows);
            host_cols.push_back(kernel_cols);

            // push the weights to memory
            int num_elements = kernel_rows * kernel_cols;
            for (int i = 0; i < num_elements; i++) {
                float num;
                file >> num;
                host_kernel_weights.push_back(num);
            }

            current_offset += num_elements;
        }
    }

    std::cout << "imported kernels on vector of size " << host_kernel_weights.size() << "\n";
    file.close();
}

void load_LPA_kernels_standardized(const std::string &filename,
                                    std::vector<float> &host_kernel_weights,
                                    std::vector<int> &host_offsets,
                                    std::vector<int> &host_rows,
                                    std::vector<int> &host_cols,
                                    int &num_dirs, int &num_scales) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    file >> num_dirs >> num_scales;

    int current_offset = 0;

    // vectors to hold the standardized target sizes for each scale
    std::vector<int> target_rows(num_scales);
    std::vector<int> target_cols(num_scales);

    for (int d = 0; d < num_dirs; ++d) {
        for (int s = 0; s < num_scales; ++s) {

            int file_rows, file_cols;
            file >> file_rows >> file_cols;

            // Save the entire matrix into a temporary buffer
            int file_elements = file_rows * file_cols;
            std::vector<float> temp_buffer(file_elements);
            for (int i = 0; i < file_elements; ++i) {
                file >> temp_buffer[i];
            }

            // set the baseline target sizes at the first pass
            if (d == 0) {
                target_rows[s] = file_rows;
                target_cols[s] = file_cols;
            }

            // retrieve the target sizes for the current scale
            int t_rows = target_rows[s];
            int t_cols = target_cols[s];

            // crop offsets
            int row_offset = (file_rows - t_rows) / 2;
            int col_offset = (file_cols - t_cols) / 2;

            //check that the kernel we are trimming is not already too small
            if (row_offset < 0 || col_offset < 0) {
                std::cerr << "Error: Matrix at direction " << d << ", scale " << s 
                          << " is too small to crop to the target size!" << std::endl;
                return; 
            }

            host_offsets.push_back(current_offset);
            host_rows.push_back(t_rows);
            host_cols.push_back(t_cols);

            // save only the cropped values
            for (int r = 0; r < t_rows; ++r) {
                for (int c = 0; c < t_cols; ++c) {
                    // Map 2D coordinates back to the 1D temporary buffer
                    int temp_index = (r + row_offset) * file_cols + (c + col_offset);
                    host_kernel_weights.push_back(temp_buffer[temp_index]);
                }
            }

            // update offset using the trimmed size, not the file size
            current_offset += (t_rows * t_cols);
        }
    }
    file.close();
}

void flip_kernels(std::vector<float>& host_kernel_weights,
                  const std::vector<int>& host_offsets,
                  const std::vector<int>& host_rows,
                  const std::vector<int>& host_cols,
                  int num_dirs, int num_scales) {
    
    int total_kernels = num_dirs * num_scales;
    
    for (int i = 0; i < total_kernels; ++i) {
        int offset = host_offsets[i];
        int num_elements = host_rows[i] * host_cols[i];
        
        // A 180-degree rotation of a 2D matrix in row-major order 
        std::reverse(host_kernel_weights.begin() + offset, 
                     host_kernel_weights.begin() + offset + num_elements);
    }
}

void display_LPA_kernels(std::vector<float> &host_kernel_weights,
                                              std::vector<int> &host_offsets,
                                              std::vector<int> &host_rows, std::vector<int> &host_cols,
                                              int num_dirs, int num_scales) {

    std::cout << "Printing kernels available in memory\n";
    std::cout << num_dirs << " " << num_scales << "\n";

    int num_kernels = host_offsets.size();
    for (int k = 0; k < num_kernels; k++) {

        int curr_offset = host_offsets[k];

        int curr_rows = host_rows[k];
        int curr_cols = host_cols[k];
        // output similar to input file
        std::cout << curr_rows << " " << curr_cols << "\n";
        int num_elements = curr_rows * curr_cols;
        for (int i = 0; i < num_elements; i++) {
            std::cout << std::fixed << std::setprecision(6)
                                << host_kernel_weights[curr_offset + i] << " ";
        }
        std::cout << "\n";

        // output in matrix form
        //  std::cout << curr_rows << " " << curr_cols << "\n";

        // for(int i = 0; i<curr_rows; i++){
        //     for(int j = 0; j<curr_cols; j++){
        //         std::cout << host_kernel_weights[curr_offset + (i * curr_cols) + j] << "
        //         ";
        //     }
        //     std::cout << "\n";
        // }
    }
}

void display_LPA_kernels_visual(std::vector<float> &host_kernel_weights,
                                                          std::vector<int> &host_offsets,
                                                          std::vector<int> &host_rows,
                                                          std::vector<int> &host_cols) {

    std::cout << "Printing visual kernels available in memory\n";

    int num_kernels = host_offsets.size();
    for (int k = 0; k < num_kernels; k++) {

        int curr_offset = host_offsets[k];

        int curr_rows = host_rows[k];
        int curr_cols = host_cols[k];

        std::cout << "rows: " << curr_rows << " cols: " << curr_cols << "\n";

        for (int i = 0; i < curr_rows; i++) {
            for (int j = 0; j < curr_cols; j++) {
                float val = host_kernel_weights[curr_offset + (i * curr_cols) + j];
                if (val != 0.0f) {
                    std::cout << "\033[32mx\033[0m "; // green 'x' for non-zero values
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << "\n";
        }
    }
}

void display_merged_LPA_cakes(const std::vector<float> &host_kernel_weights,
                              const std::vector<int> &host_offsets,
                              const std::vector<int> &host_rows,
                              const std::vector<int> &host_cols,
                              int num_dirs, int num_scales) {

    std::cout << "\nPrinting merged 'full cake' kernels per scale\n";

    // Cycle through each size/scale
    for (int s = 0; s < num_scales; ++s) {
        
        // Grab the dimensions for this scale
        int base_idx = 0 * num_scales + s;
        int curr_rows = host_rows[base_idx];
        int curr_cols = host_cols[base_idx];

        std::cout << "Scale " << s << " - rows: " << curr_rows << " cols: " << curr_cols << "\n";

        // allocate cake accumulator
        std::vector<float> cake(curr_rows * curr_cols, 0.0f);

        // collect values from all directions
        for (int d = 0; d < num_dirs; ++d) {
            int k = d * num_scales + s;
            int curr_offset = host_offsets[k];

            for (int i = 0; i < curr_rows; ++i) {
                for (int j = 0; j < curr_cols; ++j) {
                    float val = host_kernel_weights[curr_offset + (i * curr_cols) + j];
                    cake[i * curr_cols + j] += val;
                }
            }
        }

        // display the accumulated cake 
        for (int i = 0; i < curr_rows; ++i) {
            for (int j = 0; j < curr_cols; ++j) {
                float val = cake[i * curr_cols + j];
                
                // Using 1e-6f instead of strict 0.0f to handle noise from summation
                if (std::abs(val) > 1e-6f) {
                    std::cout << "\033[32mx\033[0m "; // green 'x' for non-zero values
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void compute_kernel_variances(std::vector<float> &host_kernel_weights,
							std::vector<int> &host_offsets,
							std::vector<int> &host_rows,
							std::vector<int> &host_cols,
                            std::vector<float> &kernel_variances,
                            float sigma_noise){
    
    std::cout << "Precomputing kernel variances \n";

    float sigma_noise_squared = sigma_noise * sigma_noise;

    int num_kernels = host_offsets.size();
    for (int k = 0; k < num_kernels; k++) {

        int curr_offset = host_offsets[k];
        int curr_rows = host_rows[k];
        int curr_cols = host_cols[k];
        int curr_num_el = curr_rows * curr_cols;
        
        float norm_squared = 0;
        for(int i = 0; i < curr_num_el; i++){
            float val = host_kernel_weights[curr_offset + i];
            norm_squared += val * val;
        }
        kernel_variances[k] = sigma_noise_squared * norm_squared;
    }
}

void manual_RGB_to_gray(float *img, int width, int height, int channels) {
    // ignore if image is already wb
    if (img != NULL && channels >= 3) {

        // Convert the input image to gray
        size_t img_size = width * height * channels;
        int gray_channels = channels == 4 ? 2 : 1;
        size_t gray_img_size = width * height * gray_channels;

        float *gray_img = new float[gray_img_size]();

        if (gray_img == NULL) {
            printf("Unable to allocate memory for the gray image.\n");
            exit(1);
        }

        // p tracks the original image
        // pg traks the gray img
        for (float *p = img, *pg = gray_img; p != img + img_size;
                  p += channels, pg += gray_channels) {
            // add the values of red, green, blue and devide by 3.0
            *pg = (*p + *(p + 1) + *(p + 2)) / 3.0f;
            // save alpha if present
            if (channels == 4) {
                *(pg + 1) = *(p + 3);
            }
        }

        stbi_write_hdr("data/gray_image.hdr", width, height, channels, gray_img);
        std::cout
                << "The converted wb image has been saved to data/gray_image.png \n";

        delete[] gray_img;

        // ...
    } else {
        std::cout << "Image has fewer than 3 channels, rgb->wb not applied \n";
    }
}

void create_noisy_image(float* img_clean, float* img_noisy, int img_size, float sigma_noise, unsigned int noise_seed){
    
    // Mersenne twister PRNG, initialized with a fixed seed.
    std::mt19937 gen{noise_seed};

    // Values near the mean are the most likely. Standard deviation
    // affects the dispersion of generated values from the mean.
    std::normal_distribution<float> d{0.0, sigma_noise}; 
    
    float sample;
    // add noise to the image
	for (int i = 0 ; i < img_size; i++) {
        //generate the random sample (changes at each iteration)
        sample = d(gen);

		img_noisy[i] = img_clean[i] + sample; //sample is generated from distribution having correct sigma_noise
	}
}

double calculate_PSNR(float* img_one, float* img_two, int img_size){

    //allocate a vector for the differences
    std::vector<double> differences_squared(img_size,0.0f);
    
    //calculate the pixel-by-pixel differences and accumulate the squared value
    double mse_sum = 0.0;
    for (int i = 0; i<img_size; i++){

        float diff = img_one[i] - img_two[i]; //subtract

        differences_squared[i] = diff * diff; //square

        mse_sum += differences_squared[i];
    }
    
    //calculate the mse
    double mse = mse_sum / img_size;

    //return the psnr
    return 10.0 * std::log10(1.0f / (mse));
}



void process_scale_fused(float* img_noisy, 
                         int width, int height, int channels,
                         float* kernel, int kernel_rows, int kernel_cols,
                         float kernel_variance, float threshold,
                         float* lower_bounds, float* upper_bounds,
                         float* img_denoised_d, float* var_d){

    //loop over all image pixels
    for (int outRow = 0; outRow < height; outRow++){
        for(int outCol = 0; outCol < width; outCol++){

            //if the intervals have not broken for this pixel
            if(upper_bounds[outRow * width + outCol] >= lower_bounds[outRow * width + outCol]){
                
                //run the convolution
                float conv_value = 0.0f;
                //loop over all kernel_pixels
                for (int k_row = 0; k_row < kernel_rows; k_row++){
                    for (int k_col = 0; k_col < kernel_cols; k_col++){

                        //determine the coordinates of the image pixel to convolve
                        int inRow = outRow - kernel_rows/2 + k_row;
                        int inCol = outCol - kernel_cols/2 + k_col;
                        
                        //check if kernel pixel selected is inside boundaries
                        if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                            //accumulate convolution for each pixel in filter area
                            conv_value += kernel[k_row * kernel_cols + k_col] * img_noisy[inRow * width + inCol];
                    }
                }
                //update the intervals
                float lb = conv_value - threshold;
                float ub = conv_value + threshold;

                lower_bounds[outRow * width + outCol] = std::max(lower_bounds[outRow * width + outCol], lb);
                upper_bounds[outRow * width + outCol] = std::min(upper_bounds[outRow * width + outCol], ub);

                //update the image only if the bounds helf after the new convolution
                if (upper_bounds[outRow * width + outCol] >= lower_bounds[outRow * width + outCol]) {
                img_denoised_d[outRow * width + outCol] = conv_value;
                var_d[outRow * width + outCol] = kernel_variance;
                }
            }

        }
    }
}

void anisotropic_lpa_ici_cpu_fused(float* img_noisy, float* img_denoised,
                            int width, int height, int channels,
                            float* global_weights_buffer,
                            int num_dirs, int num_scales,
                            float* host_kernel_weights,
                            int* host_offsets,
                            int* host_rows,
                            int* host_cols,
                            float* kernel_variances,
                            float ici_gamma){
    std::cout << "Executing LPA ICI fused algorithm \n";
    int img_size = width * height * channels;

    // --- ALLOCATION OF RESULT VECTORS ---

    //allocate the estimate for the direction
    std::vector<float> img_denoised_d(img_size,0.0f);

    //allocate the variances for the direction
    std::vector<float> var_d(img_size,0.0f);

    //allocate the lower and upper bounds for the direction
    std::vector<float> lower_bounds(img_size, -std::numeric_limits<float>::infinity());
    std::vector<float> upper_bounds(img_size, std::numeric_limits<float>::infinity());


    Timer t;
    //loop over directions
    for(int d = 0; d < num_dirs; d++){
        // --- INITIALIZATION OF RESULT VECTORS ---

        //initialize the estimate for the direction
        std::fill(img_denoised_d.begin(),img_denoised_d.end(),0.0f);

        //initialize the variances for the direction
        std::fill(var_d.begin(), var_d.end(),0.0f);

        //initialize the lower and upper bounds for the direction
        std::fill(lower_bounds.begin(), lower_bounds.end(), -std::numeric_limits<float>::infinity());
        std::fill(upper_bounds.begin(), upper_bounds.end(), std::numeric_limits<float>::infinity());
        
        //loop over scales
        for(int s = 0; s < num_scales; s++){

            int curr_index = d * num_scales + s;
            int curr_offset = host_offsets[curr_index];

            //extract the correct filter
            int kernel_rows = host_rows[curr_index];
            int kernel_cols = host_cols[curr_index];
            float* curr_kernel = host_kernel_weights + curr_offset;
            float curr_kernel_variance = kernel_variances[curr_index];

            //calculate the ICI threshold
            float threshold = ici_gamma * std::sqrt(curr_kernel_variance);

            process_scale_fused(img_noisy,
                                width, height, channels,
                                curr_kernel,
                                kernel_rows, kernel_cols,
                                curr_kernel_variance, threshold,
                                lower_bounds.data(), upper_bounds.data(),
                                img_denoised_d.data(), var_d.data());
            

        }

        //update directional image estimate and weights
        for (int i = 0; i < img_size; i++){
            img_denoised[i] += img_denoised_d[i] * (1.0f / var_d[i]);
            global_weights_buffer [i] += 1.0f / var_d[i];
        }
    }

    // compute the final estimate
    for (int i = 0; i < img_size; i++){
            img_denoised[i] = img_denoised[i] / global_weights_buffer[i];
    }
    std::cout << "Time taken for directional loops + aggregation: " << t.elapsed() << " seconds\n";
}