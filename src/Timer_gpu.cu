struct GpuTimer {
    cudaEvent_t start_event, stop_event;

    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        // Automatically start the clock when the timer is created
        cudaEventRecord(start_event); 
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    float elapsed() {
        // Stop the clock and calculate
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms / 1000.0f; // Return seconds to match your original timer
    }
};