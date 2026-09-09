// dm_telemetry_recorder.cpp


// this works if we only record the master DM signal (see commented code below for example that doesn't work when trying to record ch3 too, posting ch3 semaphore when putting turb on ch3)
// dm_master_telemetry_recorder.cpp
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <csignal>
#include <chrono>
#include <cstring> // <<== ADD THIS
#include "ImageStreamIO.h"
#include "fitsio.h"

// g++ -O3 -march=native -std=c++17 dm_telemetry_recorder.cpp -o dm_telemetry_recorder -lImageStreamIO -lcfitsio -lpthread

volatile std::sig_atomic_t running = 1;

void handle_sigint(int) {
    running = 0;
}

void save_to_fits(const std::string& filename, const std::vector<std::vector<double>>& frames, long naxes[2]) {
    fitsfile* fptr;
    int status = 0;

    fits_create_file(&fptr, filename.c_str(), &status);

    long naxis = 3;
    long naxes3D[3] = { naxes[0], naxes[1], static_cast<long>(frames.size()) };

    fits_create_img(fptr, DOUBLE_IMG, naxis, naxes3D, &status);

    std::vector<double> all_data;
    all_data.reserve(naxes3D[0] * naxes3D[1] * naxes3D[2]);
    for (const auto& frame : frames) {
        all_data.insert(all_data.end(), frame.begin(), frame.end());
    }

    fits_write_img(fptr, TDOUBLE, 1, all_data.size(), all_data.data(), &status);

    fits_close_file(fptr, &status);

    if (status) {
        fits_report_error(stderr, status);
    } else {
        std::cout << "\n[FITS] Saved file: " << filename << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <beam_id> [output_fits_path]" << std::endl;
        return 1;
    }

    int beam_id = std::stoi(argv[1]);
    std::string fits_filename = (argc == 3) ? argv[2] : "dm_master_telemetry_beam" + std::to_string(beam_id) + ".fits";

    if (fits_filename.front() != '!') {
        fits_filename = "!" + fits_filename; // cfitsio overwrite mode
    }

    // Open only the master DM shared memory (e.g., dm2)
    IMAGE dm_master;
    std::string master_name = "dm" + std::to_string(beam_id);

    if (ImageStreamIO_openIm(&dm_master, master_name.c_str()) != IMAGESTREAMIO_SUCCESS) {
        std::cerr << "[ERROR] Failed to open DM master SHM: " << master_name << std::endl;
        return 1;
    }

    long naxes[2] = { dm_master.md->size[0], dm_master.md->size[1] };
    size_t img_size = naxes[0] * naxes[1];

    std::vector<std::vector<double>> frames;
    size_t max_frames = 10000; // how many frames to record

    std::cout << "[INFO] Connected to " << master_name << std::endl;
    std::cout << "[INFO] Frame size: " << naxes[0] << "x" << naxes[1] << std::endl;
    std::cout << "[INFO] Will record " << max_frames << " frames." << std::endl;

    std::signal(SIGINT, handle_sigint);

    while (running && frames.size() < max_frames) {
        ImageStreamIO_semwait(&dm_master, 1);

        std::vector<double> frame(img_size);
        std::memcpy(frame.data(), dm_master.array.D, img_size * sizeof(double));

        frames.push_back(std::move(frame));

        std::cout << "\rFrames collected: " << frames.size() << "/" << max_frames << std::flush;
    }

    save_to_fits(fits_filename, frames, naxes);

    ImageStreamIO_closeIm(&dm_master);

    return 0;
}

// with TT disturb (posting sem on ch3) - bad reads 
// #include <iostream>
// #include <vector>
// #include <string>
// #include <csignal>
// #include <thread>
// #include <chrono>

// #include "ImageStreamIO.h"
// #include "fitsio.h"

// // g++ -O3 -march=native -std=c++17 dm_telemetry_recorder.cpp -o dm_telemetry_recorder -lImageStreamIO -lcfitsio -lpthread

// // Rolling buffer class
// struct RollingBuffer {
//     std::vector<std::vector<double>> frames;
//     size_t capacity;
//     size_t img_size;

//     RollingBuffer(size_t cap, size_t sz) : capacity(cap), img_size(sz) {}

//     void addFrame(const double* data) {
//         if (frames.size() >= capacity) {
//             frames.erase(frames.begin());
//         }
//         frames.emplace_back(data, data + img_size);
//     }
// };

// // Global flag to stop cleanly
// volatile std::sig_atomic_t running = 1;

// void handle_sigint(int) {
//     running = 0;
// }

// // Save all buffers into a single FITS file
// void save_to_fits(const std::string& filename,
//                   const RollingBuffer& buf03,
//                   const RollingBuffer& buf_master,
//                   long naxes[2])
// {
//     fitsfile* fptr;
//     int status = 0;

//     if (fits_create_file(&fptr, filename.c_str(), &status)) {
//         fits_report_error(stderr, status);
//         return;
//     }

//     auto write_extension = [&](const RollingBuffer& buf, const char* extname) {
//         long naxis = 3;
//         long naxes3D[3] = {naxes[0], naxes[1], static_cast<long>(buf.frames.size())};

//         if (fits_create_img(fptr, DOUBLE_IMG, naxis, naxes3D, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         if (fits_write_img(fptr, TDOUBLE, 1, naxes3D[0]*naxes3D[1]*naxes3D[2],
//                            (void*)buf.frames.front().data(), &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         if (fits_update_key(fptr, TSTRING, "EXTNAME", (void*)extname, nullptr, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }
//     };

//     write_extension(buf03, "DM_DISP03");
//     write_extension(buf_master, "DM_MASTER");

//     if (fits_close_file(fptr, &status)) {
//         fits_report_error(stderr, status);
//     }

//     std::cout << "\n[FITS] Saved file: " << filename << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2 && argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <beam_id> [output_fits_path]" << std::endl;
//         return 1;
//     }

//     int beam_id = std::stoi(argv[1]);

//     std::string fits_filename;
//     if (argc == 3) {
//         fits_filename = argv[2];
//         if (fits_filename.front() != '!') {
//             fits_filename = "!" + fits_filename;
//         }
//     } else {
//         fits_filename = "!dm_telemetry_beam" + std::to_string(beam_id) + ".fits";
//     }

//     // Open DM SHMs
//     IMAGE dm_disp03, dm_master;
//     std::string disp03_name = "dm" + std::to_string(beam_id) + "disp03";
//     std::string master_name = "dm" + std::to_string(beam_id);

//     if (ImageStreamIO_openIm(&dm_disp03, disp03_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_master, master_name.c_str()) != IMAGESTREAMIO_SUCCESS) 
//     {
//         std::cerr << "[ERROR] Failed to open DM SHMs." << std::endl;
//         return 1;
//     }

//     long naxes[2] = { dm_master.md[0].size[0], dm_master.md[0].size[1] };
//     size_t img_size = naxes[0] * naxes[1];
//     size_t buffer_capacity = 1000;

//     RollingBuffer buf03(buffer_capacity, img_size);
//     RollingBuffer buf_master(buffer_capacity, img_size);

//     std::cout << "[INFO] Connected to DM SHMs for beam " << beam_id << std::endl;
//     std::cout << "[INFO] Image size: " << naxes[0] << " x " << naxes[1] << std::endl;
//     std::cout << "[INFO] Output FITS file: " << fits_filename << std::endl;
//     std::cout << "[INFO] Starting acquisition..." << std::endl;

//     std::signal(SIGINT, handle_sigint);

//     while (running) {
//         // --- Safe synchronization ---
//         ImageStreamIO_semwait(&dm_disp03, 1);
//         ImageStreamIO_semwait(&dm_master, 1);

//         buf03.addFrame((double*)dm_disp03.array.D);
//         buf_master.addFrame((double*)dm_master.array.D);

//         std::cout << "\rFrames collected: " << buf03.frames.size() << "/" << buffer_capacity << std::flush;

//         if (buf03.frames.size() >= buffer_capacity) {
//             std::cout << "\n[INFO] Rolling buffer full." << std::endl;
//             break;
//         }
//     }

//     running = 0;

//     save_to_fits(fits_filename, buf03, buf_master, naxes);

//     ImageStreamIO_closeIm(&dm_disp03);
//     ImageStreamIO_closeIm(&dm_master);

//     return 0;
// }

// // dm_telemetry_recorder.cpp
// #include <iostream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <csignal>
// #include <filesystem>
// #include <atomic>

// #include "ImageStreamIO.h"
// #include "fitsio.h"

// // g++ -O3 -march=native -std=c++17 dm_telemetry_recorder.cpp -o dm_telemetry_recorder -lImageStreamIO -lcfitsio -lpthread

// // dm_telemetry_recorder.cpp
// #include <iostream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <csignal>
// #include <filesystem>

// #include "ImageStreamIO.h"
// #include "fitsio.h"

// // Compile: g++ -o dm_telemetry_recorder dm_telemetry_recorder.cpp -std=c++17 -lImageStreamIO -lcfitsio -lpthread

// // Rolling buffer class for each DM channel
// struct RollingBuffer {
//     std::vector<std::vector<float>> frames;
//     size_t capacity;
//     size_t img_size;

//     RollingBuffer(size_t capacity, size_t img_size)
//         : capacity(capacity), img_size(img_size) {}

//     void addFrame(const float* data) {
//         if (frames.size() >= capacity) {
//             frames.erase(frames.begin());
//         }
//         frames.emplace_back(data, data + img_size);
//     }
// };

// // Global flag to cleanly stop
// volatile std::sig_atomic_t running = 1;

// // Signal handler for CTRL+C
// void handle_sigint(int) {
//     running = 0;
// }

// // Helper to safely copy double* → float*
// void copy_double_to_float(const double* src, float* dst, size_t n) {
//     for (size_t i = 0; i < n; ++i) {
//         dst[i] = static_cast<float>(src[i]);
//     }
// }

// // Save the rolling buffers into a multi-extension FITS file
// void save_to_fits(const std::string& filename,
//                   const RollingBuffer& buf00,
//                   const RollingBuffer& buf01,
//                   const RollingBuffer& buf02,
//                   const RollingBuffer& buf03,
//                   const RollingBuffer& master_buf,
//                   long naxes[2]) 
// {
//     fitsfile* fptr;
//     int status = 0;

//     if (fits_create_file(&fptr, filename.c_str(), &status)) {
//         fits_report_error(stderr, status);
//         return;
//     }

//     auto write_extension = [&](const RollingBuffer& buf, const char* extname) {
//         long naxis = 3;
//         long naxes3D[3] = { static_cast<long>(naxes[0]), static_cast<long>(naxes[1]), static_cast<long>(buf.frames.size()) };

//         if (fits_create_img(fptr, FLOAT_IMG, naxis, naxes3D, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         // Write the data
//         if (fits_write_img(fptr, TFLOAT, 1, naxes3D[0] * naxes3D[1] * naxes3D[2],
//                            (void*)buf.frames.front().data(), &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         // Set extension name
//         if (fits_update_key(fptr, TSTRING, "EXTNAME", (void*)extname, nullptr, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }
//     };

//     write_extension(buf00, "DM_DISP00");
//     write_extension(buf01, "DM_DISP01");
//     write_extension(buf02, "DM_DISP02");
//     write_extension(buf03, "DM_DISP03");
//     write_extension(master_buf, "DM_MASTER");

//     if (fits_close_file(fptr, &status)) {
//         fits_report_error(stderr, status);
//     }

//     std::cout << "\n[FITS] Saved file: " << filename << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2 && argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <beam_id> [output_fits_path]" << std::endl;
//         return 1;
//     }

//     int beam_id = std::stoi(argv[1]);

//     std::string fits_filename;
//     if (argc == 3) {
//         fits_filename = argv[2];
//         if (fits_filename.front() != '!') {
//             fits_filename = "!" + fits_filename;  // overwrite flag
//         }
//     } else {
//         fits_filename = "!dm_telemetry_beam" + std::to_string(beam_id) + ".fits";
//     }

//     // Connect to DM shared memories
//     IMAGE dm_disp00, dm_disp01, dm_disp02, dm_disp03, dm_master;
//     std::string disp00_name = "dm" + std::to_string(beam_id) + "disp00";
//     std::string disp01_name = "dm" + std::to_string(beam_id) + "disp01";
//     std::string disp02_name = "dm" + std::to_string(beam_id) + "disp02";
//     std::string disp03_name = "dm" + std::to_string(beam_id) + "disp03";
//     std::string master_name = "dm" + std::to_string(beam_id);

//     if (ImageStreamIO_openIm(&dm_disp00, disp00_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp01, disp01_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp02, disp02_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp03, disp03_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_master, master_name.c_str()) != IMAGESTREAMIO_SUCCESS) 
//     {
//         std::cerr << "[ERROR] Failed to open one or more DM SHMs." << std::endl;
//         return 1;
//     }

//     long naxes[2] = { dm_master.md[0].size[0], dm_master.md[0].size[1] };
//     size_t img_size = naxes[0] * naxes[1];
//     size_t buffer_capacity = 1000; // Max frames to collect

//     RollingBuffer buf00(buffer_capacity, img_size);
//     RollingBuffer buf01(buffer_capacity, img_size);
//     RollingBuffer buf02(buffer_capacity, img_size);
//     RollingBuffer buf03(buffer_capacity, img_size);
//     RollingBuffer master_buf(buffer_capacity, img_size);

//     std::cout << "[INFO] Connected to DM SHMs for beam " << beam_id << std::endl;
//     std::cout << "[INFO] Image size: " << naxes[0] << "x" << naxes[1] << std::endl;
//     std::cout << "[INFO] Output FITS file: " << fits_filename << std::endl;
//     std::cout << "[INFO] Starting acquisition..." << std::endl;

//     std::signal(SIGINT, handle_sigint);

//     while (running) {
//         ImageStreamIO_semwait(&dm_master, 1); // Wait for new frame

//         // Wait until all memories are safe to read
//         while (dm_disp00.md->write != 0 ||
//                dm_disp01.md->write != 0 ||
//                dm_disp02.md->write != 0 ||
//                dm_disp03.md->write != 0 ||
//                dm_master.md->write != 0) 
//         {
//             std::this_thread::sleep_for(std::chrono::microseconds(500));
//         }

//         // Create temp arrays to cast safely
//         std::vector<float> frame00(img_size);
//         std::vector<float> frame01(img_size);
//         std::vector<float> frame02(img_size);
//         std::vector<float> frame03(img_size);
//         std::vector<float> frameMaster(img_size);

//         // Copy and cast from double
//         copy_double_to_float(dm_disp00.array.D, frame00.data(), img_size);
//         copy_double_to_float(dm_disp01.array.D, frame01.data(), img_size);
//         copy_double_to_float(dm_disp02.array.D, frame02.data(), img_size);
//         copy_double_to_float(dm_disp03.array.D, frame03.data(), img_size);
//         copy_double_to_float(dm_master.array.D, frameMaster.data(), img_size);

//         // Add to buffers
//         buf00.addFrame(frame00.data());
//         buf01.addFrame(frame01.data());
//         buf02.addFrame(frame02.data());
//         buf03.addFrame(frame03.data());
//         master_buf.addFrame(frameMaster.data());

//         std::cout << "\rFrames collected: " << buf00.frames.size() << "/" << buffer_capacity << std::flush;

//         if (buf00.frames.size() >= buffer_capacity) {
//             std::cout << "\n[INFO] Rolling buffer full." << std::endl;
//             break;
//         }
//     }

//     running = 0;

//     save_to_fits(fits_filename, buf00, buf01, buf02, buf03, master_buf, naxes);

//     ImageStreamIO_closeIm(&dm_disp00);
//     ImageStreamIO_closeIm(&dm_disp01);
//     ImageStreamIO_closeIm(&dm_disp02);
//     ImageStreamIO_closeIm(&dm_disp03);
//     ImageStreamIO_closeIm(&dm_master);

//     return 0;
// }


// // dm_telemetry_recorder.cpp
// #include <iostream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <csignal>
// #include <filesystem>

// #include "ImageStreamIO.h"
// #include "fitsio.h"

// // g++ -o dm_telemetry_recorder dm_telemetry_recorder.cpp -std=c++17 -lImageStreamIO -lcfitsio -lpthread


// // Rolling buffer class for each DM channel
// struct RollingBuffer {
//     std::vector<std::vector<float>> frames;
//     size_t capacity;
//     size_t img_size;

//     RollingBuffer(size_t capacity, size_t img_size)
//         : capacity(capacity), img_size(img_size) {}

//     void addFrame(const float* data) {
//         if (frames.size() >= capacity) {
//             frames.erase(frames.begin());
//         }
//         frames.emplace_back(data, data + img_size);
//     }
// };

// // Global flag to cleanly stop
// volatile std::sig_atomic_t running = 1;

// // Signal handler for CTRL+C
// void handle_sigint(int) {
//     running = 0;
// }

// void save_to_fits(const std::string& filename,
//                   const RollingBuffer& buf00,
//                   const RollingBuffer& buf01,
//                   const RollingBuffer& buf02,
//                   const RollingBuffer& buf03,
//                   const RollingBuffer& master_buf,
//                   long naxes[2]) 
// {
//     fitsfile* fptr;
//     int status = 0;

//     if (fits_create_file(&fptr, filename.c_str(), &status)) {
//         fits_report_error(stderr, status);
//         return;
//     }

//     auto write_extension = [&](const RollingBuffer& buf, const char* extname) {
//         long naxis = 3;
//         long naxes3D[3] = { static_cast<long>(naxes[0]), static_cast<long>(naxes[1]), static_cast<long>(buf.frames.size()) };

//         if (fits_create_img(fptr, FLOAT_IMG, naxis, naxes3D, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         if (fits_write_img(fptr, TFLOAT, 1, naxes3D[0]*naxes3D[1]*naxes3D[2], (void*)buf.frames.front().data(), &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }

//         if (fits_update_key(fptr, TSTRING, "EXTNAME", (void*)extname, nullptr, &status)) {
//             fits_report_error(stderr, status);
//             return;
//         }
//     };

//     write_extension(buf00, "DM_DISP00");
//     write_extension(buf01, "DM_DISP01");
//     write_extension(buf02, "DM_DISP02");
//     write_extension(buf03, "DM_DISP03");
//     write_extension(master_buf, "DM_MASTER");

//     if (fits_close_file(fptr, &status)) {
//         fits_report_error(stderr, status);
//     }

//     std::cout << "\n[FITS] Saved file: " << filename << std::endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2 && argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <beam_id> [output_fits_path]" << std::endl;
//         return 1;
//     }

//     int beam_id = std::stoi(argv[1]);

//     std::string fits_filename;
//     if (argc == 3) {
//         fits_filename = argv[2];
//         if (fits_filename.front() != '!') {
//             fits_filename = "!" + fits_filename;  // overwrite flag for CFITSIO
//         }
//     } else {
//         fits_filename = "!dm_telemetry_beam" + std::to_string(beam_id) + ".fits";
//     }

//     // Connect to DM SHMs
//     IMAGE dm_disp00, dm_disp01, dm_disp02, dm_disp03, dm_master;
//     std::string disp00_name = "dm" + std::to_string(beam_id) + "disp00";
//     std::string disp01_name = "dm" + std::to_string(beam_id) + "disp01";
//     std::string disp02_name = "dm" + std::to_string(beam_id) + "disp02";
//     std::string disp03_name = "dm" + std::to_string(beam_id) + "disp03";
//     std::string master_name = "dm" + std::to_string(beam_id);

//     if (ImageStreamIO_openIm(&dm_disp00, disp00_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp01, disp01_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp02, disp02_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_disp03, disp03_name.c_str()) != IMAGESTREAMIO_SUCCESS ||
//         ImageStreamIO_openIm(&dm_master, master_name.c_str()) != IMAGESTREAMIO_SUCCESS) 
//     {
//         std::cerr << "[ERROR] Failed to open one or more DM SHMs." << std::endl;
//         return 1;
//     }

//     long naxes[2] = { dm_master.md[0].size[0], dm_master.md[0].size[1] };
//     size_t img_size = naxes[0] * naxes[1];
//     size_t buffer_capacity = 1000; // Default buffer capacity

//     RollingBuffer buf00(buffer_capacity, img_size);
//     RollingBuffer buf01(buffer_capacity, img_size);
//     RollingBuffer buf02(buffer_capacity, img_size);
//     RollingBuffer buf03(buffer_capacity, img_size);
//     RollingBuffer master_buf(buffer_capacity, img_size);

//     std::cout << "[INFO] Connected to DM SHMs for beam " << beam_id << std::endl;
//     std::cout << "[INFO] Image size: " << naxes[0] << "x" << naxes[1] << std::endl;
//     std::cout << "[INFO] Output FITS file: " << fits_filename << std::endl;
//     std::cout << "[INFO] Starting acquisition..." << std::endl;

//     std::signal(SIGINT, handle_sigint);

//     while (running) {
//         ImageStreamIO_semwait(&dm_master, 1);
//         std::this_thread::sleep_for(std::chrono::microseconds(500));
//         // while (dm_master.md->write != 0) {
//         //     std::this_thread::sleep_for(std::chrono::microseconds(500)); // very short wait
//         // }


//         buf00.addFrame((float*)dm_disp00.array.D);
//         buf01.addFrame((float*)dm_disp01.array.D);
//         buf02.addFrame((float*)dm_disp02.array.D);
//         buf03.addFrame((float*)dm_disp03.array.D);
//         master_buf.addFrame((float*)dm_master.array.D);

//         std::cout << "\rFrames collected: " << buf00.frames.size() << "/" << buffer_capacity << std::flush;

//         if (buf00.frames.size() >= buffer_capacity) {
//             std::cout << "\n[INFO] Rolling buffer full." << std::endl;
//             break;
//         }
//     }

//     running = 0;

//     save_to_fits(fits_filename, buf00, buf01, buf02, buf03, master_buf, naxes);

//     ImageStreamIO_closeIm(&dm_disp00);
//     ImageStreamIO_closeIm(&dm_disp01);
//     ImageStreamIO_closeIm(&dm_disp02);
//     ImageStreamIO_closeIm(&dm_disp03);
//     ImageStreamIO_closeIm(&dm_master);

//     return 0;
// }