// latency_test.cpp

#include "../baldr.h"             // for IMAGE, updateDMSharedMemory
#include <ImageStreamIO.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <getopt.h>

using json = nlohmann::json;

// g++ latency_test.cpp -std=c++17 -I/usr/include/eigen3 -lImageStreamIO -pthread -lrt -o latency_test
void updateDMSharedMemory(IMAGE &dmImg, const Eigen::VectorXd &dmCmd) {
    auto *md = dmImg.md;
    int N = md->nelement;

    // Entry log
    //std::cout << "[DBG] updateDMSharedMemory() called with dmCmd.size() = "
    //          << dmCmd.size() << " (expecting " << N << ")\n";

    //  Check size
    if ((int)dmCmd.size() != N) {
        std::cerr << "[ERROR] size mismatch: got " << dmCmd.size()
                  << " vs " << N << "\n";
        return;
    }

    //  Log a few sample values from the command
    //std::cout << "[DBG] dmCmd first 5 vals: ";
    //for (int i = 0; i < std::min(5, N); ++i) std::cout << dmCmd[i] << " ";
    //std::cout << "\n";

    //  Print metadata before write
    //std::cout << "[DBG] Before write: cnt0=" << md->cnt0
    //          << "  cnt1=" << md->cnt1
    //          << "  write=" << md->write << "\n";

    //  Mark write
    md->write = 1;
    std::atomic_thread_fence(std::memory_order_release);

    // Copy the data
    std::memcpy(dmImg.array.D, dmCmd.data(), N * sizeof(double));

    //  Bump counters
    md->cnt0++;
    md->cnt1++; //!!! MJI, not sure why this was set to 0.

    // Post semaphore
    int ret = ImageStreamIO_sempost(&dmImg, -1);
    //std::cout << "[DBG] ImageStreamIO_sempost returned " << ret << "\n";

    //  Clear write flag
    md->write = 0;

    //  Print metadata after write
    //std::cout << "[DBG] After write: cnt0=" << md->cnt0
    //          << "  cnt1=" << md->cnt1
    //          << "  write=" << md->write << "\n";

    //  Peek at shared memory contents
    //std::cout << "[DBG] dmImg.array.D first 10 vals: ";
    //for (int i = 0; i < std::min(10, N); ++i) std::cout << dmImg.array.D[i] << " ";
    //std::cout << "\n";
}


int main(int argc, char** argv) {
    // Default parameters ────────────────────────────────
    int    beam_id        = 1;
    int    poke_interval  = 100;        // frames on/off
    double poke_amp       = 0.1;      // DM units
    double duration_s     = 5.0;      // total test time in seconds

    //  Parse CLI args ───────────────────────────────────
    static struct option long_opts[] = {
        {"beam_id",       required_argument, 0, 'b'},
        {"poke_interval", required_argument, 0, 'n'},
        {"poke_amp",      required_argument, 0, 'a'},
        {"duration_s",    required_argument, 0, 'd'},
        {0,0,0,0}
    };
    int opt, idx;
    while ((opt = getopt_long(argc, argv, "b:n:a:d:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'b': beam_id       = std::stoi(optarg); break;
            case 'n': poke_interval = std::stoi(optarg); break;
            case 'a': poke_amp      = std::stod(optarg); break;
            case 'd': duration_s    = std::stod(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [--beam_id N] [--poke_interval N]"
                             " [--poke_amp X] [--duration_s X]\n";
                return 1;
        }
    }

    // ── Open shared‑memory images ─────────────────────────
    std::cout << "open shared memory" << std::endl;
    IMAGE subarray{}, dm_rtc{}, dm_rtc0{};
    std::string subname = "baldr" + std::to_string(beam_id);
    if (ImageStreamIO_openIm(&subarray, subname.c_str()) != IMAGESTREAMIO_SUCCESS) {
        std::cerr << "Failed to open subarray SHM '" << subname << "'\n";
        return 1;
    }
    std::string dmname  = "dm"   + std::to_string(beam_id) + "disp02";
    std::string dmname0 = "dm"   + std::to_string(beam_id);
    if (ImageStreamIO_openIm(&dm_rtc,  dmname.c_str())  != IMAGESTREAMIO_SUCCESS ||
        ImageStreamIO_openIm(&dm_rtc0, dmname0.c_str()) != IMAGESTREAMIO_SUCCESS)
    {
        std::cerr << "Failed to open DM SHM\n";
        return 1;
    }

    // semaphore index for frames
    int semid = ImageStreamIO_getsemwaitindex(&subarray, 0);

    // image dimensions
    int totalPixels = subarray.md->nelement;

    // build cross indices for DM shape
    int nx = dm_rtc.md->size[0];
    int ny = (dm_rtc.md->naxis>1 ? dm_rtc.md->size[1] : 1);
    int N  = dm_rtc.md->nelement;
    int cx = nx/2, cy = ny/2;
    std::vector<int> crossIdx;
    crossIdx.reserve(nx + ny);
    for (int x = 0; x < nx; ++x) crossIdx.push_back(cy * nx + x);
    for (int y = 0; y < ny; ++y) crossIdx.push_back(y  * nx + cx);

    // prepare DM command vector
    Eigen::VectorXd dmCmd = Eigen::VectorXd::Zero(N);

    // ─ Telemetry buffers ───────────
    std::vector<double>              timestamps;
    std::vector<double>              poke_vals;
    std::vector<std::vector<double>> images;

    // ───  Main loop @ ~1 kHz
    auto start_time = std::chrono::steady_clock::now();
    auto next_tick  = start_time;
    //const auto loop_time = std::chrono::microseconds(500);
    //--to run slow as test
    //const auto loop_time = std::chrono::milliseconds(500);
    //duration_s = poke_interval * 2;   // 20*2 = 40.0 seconds
    int frameCount = 0;
    
    std::cout << "starting while loop" << std::endl;
    while (true) {
        // exit after duration_s
        auto now = std::chrono::steady_clock::now();
        if (now - start_time > std::chrono::duration<double>(duration_s))
            break;
        //std::cout << "looping wait new frame" << std::endl;
        // wait for new frame
        ImageStreamIO_semwait(&subarray, semid);
        //std::cout << "looping got new frame" << std::endl;
        // timestamp ms
        double t_ms = std::chrono::duration<double,std::milli>(
                        now - start_time).count();
        timestamps.push_back(t_ms);

        // grab image
        
        int32_t* raw = subarray.array.SI32; //.UI16;
        images.emplace_back(raw, raw + totalPixels);

        // determine on/off phase
        bool onPhase = ((frameCount / poke_interval) % 2) == 0;
        dmCmd.setZero();
        if (onPhase) {
            //std::cout << "poke here" << std::endl;
            for (int idx : crossIdx) dmCmd(idx) = poke_amp;
        }

        // apply poke and notify
        updateDMSharedMemory(dm_rtc, dmCmd);
        ImageStreamIO_sempost(&dm_rtc0, -1);

        poke_vals.push_back(onPhase ? poke_amp : 0.0);
        frameCount++;

        // throttle to 1kHz
        //next_tick += loop_time;
        //std::this_thread::sleep_until(next_tick);
    }

    // ───Save telemetry..
    json j;
    j["timestamps_ms"] = timestamps;
    j["poke_values"]   = poke_vals;
    j["images"]        = images;

    std::ostringstream oss;
    oss << "/home/asg/Music/latency_telem_beam" << beam_id << ".json";

    std::ofstream ofs(oss.str());
    ofs << j.dump(2) << std::endl;
    std::cout << "Wrote telemetry to /home/asg/Music/latency_telem_beam#.json\n";

    return 0;
}
