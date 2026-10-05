#pragma once

#include "predictive_control.hpp"

#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace heimdallr_ddspc {

struct SnapshotJob {
    std::unique_ptr<ModelSnapshot> model;
    std::int64_t transition_time_ns = 0;
    std::uint64_t sequence = 0;
    const char* trigger = "servo";
};

nlohmann::json snapshot_json(const SnapshotJob& job);
std::filesystem::path default_snapshot_root();

class SnapshotWriter {
   public:
    using ResultCallback = std::function<void(bool, const std::string&)>;

    SnapshotWriter(std::filesystem::path root, ResultCallback callback);
    ~SnapshotWriter();

    SnapshotWriter(const SnapshotWriter&) = delete;
    SnapshotWriter& operator=(const SnapshotWriter&) = delete;

    bool enqueue(SnapshotJob job);
    void stop();

   private:
    void run();
    std::filesystem::path write(const SnapshotJob& job);

    std::filesystem::path root_;
    ResultCallback callback_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<SnapshotJob> jobs_;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace heimdallr_ddspc
