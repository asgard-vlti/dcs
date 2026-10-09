#include "ddspc_snapshot.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace heimdallr_ddspc {
namespace {

std::string system_error(const char* action) {
    return std::string(action) + ": " + std::strerror(errno);
}

std::tm utc_time(std::int64_t nanoseconds) {
    const std::time_t seconds = nanoseconds / 1000000000;
    std::tm utc{};
    if (!gmtime_r(&seconds, &utc)) {
        throw std::runtime_error("Could not convert snapshot time to UTC");
    }
    return utc;
}

std::string utc_stamp(std::int64_t nanoseconds, const char* format) {
    const std::tm utc = utc_time(nanoseconds);
    char buffer[64];
    const int length = std::snprintf(
        buffer, sizeof(buffer), format, utc.tm_year + 1900, utc.tm_mon + 1,
        utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec,
        static_cast<long long>((nanoseconds / 1000000) % 1000));
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer)) {
        throw std::runtime_error("Snapshot timestamp is too long");
    }
    return buffer;
}

nlohmann::json number_json(double value, bool& nonfinite) {
    if (std::isnan(value)) {
        nonfinite = true;
        return "NaN";
    }
    if (std::isinf(value)) {
        nonfinite = true;
        return value > 0.0 ? "Infinity" : "-Infinity";
    }
    return value;
}

template <typename Matrix>
nlohmann::json matrix_json(const Matrix& matrix, bool& nonfinite) {
    nlohmann::json rows = nlohmann::json::array();
    for (int i = 0; i < matrix.rows(); ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(number_json(matrix(i, j), nonfinite));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void write_all(int fd, const std::string& data) {
    std::size_t written = 0;
    while (written < data.size()) {
        const ssize_t count = ::write(fd, data.data() + written,
                                      data.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error(system_error("Snapshot write failed"));
        written += static_cast<std::size_t>(count);
    }
}

}  // namespace

nlohmann::json snapshot_json(const SnapshotJob& job) {
    if (!job.model) throw std::invalid_argument("Snapshot has no model");
    const auto& model = *job.model;
    bool nonfinite = false;
    nlohmann::json factor = nlohmann::json::array();
    for (int i = 0; i < ModelSnapshot::Controller::Features; ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < ModelSnapshot::Controller::Features; ++j) {
            row.push_back(number_json(
                model.factor[i * ModelSnapshot::Controller::Features + j],
                nonfinite));
        }
        factor.push_back(std::move(row));
    }

    const auto& p = model.parameters;
    nlohmann::json result = {
        {"schema_version", 1},
        {"transition",
         {{"from", "ddspc"},
          {"to", "off"},
          {"trigger", job.trigger},
          {"sequence", job.sequence},
          {"time_utc", utc_stamp(job.transition_time_ns,
                                  "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ")}}},
        {"model",
         {{"source", model.source},
          {"time_utc",
           model.model_time_ns == 0
               ? nlohmann::json(nullptr)
               : nlohmann::json(utc_stamp(
                     model.model_time_ns,
                     "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ"))},
          {"trained", model.trained},
          {"iterations", model.iterations},
          {"rls_updates", model.rls_updates},
          {"exploration_frames", model.exploration_frames},
          {"regularization", number_json(model.regularization, nonfinite)},
          {"frozen", model.frozen},
          {"freeze_reason", model.freeze_reason
                                ? nlohmann::json(model.freeze_reason)
                                : nlohmann::json(nullptr)},
          {"freeze_frame", model.frozen
                               ? nlohmann::json(model.freeze_frame)
                               : nlohmann::json(nullptr)},
          {"freeze_time_utc", model.frozen
                                  ? nlohmann::json(utc_stamp(
                                        model.freeze_time_ns,
                                        "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ"))
                                  : nlohmann::json(nullptr)}}},
        {"dimensions",
         {{"history", ModelSnapshot::Controller::HistorySamples},
          {"future", ModelSnapshot::Controller::FutureSamples},
          {"features", ModelSnapshot::Controller::Features},
          {"outputs", ModelSnapshot::Controller::Outputs},
          {"control_features", ModelSnapshot::Controller::ControlFeatures}}},
        {"parameters",
         {{"reg_start", number_json(p.reg_start, nonfinite)},
          {"reg_cutoff", number_json(p.reg_cutoff, nonfinite)},
          {"reg_divisor", number_json(p.reg_divisor, nonfinite)},
          {"reg_interval", p.reg_interval},
          {"n_exploration", p.n_exploration},
          {"exploration_sigma", number_json(p.exploration_sigma, nonfinite)},
          {"gamma", number_json(p.gamma, nonfinite)},
          {"continue_learning", p.continue_learning}}},
        {"rls",
         {{"initial_covariance", number_json(model.initial_covariance, nonfinite)},
          {"forgetting_factor", number_json(p.gamma, nonfinite)},
          {"R", std::move(factor)},
          {"weights", matrix_json(model.weights, nonfinite)}}},
        {"inverse", model.trained ? matrix_json(model.inverse, nonfinite)
                                    : nlohmann::json(nullptr)},
        {"predictive", model.trained ? matrix_json(model.predictive, nonfinite)
                                       : nlohmann::json(nullptr)},
    };
    result["nonfinite_values"] = nonfinite;
    return result;
}

std::filesystem::path default_snapshot_root() {
#ifdef SIMULATE
    const char* home = std::getenv("HOME");
    if (!home || !*home) {
        throw std::runtime_error("HOME is required for simulated DDSPC snapshots");
    }
    return std::filesystem::path(home) / "Documents/0projects/asgard/sim-data";
#else
    return "/data";
#endif
}

SnapshotWriter::SnapshotWriter(std::filesystem::path root,
                               ResultCallback callback)
    : root_(std::move(root)), callback_(std::move(callback)),
      thread_(&SnapshotWriter::run, this) {}

SnapshotWriter::~SnapshotWriter() { stop(); }

bool SnapshotWriter::enqueue(SnapshotJob job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return false;
        jobs_.push_back(std::move(job));
    }
    ready_.notify_one();
    return true;
}

void SnapshotWriter::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
}

void SnapshotWriter::run() {
    for (;;) {
        SnapshotJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
            if (jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        bool success = false;
        std::string message;
        try {
            message = write(job).string();
            success = true;
        } catch (const std::exception& e) {
            message = e.what();
        }
        try {
            callback_(success, message);
        } catch (...) {
            // A logging failure must not stop later snapshots.
        }
    }
}

std::filesystem::path SnapshotWriter::write(const SnapshotJob& job) {
    const std::string stamp = utc_stamp(
        job.transition_time_ns, "%04d%02d%02dT%02d:%02d:%02d.%03lld");
    const auto directory = root_ / stamp.substr(0, 8);
    std::filesystem::create_directories(directory);
    const std::string base = "ddspc_" + stamp.substr(8) + "_" +
                             std::to_string(getpid()) + "_" +
                             std::to_string(job.sequence);
    const std::string data = snapshot_json(job).dump(2) + "\n";

    std::filesystem::path temporary;
    int fd = -1;
    for (int suffix = 0; suffix < 1000; ++suffix) {
        temporary = directory / ("." + base + "_" + std::to_string(suffix) + ".tmp");
        fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) break;
        if (errno != EEXIST) throw std::runtime_error(system_error("Snapshot open failed"));
    }
    if (fd < 0) throw std::runtime_error("Snapshot temporary filenames exhausted");

    try {
        write_all(fd, data);
        if (::fsync(fd) != 0) throw std::runtime_error(system_error("Snapshot fsync failed"));
        if (::close(fd) != 0) {
            fd = -1;
            throw std::runtime_error(system_error("Snapshot close failed"));
        }
        fd = -1;
        std::filesystem::path final;
        bool published = false;
        for (int suffix = 0; suffix < 1000; ++suffix) {
            final = directory / (base + (suffix ? "_" + std::to_string(suffix) : "") + ".json");
            if (::link(temporary.c_str(), final.c_str()) == 0) {
                published = true;
                break;
            }
            if (errno != EEXIST) throw std::runtime_error(system_error("Snapshot publish failed"));
        }
        if (!published) throw std::runtime_error("Snapshot filenames exhausted");
        ::unlink(temporary.c_str());
        const int dir_fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
        if (dir_fd < 0) throw std::runtime_error(system_error("Snapshot directory open failed"));
        const int sync_result = ::fsync(dir_fd);
        const int sync_error = errno;
        ::close(dir_fd);
        if (sync_result != 0) {
            errno = sync_error;
            throw std::runtime_error(system_error("Snapshot directory fsync failed"));
        }
        return final;
    } catch (...) {
        if (fd >= 0) ::close(fd);
        ::unlink(temporary.c_str());
        throw;
    }
}

}  // namespace heimdallr_ddspc
