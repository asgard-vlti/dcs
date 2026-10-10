#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <unistd.h>

namespace heimdallr_ft {

constexpr std::size_t kRingCapacity = 1024;
constexpr std::size_t kPendingCapacity = 32;
constexpr std::uint32_t kMaxBatch = 128;
constexpr int kServoOff = 4;

struct Sample {
    std::uint32_t seq = 0;
    std::uint16_t cnt = 0;
    int servo_mode = kServoOff;
    std::int64_t time_ns = 0;
    std::array<double, 6> gd_snr{};
    std::array<double, 6> pd_snr{};
    std::array<double, 6> gd_bl{};
    std::array<double, 4> pd_tel{};
    std::array<double, 4> gd_tel{};
    std::array<double, 4> dm_piston{};
};

struct Batch {
    std::uint64_t stream_id = 0;
    std::uint32_t oldest_seq = 0;
    std::uint32_t latest_seq = 0;
    std::uint64_t dropped_total = 0;
    bool reset = false;
    bool overrun = false;
    std::vector<Sample> rows;
};

class Ring {
   public:
    explicit Ring(std::uint64_t stream_id = 0)
        : stream_id_(stream_id == 0 ? make_stream_id() : stream_id) {}

    bool should_capture(int servo_mode,
                        std::chrono::steady_clock::time_point now) {
        if (servo_mode != kServoOff) {
            was_off_ = false;
            return true;
        }
        if (!was_off_ || now >= next_off_capture_) {
            was_off_ = true;
            next_off_capture_ = now + std::chrono::milliseconds(100);
            return true;
        }
        return false;
    }

    void publish(Sample sample) {
        sample.seq = next_seq_.fetch_add(1, std::memory_order_relaxed);
        const bool first_active = sample.servo_mode != kServoOff &&
                                  last_selected_mode_ == kServoOff;
        last_selected_mode_ = sample.servo_mode;
        flush_pending();
        if (pending_count_ == kPendingCapacity) {
            if (!first_active || !evict_pending_off()) {
                dropped_total_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        pending_[(pending_head_ + pending_count_) % kPendingCapacity] = sample;
        ++pending_count_;
        flush_pending();
    }

    void flush_pending() {
        if (pending_count_ == 0) return;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return;
        while (pending_count_ != 0) {
            if (ring_count_ == kRingCapacity) {
                ring_head_ = (ring_head_ + 1) % kRingCapacity;
                --ring_count_;
            }
            ring_[(ring_head_ + ring_count_) % kRingCapacity] =
                pending_[pending_head_];
            last_ring_seq_ = pending_[pending_head_].seq;
            ++ring_count_;
            pending_head_ = (pending_head_ + 1) % kPendingCapacity;
            --pending_count_;
        }
    }

    Batch read_since(std::uint64_t requested_stream_id,
                     std::uint32_t after_seq, std::uint32_t limit) const {
        Batch batch;
        batch.stream_id = stream_id_;
        batch.dropped_total = dropped_total_.load(std::memory_order_relaxed);
        const std::size_t count = std::min<std::size_t>(limit, kMaxBatch);
        batch.rows.reserve(count);
        std::lock_guard<std::mutex> lock(mutex_);
        batch.oldest_seq = ring_count_ == 0 ? last_ring_seq_
                                            : ring_[ring_head_].seq;
        batch.latest_seq = ring_count_ == 0
                               ? last_ring_seq_
                               : ring_[(ring_head_ + ring_count_ - 1) %
                                       kRingCapacity]
                                     .seq;
        if (limit == 0) return batch;
        batch.reset = requested_stream_id != stream_id_;
        for (std::size_t i = 0; i < ring_count_ && batch.rows.size() < count;
             ++i) {
            const Sample& row = ring_[(ring_head_ + i) % kRingCapacity];
            const std::uint32_t distance = row.seq - after_seq;
            if (batch.reset || (distance != 0 && distance < 0x80000000u)) {
                batch.rows.push_back(row);
            }
        }
        if (!batch.reset && !batch.rows.empty()) {
            batch.overrun =
                std::uint32_t(batch.rows.front().seq - after_seq) != 1;
        }
        return batch;
    }

#ifdef FT_TELEMETRY_TEST
    template <typename Fn>
    void with_read_lock_for_test(Fn&& fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        fn();
    }

    void set_next_seq_for_test(std::uint32_t seq) {
        next_seq_.store(seq);
        last_ring_seq_ = seq - 1;
    }
#endif

   private:
    bool evict_pending_off() {
        for (std::size_t offset = pending_count_; offset > 0; --offset) {
            const std::size_t index =
                (pending_head_ + offset - 1) % kPendingCapacity;
            if (pending_[index].servo_mode != kServoOff) continue;
            for (std::size_t next = offset; next < pending_count_; ++next) {
                pending_[(pending_head_ + next - 1) % kPendingCapacity] =
                    pending_[(pending_head_ + next) % kPendingCapacity];
            }
            --pending_count_;
            dropped_total_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    static std::uint64_t make_stream_id() {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        const auto id = (std::uint64_t(nanoseconds) << 1) ^
                        std::uint64_t(::getpid());
        return id == 0 ? 1 : id;
    }

    const std::uint64_t stream_id_;
    mutable std::mutex mutex_;
    std::array<Sample, kRingCapacity> ring_{};
    std::size_t ring_head_ = 0;
    std::size_t ring_count_ = 0;
    std::uint32_t last_ring_seq_ = 0;
    std::array<Sample, kPendingCapacity> pending_{};
    std::size_t pending_head_ = 0;
    std::size_t pending_count_ = 0;
    std::atomic<std::uint32_t> next_seq_{1};
    std::atomic<std::uint64_t> dropped_total_{0};
    int last_selected_mode_ = kServoOff;
    bool was_off_ = false;
    std::chrono::steady_clock::time_point next_off_capture_{};
};

}  // namespace heimdallr_ft
