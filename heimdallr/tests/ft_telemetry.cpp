#include "../ft_telemetry.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>

using heimdallr_ft::Ring;
using heimdallr_ft::Sample;

int main() {
    const auto start = std::chrono::steady_clock::time_point{};
    Ring transition(11);
    assert(transition.should_capture(heimdallr_ft::kServoOff, start));
    transition.publish(Sample{});
    const auto cursor = transition.read_since(0, 0, 0);
    assert(cursor.latest_seq == 1);
    for (int i = 0; i < 410; ++i) {
        const auto frame_time = start + std::chrono::microseconds(250 * i);
        assert(transition.should_capture(1, frame_time));
        Sample row;
        row.servo_mode = 1;
        row.cnt = i + 2;
        transition.publish(row);
    }
    std::uint32_t after = cursor.latest_seq;
    int received = 0;
    while (received < 410) {
        const auto batch = transition.read_since(cursor.stream_id, after, 128);
        assert(!batch.reset && !batch.overrun);
        assert(!batch.rows.empty() && batch.rows.size() <= 128);
        for (const auto& row : batch.rows) {
            assert(row.seq == std::uint32_t(received + 2));
            assert(row.cnt == received + 2);
            after = row.seq;
            ++received;
        }
    }
    assert(received == 410);
    assert(transition.should_capture(heimdallr_ft::kServoOff,
                                     start + std::chrono::milliseconds(103)));
    assert(!transition.should_capture(heimdallr_ft::kServoOff,
                                      start + std::chrono::milliseconds(104)));

    Ring contention(12);
    contention.with_read_lock_for_test([&] {
        for (int i = 0; i < 32; ++i) contention.publish(Sample{});
        Sample first_active;
        first_active.servo_mode = 1;
        contention.publish(first_active);
        first_active.cnt = 34;
        contention.publish(first_active);
    });
    contention.flush_pending();
    const auto pending = contention.read_since(12, 0, 128);
    assert(pending.dropped_total == 2);
    assert(!pending.overrun);
    assert(pending.rows.size() == 32);
    assert(pending.rows.front().seq == 1);
    assert(pending.rows[30].seq == 31);
    assert(pending.rows[31].seq == 33);
    assert(pending.rows[31].servo_mode == 1);

    Ring active_contention(13);
    active_contention.with_read_lock_for_test([&] {
        Sample active;
        active.servo_mode = 1;
        for (int i = 0; i < 33; ++i) active_contention.publish(active);
    });
    active_contention.flush_pending();
    const auto retained = active_contention.read_since(13, 0, 128);
    assert(retained.rows.size() == 32);
    assert(retained.rows.front().seq == 1);
    assert(retained.rows.back().seq == 32);
    assert(retained.dropped_total == 1);

    Ring mixed_contention(18);
    mixed_contention.with_read_lock_for_test([&] {
        Sample active;
        active.servo_mode = 1;
        for (int i = 0; i < 31; ++i) mixed_contention.publish(active);
        mixed_contention.publish(Sample{});
        mixed_contention.publish(active);
    });
    mixed_contention.flush_pending();
    const auto mixed = mixed_contention.read_since(18, 0, 128);
    assert(mixed.dropped_total == 1);
    assert(mixed.rows.size() == 32);
    assert(mixed.rows[30].seq == 31);
    assert(mixed.rows[31].seq == 33);
    assert(mixed.rows[31].servo_mode == 1);

    Ring pending_handshake(17);
    pending_handshake.with_read_lock_for_test(
        [&] { pending_handshake.publish(Sample{}); });
    assert(pending_handshake.read_since(0, 0, 0).latest_seq == 0);
    pending_handshake.flush_pending();
    assert(pending_handshake.read_since(17, 0, 128).rows.front().seq == 1);

    Ring overflow(14);
    for (int i = 0; i < 1030; ++i) overflow.publish(Sample{});
    const auto overwritten = overflow.read_since(14, 0, 128);
    assert(overwritten.overrun);
    assert(overwritten.oldest_seq == 7);
    assert(overwritten.latest_seq == 1030);
    assert(overwritten.rows.front().seq == 7);

    Ring wrapped(15);
    wrapped.set_next_seq_for_test(0xfffffffeu);
    const auto wrap_cursor = wrapped.read_since(0, 0, 0);
    assert(wrap_cursor.latest_seq == 0xfffffffdu);
    for (int i = 0; i < 4; ++i) wrapped.publish(Sample{});
    const auto wrap_batch = wrapped.read_since(15, wrap_cursor.latest_seq, 128);
    assert(!wrap_batch.overrun && wrap_batch.rows.size() == 4);
    assert(wrap_batch.rows[0].seq == 0xfffffffeu);
    assert(wrap_batch.rows[1].seq == 0xffffffffu);
    assert(wrap_batch.rows[2].seq == 0);
    assert(wrap_batch.rows[3].seq == 1);
    const auto restarted = wrapped.read_since(14, 100, 128);
    assert(restarted.reset && restarted.rows.size() == 4);

    Ring sustained(19);
    std::uint32_t sustained_after = 0;
    const auto sustained_start = std::chrono::steady_clock::now();
    for (int i = 1; i <= 4000; ++i) {
        sustained.publish(Sample{});
        if (i % 40 != 0) continue;
        const auto batch = sustained.read_since(19, sustained_after, 128);
        assert(!batch.overrun && batch.rows.size() == 40);
        sustained_after = batch.rows.back().seq;
    }
    const auto sustained_end = std::chrono::steady_clock::now();
    assert(sustained_after == 4000);

    Ring benchmark(16);
    Sample active;
    active.servo_mode = 1;
    const auto publish_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 4000; ++i) benchmark.publish(active);
    const auto publish_end = std::chrono::steady_clock::now();
    auto bench_cursor = benchmark.read_since(0, 0, 0);
    for (int i = 0; i < 4000; ++i) benchmark.publish(active);
    std::uint32_t read_after = bench_cursor.latest_seq;
    std::size_t rows_read = 0;
    const auto drain_start = std::chrono::steady_clock::now();
    while (rows_read < heimdallr_ft::kRingCapacity) {
        const auto batch = benchmark.read_since(16, read_after, 128);
        assert(!batch.rows.empty());
        rows_read += batch.rows.size();
        read_after = batch.rows.back().seq;
    }
    const auto drain_end = std::chrono::steady_clock::now();
    const auto publish_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                publish_end - publish_start)
                                .count() /
                            4000;
    const auto drain_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              drain_end - drain_start)
                              .count();
    std::cout << "FT telemetry checks passed; publish " << publish_ns
              << " ns/frame, drain " << drain_ns << " ns/1024 rows, 4000"
              << " rows plus 100 reads in "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     sustained_end - sustained_start)
                     .count()
              << " us\n";
}
