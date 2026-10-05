#include "../ddspc_snapshot.hpp"
#include "../fringe_frame_wait.hpp"
#include "../servo_transitions.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using heimdallr_ddspc::DdspcServo;
using heimdallr_ddspc::Modes;
using heimdallr_ddspc::SnapshotJob;
using heimdallr_ddspc::SnapshotWriter;
using heimdallr_ddspc::Telescopes;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void advance(DdspcServo& servo, Telescopes& dm, int count) {
    for (int i = 0; i < count; ++i) {
        const Telescopes phase(std::sin(0.2 * i), std::cos(0.13 * i),
                               std::sin(0.07 * i), std::cos(0.11 * i));
        dm = servo.propose(phase, dm, 2.1, 6.0, 0.4, Modes::Zero());
        servo.update(dm, 2.1, 6.0);
    }
}

int main() {
    int mode = 4;
    std::uint64_t sequence = 0;
    std::atomic<std::uint64_t> generation{0};
    std::deque<heimdallr_ddspc::ServoTransition> transitions;
    heimdallr_ddspc::Parameters params;
    heimdallr_ddspc::record_servo_transition(
        mode, 5, "servo", params, 1, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 4, "servo", params, 2, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 5, "servo", params, 3, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 4, "offload gd", params, 4, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 4, "servo", params, 5, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 5, "servo", params, 6, transitions, sequence, generation);
    heimdallr_ddspc::record_servo_transition(
        mode, 4, "offload mod", params, 7, transitions, sequence, generation);
    check(transitions.size() == 6 && generation.load() == 6 &&
              transitions[1].from == 5 && transitions[1].to == 4 &&
              transitions[3].sequence == 4 &&
              std::string(transitions[3].trigger) == "offload gd" &&
              std::string(transitions[5].trigger) == "offload mod",
          "Rapid off transitions were lost or duplicated");

    sem_t k1, k2;
    check(sem_init(&k1, 0, 0) == 0 && sem_init(&k2, 0, 0) == 0,
          "Could not create test semaphores");
    bool k1_ready = false;
    bool k2_ready = false;
    check(!heimdallr_ddspc::wait_for_frame(&k1, k1_ready, 1000000),
          "Paused K1 stream did not time out");
    check(sem_post(&k1) == 0 &&
              heimdallr_ddspc::wait_for_frame(&k1, k1_ready, 1000000) &&
              !heimdallr_ddspc::wait_for_frame(&k2, k2_ready, 1000000) &&
              k1_ready && !k2_ready,
          "Paused K2 stream lost its matching K1 signal");
    check(sem_post(&k2) == 0 &&
              heimdallr_ddspc::wait_for_frame(&k2, k2_ready, 1000000),
          "Matching K2 signal was not consumed");
    sem_destroy(&k1);
    sem_destroy(&k2);

    DdspcServo servo;
    Telescopes dm = Telescopes::Zero();
    servo.enter();
    advance(servo, dm, 60);
    const auto expected_factor = servo.controller().rls().factor();
    const auto expected_weights = servo.controller().rls().weights().eval();
    const auto expected_inverse = servo.controller().inverse().eval();
    const auto expected_predictive = servo.controller().predictive().eval();
    auto active = servo.snapshot_for_off();
    check(active->trained && std::string(active->source) == "active",
          "Active trained model was not selected");
    check(active->iterations == 60, "Wrong model update count");
    check(active->factor == expected_factor, "RLS factor changed in snapshot");
    check(active->weights == expected_weights, "RLS weights changed in snapshot");
    check(active->inverse == expected_inverse, "SVD inverse changed in snapshot");
    check(active->predictive == expected_predictive,
          "Predictive matrix changed in snapshot");

    servo.enter();
    advance(servo, dm, 60);
    const auto retained_factor = servo.controller().rls().factor();
    const auto retained_weights = servo.controller().rls().weights().eval();
    servo.invalidate();
    advance(servo, dm, 2);
    auto retained = servo.snapshot_for_off();
    check(retained->trained && std::string(retained->source) == "retained",
          "Last trained segment was not retained");
    check(retained->factor == retained_factor &&
              retained->weights == retained_weights,
          "Retained model changed after a short new segment");

    servo.enter();
    auto untrained = servo.snapshot_for_off();
    check(!untrained->trained && std::string(untrained->source) == "untrained",
          "Empty run was not marked untrained");
    SnapshotJob empty_job{std::move(untrained), 1791200000000000000, 1,
                          "servo"};
    const auto empty_json = heimdallr_ddspc::snapshot_json(empty_job);
    check(empty_json["inverse"].is_null() &&
              empty_json["predictive"].is_null(),
          "Untrained run has a control matrix");

    active->weights(0, 0) = std::numeric_limits<double>::quiet_NaN();
    SnapshotJob first{std::move(active), 1791200000000000000, 7,
                      "offload gd"};
    const auto first_json = heimdallr_ddspc::snapshot_json(first);
    check(first_json["rls"]["weights"][0][0] == "NaN" &&
              first_json["nonfinite_values"] == true,
          "Non-finite coefficient was silently discarded");
    check(first_json["rls"]["R"].size() == 249 &&
              first_json["rls"]["weights"].size() == 249 &&
              first_json["inverse"].size() == 12 &&
              first_json["predictive"].size() == 3,
          "Snapshot matrix dimensions are wrong");
    check(first_json["rls"]["R"][1][2] == expected_factor[1 * 249 + 2] &&
              first_json["rls"]["weights"][0][1] == expected_weights(0, 1) &&
              first_json["inverse"][0][1] == expected_inverse(0, 1) &&
              first_json["predictive"][1][2] == expected_predictive(1, 2),
          "JSON matrix row or column order is wrong");

    char root_template[] = "/tmp/heimdallr-ddspc-XXXXXX";
    const char* root_name = mkdtemp(root_template);
    check(root_name != nullptr, "Could not create test directory");
    const std::filesystem::path root(root_name);
    std::vector<std::pair<bool, std::string>> results;
    std::mutex results_mutex;
    auto callback = [&](bool success, const std::string& message) {
        std::lock_guard<std::mutex> lock(results_mutex);
        results.emplace_back(success, message);
    };
    {
        SnapshotWriter writer(root, callback);
        check(writer.enqueue(std::move(first)), "First snapshot was rejected");
        retained->weights(0, 0) = 42.0;
        SnapshotJob second{std::move(retained), 1791200000000000000, 7,
                           "servo"};
        check(writer.enqueue(std::move(second)), "Second snapshot was rejected");
        writer.stop();
    }
    check(results.size() == 2 && results[0].first && results[1].first,
          "Snapshot writer did not drain both jobs");
    check(results[0].second != results[1].second,
          "Same-timestamp snapshots overwrote each other");
    for (const auto& result : results) {
        std::ifstream file(result.second);
        check(file.good(), "Snapshot file is missing");
        nlohmann::json data;
        file >> data;
        check(data["schema_version"] == 1 &&
                  data["transition"]["to"] == "off",
              "Published snapshot is not valid JSON");
    }

    const auto blocked_root = root / "blocked";
    std::ofstream(blocked_root).put('x');
    results.clear();
    {
        SnapshotWriter writer(blocked_root, callback);
        DdspcServo fresh;
        fresh.enter();
        SnapshotJob failed{fresh.snapshot_for_off(), 1791200000000000000,
                           8, "servo"};
        check(writer.enqueue(std::move(failed)), "Failure job was rejected");
        writer.stop();
    }
    check(results.size() == 1 && !results[0].first,
          "Filesystem failure was not reported");
    std::filesystem::remove_all(root);
    std::cout << "DDSPC snapshot checks passed\n";
}
