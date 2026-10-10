#include "../ddspc_snapshot.hpp"
#include "../ddspc_piston_hold.hpp"
#include "../fringe_frame_wait.hpp"
#include "../servo_transitions.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
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

template <int HistoryLength, int FutureLength>
void check_preset() {
    heimdallr_ddspc::Parameters params;
    params.history = HistoryLength;
    params.future = FutureLength;
    DdspcServo servo;
    Telescopes dm = Telescopes::Zero();
    servo.enter(params);
    constexpr int training_delay = HistoryLength + FutureLength;
    constexpr int frames = training_delay + 4;
    advance(servo, dm, frames);
    check(servo.controller<HistoryLength, FutureLength>().rls_updates() == 4,
          "Preset training delay is wrong");
    auto active = servo.snapshot_for_off();
    check(active->trained && active->rls_updates == 4 &&
              active->factor.size() ==
                  static_cast<std::size_t>(active->features * active->features),
          "Preset active model dimensions are wrong");
    SnapshotJob job{std::move(active), 1791200000000000000, 1, "servo"};
    const auto data = heimdallr_ddspc::snapshot_json(job);
    check(data["schema_version"] == 1 &&
              data["dimensions"]["history"] == HistoryLength &&
              data["dimensions"]["future"] == FutureLength &&
              data["rls"]["R"].size() ==
                  static_cast<std::size_t>(job.model->features) &&
              data["rls"]["weights"].size() ==
                  static_cast<std::size_t>(job.model->features) &&
              data["inverse"].size() ==
                  static_cast<std::size_t>(job.model->outputs) &&
              data["predictive"][0].size() ==
                  static_cast<std::size_t>(job.model->control_features),
          "Preset snapshot JSON dimensions are wrong");

    servo.enter(params);
    advance(servo, dm, frames);
    servo.invalidate();
    advance(servo, dm, 1);
    auto retained = servo.snapshot_for_off();
    check(retained->trained && std::string(retained->source) == "retained" &&
              retained->parameters.history == HistoryLength &&
              retained->parameters.future == FutureLength,
          "Preset retained model was lost");

    params.continue_learning = false;
    params.n_exploration = frames;
    servo.enter(params);
    advance(servo, dm, frames);
    check(servo.frozen() && servo.rls_updates() == 4,
          "Preset did not freeze at the exploration boundary");
    advance(servo, dm, 2);
    check(servo.rls_updates() == 4,
          "Preset learned after freezing");
    auto frozen = servo.snapshot_for_off();
    check(frozen->frozen && frozen->rls_updates == 4 &&
              frozen->parameters.history == HistoryLength,
          "Preset frozen snapshot is wrong");
}

void check_preset_switches() {
    DdspcServo servo;
    Telescopes dm = Telescopes::Zero();
    heimdallr_ddspc::Parameters params;
    for (const auto& [history, future] :
         {std::pair{20, 2}, std::pair{30, 3},
          std::pair{40, 3}, std::pair{50, 3}}) {
        params.history = history;
        params.future = future;
        servo.enter(params);
        advance(servo, dm, history + future + 1);
        auto model = servo.snapshot_for_off();
        check(model->rls_updates == 1 &&
                  model->parameters.history == history &&
                  model->parameters.future == future,
              "Switching presets retained the previous controller");
    }
}

int main() {
    using Controller = heimdallr_ddspc::PredictiveControl<>;
    check_preset<20, 2>();
    check_preset<30, 3>();
    check_preset<40, 3>();
    check_preset<50, 3>();
    check_preset_switches();
    constexpr int updates_after_fifty = 50 - Controller::TrainingDelay;
    heimdallr_ddspc::PistonResetHold piston_hold;
    const Telescopes nonzero_piston = Telescopes::Constant(0.2);
    check(!piston_hold.active() &&
              piston_hold.command(nonzero_piston, false) == nonzero_piston,
          "Piston command was changed before a reset");
    piston_hold.arm();
    check(piston_hold.active() &&
              piston_hold.command(nonzero_piston, piston_hold.active()).isZero(),
          "Fit reset did not immediately select zero piston");
    check(piston_hold.on_paired_frame(false) && piston_hold.active(),
          "Bad or missing camera pair shortened the piston hold");
    for (int i = 0; i < heimdallr_ddspc::PistonResetHold::Frames; ++i) {
        const bool hold_this_frame = piston_hold.on_paired_frame(true);
        check(hold_this_frame &&
                  piston_hold.command(nonzero_piston, hold_this_frame).isZero(),
              "Piston was not zero throughout the five valid pairs");
        if (i == 2) {
            check(piston_hold.on_paired_frame(false),
                  "A gapped camera pair ended the piston hold");
        }
    }
    check(!piston_hold.on_paired_frame(true) &&
              piston_hold.command(nonzero_piston, false) == nonzero_piston,
          "Piston control did not resume on the sixth valid pair");
    piston_hold.arm();
    piston_hold.clear();
    check(!piston_hold.active() && !piston_hold.on_paired_frame(true),
          "Leaving DDSPC did not clear the piston hold");

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
    int queued_mode = 4;
    std::uint64_t queued_sequence = 0;
    std::atomic<std::uint64_t> queued_generation{0};
    std::deque<heimdallr_ddspc::ServoTransition> queued_events;
    heimdallr_ddspc::record_servo_transition(
        queued_mode, 5, "servo", params, 1, queued_events,
        queued_sequence, queued_generation);
    heimdallr_ddspc::record_freeze_request(
        queued_mode, 5, params, 2, queued_events, queued_sequence,
        queued_generation);
    heimdallr_ddspc::record_servo_transition(
        queued_mode, 4, "servo", params, 3, queued_events,
        queued_sequence, queued_generation);
    check(queued_events.size() == 3 && !queued_events[0].freeze &&
              queued_events[1].freeze && !queued_events[2].freeze &&
              queued_events[1].time_ns < queued_events[2].time_ns,
          "Freeze request was not ordered before mode exit");

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
    check(active->rls_updates == 60 - Controller::TrainingDelay &&
              !active->frozen,
          "Active model learning count is wrong");
    check(std::equal(active->factor.begin(), active->factor.end(),
                     expected_factor.begin(), expected_factor.end()),
          "RLS factor changed in snapshot");
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
    check(std::equal(retained->factor.begin(), retained->factor.end(),
                     retained_factor.begin(), retained_factor.end()) &&
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

    heimdallr_ddspc::Parameters freeze_params;
    freeze_params.continue_learning = false;
    freeze_params.n_exploration = 50;
    freeze_params.reg_interval = 5;
    freeze_params.reg_divisor = 1.1;
    DdspcServo interrupted;
    Telescopes interrupted_dm = Telescopes::Zero();
    interrupted.enter(freeze_params);
    advance(interrupted, interrupted_dm, 49);
    check(interrupted.controller().rls_updates() > 0,
          "Learning interruption test did not train a model");
    interrupted.invalidate();
    check(!interrupted.frozen() && interrupted.exploration_frames() == 49 &&
              interrupted.controller().iterations() == 0 &&
              interrupted.controller().rls_updates() == 0,
          "Fit interruption changed the exploration count");
    advance(interrupted, interrupted_dm, 1);
    check(interrupted.frozen() && interrupted.exploration_frames() == 50,
          "Learning did not freeze at the original exploration boundary");
    DdspcServo automatic;
    Telescopes automatic_dm = Telescopes::Zero();
    automatic.enter(freeze_params);
    advance(automatic, automatic_dm, 50);
    check(automatic.frozen() && automatic.exploration_frames() == 50 &&
              automatic.controller().rls_updates() == updates_after_fifty,
          "Automatic freeze missed the exploration boundary");
    const auto frozen_factor = automatic.controller().rls().factor();
    const auto frozen_weights = automatic.controller().rls().weights().eval();
    const auto frozen_predictive = automatic.controller().predictive().eval();
    const double frozen_regularization = automatic.controller().regularization();
    advance(automatic, automatic_dm, 20);
    check(automatic.controller().rls().factor() == frozen_factor &&
              automatic.controller().rls().weights() == frozen_weights &&
              automatic.controller().predictive() == frozen_predictive &&
              automatic.controller().regularization() == frozen_regularization &&
              automatic.controller().rls_updates() == updates_after_fifty &&
              automatic.controller().iterations() == 70,
          "Frozen control law changed after exploration");
    automatic.invalidate();
    advance(automatic, automatic_dm, 3);
    check(automatic.controller().rls().factor() == frozen_factor &&
              automatic.controller().predictive() == frozen_predictive &&
              automatic.controller().rls_updates() == updates_after_fifty,
          "Frozen model was lost after tracking interruption");
    auto automatic_model = automatic.snapshot_for_off();
    SnapshotJob automatic_job{std::move(automatic_model), 1791200000000000000,
                              2, "servo"};
    const auto automatic_json = heimdallr_ddspc::snapshot_json(automatic_job);
    check(automatic_json["model"]["frozen"] == true &&
              automatic_json["model"]["freeze_reason"] == "after_exploration" &&
              automatic_json["model"]["freeze_frame"] == 50 &&
              automatic_json["model"]["freeze_time_utc"].is_string() &&
              automatic_json["model"]["rls_updates"] == updates_after_fifty &&
              automatic_json["parameters"]["continue_learning"] == false,
          "Automatic freeze was not recorded in snapshot JSON");

    heimdallr_ddspc::Parameters continuing_params;
    continuing_params.n_exploration = 2;
    DdspcServo continuing;
    Telescopes continuing_dm = Telescopes::Zero();
    continuing.enter(continuing_params);
    advance(continuing, continuing_dm, 50);
    check(!continuing.frozen() &&
              continuing.controller().rls_updates() == updates_after_fifty,
          "Default controller stopped learning after exploration");
    continuing.freeze_manual();
    const auto manual_factor = continuing.controller().rls().factor();
    advance(continuing, continuing_dm, 10);
    check(continuing.controller().rls().factor() == manual_factor &&
              continuing.controller().rls_updates() == updates_after_fifty,
          "Manual freeze after exploration did not stop learning");

    DdspcServo manual;
    manual.enter();
    manual.freeze_manual();
    manual.freeze_manual();
    const Telescopes no_dither = manual.propose(
        Telescopes::Zero(), Telescopes::Zero(), 2.1, 6.0, 1.0,
        Modes::Constant(10.0));
    check(no_dither.isZero() && manual.frozen(),
          "Manual freeze did not stop exploration dither");
    manual.update(no_dither, 2.1, 6.0);
    auto manual_model = manual.snapshot_for_off();
    SnapshotJob manual_job{std::move(manual_model), 1791200000000000000,
                           3, "servo"};
    const auto manual_json = heimdallr_ddspc::snapshot_json(manual_job);
    check(manual_json["model"]["frozen"] == true &&
              manual_json["model"]["freeze_reason"] == "manual" &&
              manual_json["model"]["freeze_frame"] == 0 &&
              manual_json["model"]["trained"] == false &&
              manual_json["model"]["rls_updates"] == 0 &&
              manual_json["predictive"].is_null(),
          "Early manual freeze was not recorded as untrained");

    freeze_params.n_exploration = 0;
    DdspcServo immediate;
    immediate.enter(freeze_params);
    check(immediate.frozen() && immediate.controller().rls_updates() == 0,
          "Zero exploration did not freeze immediately");

    active->weights(0, 0) = std::numeric_limits<double>::quiet_NaN();
    SnapshotJob first{std::move(active), 1791200000000000000, 7,
                      "offload gd"};
    const auto first_json = heimdallr_ddspc::snapshot_json(first);
    check(first_json["rls"]["weights"][0][0] == "NaN" &&
              first_json["nonfinite_values"] == true,
          "Non-finite coefficient was silently discarded");
    check(first_json["rls"]["R"].size() == Controller::Features &&
              first_json["rls"]["weights"].size() == Controller::Features &&
              first_json["inverse"].size() == Controller::Outputs &&
              first_json["predictive"].size() == 3,
          "Snapshot matrix dimensions are wrong");
    check(first_json["dimensions"]["history"] == 30 &&
              first_json["dimensions"]["future"] == 3,
          "Snapshot controller dimensions are wrong");
    check(first_json["rls"]["R"][1][2] ==
                  expected_factor[Controller::Features + 2] &&
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
