#pragma once

#include "predictive_control.hpp"

#include <cmath>

namespace heimdallr_ddspc {

class PistonResetHold {
   public:
    static constexpr int Frames = 5;

    void arm(const Telescopes& last_command, double saturation_limit) {
        if (!active()) {
            held_command_ = last_command;
            const double threshold = 0.95 * saturation_limit;
            for (int i = 0; i < held_command_.size(); ++i) {
                if (!std::isfinite(held_command_(i)) ||
                    std::abs(held_command_(i)) >= threshold) {
                    held_command_.setZero();
                    break;
                }
            }
        }
        frames_remaining_ = Frames;
    }
    void clear() {
        frames_remaining_ = 0;
        held_command_.setZero();
    }
    bool active() const { return frames_remaining_ > 0; }

    bool on_paired_frame(bool valid) {
        if (!active()) return false;
        if (valid) --frames_remaining_;
        return true;
    }

    Telescopes command(const Telescopes& requested, bool hold_this_frame) const {
        return hold_this_frame ? held_command_ : requested;
    }

   private:
    int frames_remaining_ = 0;
    Telescopes held_command_ = Telescopes::Zero();
};

}  // namespace heimdallr_ddspc
