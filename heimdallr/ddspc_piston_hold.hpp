#pragma once

#include "predictive_control.hpp"

namespace heimdallr_ddspc {

class PistonResetHold {
   public:
    static constexpr int Frames = 5;

    void arm() { frames_remaining_ = Frames; }
    void clear() { frames_remaining_ = 0; }
    bool active() const { return frames_remaining_ > 0; }

    bool on_paired_frame(bool valid) {
        if (!active()) return false;
        if (valid) --frames_remaining_;
        return true;
    }

    Telescopes command(const Telescopes& requested, bool hold_this_frame) const {
        return hold_this_frame ? Telescopes::Zero() : requested;
    }

   private:
    int frames_remaining_ = 0;
};

}  // namespace heimdallr_ddspc
