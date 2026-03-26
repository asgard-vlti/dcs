#include "heimdallr.h"
#include <nlohmann/json.hpp>
#include <zmq.hpp>
#include <chrono>
#include <thread>

#define CAM_PORT 6667
#define POLL_PERIOD_MS 500
#define TIMEOUT_MS 2000

namespace {

std::atomic<bool> keep_cam_polling{false};
std::thread cam_poll_thread;
std::string status_string = "status";
std::string cam_endpoint() {
    return "tcp://192.168.100.2:" + std::to_string(CAM_PORT);
}

void create_cam_socket(zmq::socket_t& socket) {
    int timeout_ms = TIMEOUT_MS;
    int linger_ms = 0;
    socket.setsockopt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    socket.setsockopt(ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
    socket.setsockopt(ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
    socket.connect(cam_endpoint());
}

void reconnect_cam_socket(zmq::socket_t& socket, zmq::context_t& context) {
    try {
        socket.close();
    } catch (const zmq::error_t&) {
    }
    socket = zmq::socket_t(context, zmq::socket_type::req);
    create_cam_socket(socket);
}

void update_camera_status(const nlohmann::json& status) {
    if (!status.contains("fps") || !status.contains("nbreads") || !status.contains("tsig_len")) {
        return;
    }

    const double fps = status["fps"].get<double>();
    if (fps <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> lock(beam_mutex);
    control_u.dit = 1.0 / fps;
    control_u.nbreads = status["nbreads"].get<int>();
    control_u.tsig_len = status["tsig_len"].get<int>();
    //std::cout << "Camera status updated: fps=" << fps << ", nbreads=" << control_u.nbreads << ", tsig_len=" << control_u.tsig_len << std::endl;   
}

void camera_poll_loop() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::req);
    create_cam_socket(socket);
    while (keep_cam_polling.load()) {
        bool need_reconnect = false;

        try {
            socket.send(zmq::buffer(status_string), zmq::send_flags::none);
            zmq::message_t reply;
            auto result = socket.recv(reply, zmq::recv_flags::none);
            if (!result.has_value()) {
                need_reconnect = true;
            } else {
                const auto payload = std::string(static_cast<char*>(reply.data()), reply.size());
                auto status = nlohmann::json::parse(payload, nullptr, false);
                if (!status.is_discarded()) {
                    update_camera_status(status);
                } else std::cout << "Json ERROR!!!" << std::endl;
            }
        } catch (const zmq::error_t&) {
            need_reconnect = true;
        } catch (const std::exception&) {
            // Keep polling even if one payload is malformed.
        }

        if (need_reconnect && keep_cam_polling.load()) {
        	std::cout << "Reconnecting" << std::endl;
            reconnect_cam_socket(socket, context);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_PERIOD_MS));
    }

    try {
        socket.close();
    } catch (const zmq::error_t&) {
    }
}

} // namespace

void start_camera_client() {
    if (keep_cam_polling.exchange(true)) {
        return;
    }
    cam_poll_thread = std::thread(camera_poll_loop);
}

void stop_camera_client() {
    keep_cam_polling.store(false);
    if (cam_poll_thread.joinable()) {
        cam_poll_thread.join();
    }
}
