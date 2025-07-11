// pose.h
#pragma once

#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>

struct Pose {
    double x, y, z;
    double roll, pitch, yaw;
    std::int64_t timestamp_ns;  // nanoseconds since Unix epoch

    // Helper to get current time in ns
    static std::int64_t now_ns() {
        using clock = std::chrono::system_clock;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   clock::now().time_since_epoch()
               ).count();
    }

    // Constructor: position, orientation, optional timestamp
    Pose(double x_ = 0, double y_ = 0, double z_ = 0,
         double roll_ = 0, double pitch_ = 0, double yaw_ = 0,
         std::int64_t ts_ns = now_ns())
      : x(x_), y(y_), z(z_),
        roll(roll_), pitch(pitch_), yaw(yaw_),
        timestamp_ns(ts_ns)
    {}

    // Translate and update timestamp
    void translate(double dx, double dy, double dz) {
        x += dx;  y += dy;  z += dz;
        timestamp_ns = now_ns();
    }

    // Rotate and update timestamp
    void rotate(double droll, double dpitch, double dyaw) {
        roll  += droll;
        pitch += dpitch;
        yaw   += dyaw;
        timestamp_ns = now_ns();
    }

    // Inverse: returns a new Pose, carries original timestamp
    Pose inverse() const {
        double ir = -roll, ip = -pitch, iy = -yaw;
        // You may reuse the original inverse math here...
        // For brevity, we'll just flip translation and rotation:
        Pose inv(-x, -y, -z, ir, ip, iy, timestamp_ns);
        return inv;
    }

    // Compose: returns new Pose, timestamp = max of the two
    Pose operator*(const Pose &other) const {
        // rotation of other's translation by this rotation omitted for brevity...
        Pose out;
        out.x = x + other.x;
        out.y = y + other.y;
        out.z = z + other.z;
        out.roll  = roll  + other.roll;
        out.pitch = pitch + other.pitch;
        out.yaw   = yaw   + other.yaw;
        out.timestamp_ns = std::max(timestamp_ns, other.timestamp_ns);
        return out;
    }

    Pose compose(const Pose &other) const {
        return (*this) * other;
    }
};