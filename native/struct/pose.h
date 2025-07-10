// pose.h
#pragma once
#include <cmath>

struct Pose {
    double x, y, z;
    double roll, pitch, yaw;

    Pose(double x_=0, double y_=0, double z_ = 0,
         double roll_=0, double pitch_=0, double yaw_=0)
      : x(x_), y(y_), z(z_), roll(roll_), pitch(pitch_), yaw(yaw_) {}

    // translate & rotate as before
    void translate(double dx, double dy, double dz) {
        x += dx; y += dy; z += dz;
    }
    void rotate(double droll, double dpitch, double dyaw) {
        roll  += droll;
        pitch += dpitch;
        yaw   += dyaw;
    }

    // 1) Inverse of this pose
    Pose inverse() const {
        // First invert rotation: negate each angle
        double ir = -roll, ip = -pitch, iy = -yaw;
        // Then invert translation: rotate the negated translation by inv-rotation
        // Using simple inverse of yaw-pitch-roll (ZYX) order:
        // We'll apply the inverse rotation in reverse order.
        // For small angles this approximation works; for full generality you'd do a full matrix inverse.
        // Here’s a quick ZYX rotation of (-x, -y, -z):
        double cx = std::cos(ip), sx = std::sin(ip);
        double cy = std::cos(iy), sy = std::sin(iy);
        double cz = std::cos(ir), sz = std::sin(ir);

        // rotate point (-x,-y,-z) by Rz(iy)*Ry(ip)*Rx(ir):
        double nx =  cy*cz * (-x) + (cz*sx*sy - cx*sz) * (-y) + (sx*sz + cx*cz*sy) * (-z);
        double ny =  cy*sz * (-x) + (cx*cz + sx*sy*sz) * (-y) + (cx*sy*sz - cz*sx) * (-z);
        double nz = -sy    * (-x) + cy*sx           * (-y) + cx*cy           * (-z);

        return Pose(nx, ny, nz, ir, ip, iy);
    }

    // 2) Composition: this * other
    Pose operator*(const Pose &other) const {
        // apply this, then other: new_pose = this ∘ other
        // First rotate other’s translation by this rotation
        // (again, using ZYX Euler for simplicity)
        double cx = std::cos(roll), sx = std::sin(roll);
        double cy = std::cos(pitch), sy = std::sin(pitch);
        double cz = std::cos(yaw), sz = std::sin(yaw);

        double ox = other.x, oy = other.y, oz = other.z;
        double rx = cz*cy*ox + (cz*sy*sx - sz*cx)*oy + (cz*sy*cx + sz*sx)*oz;
        double ry = sz*cy*ox + (sz*sy*sx + cz*cx)*oy + (sz*sy*cx - cz*sx)*oz;
        double rz =      -sy*ox +               cy*sx*oy +               cy*cx*oz;

        // then translate
        double nx = x + rx;
        double ny = y + ry;
        double nz = z + rz;

        // and accumulate rotations (just add Euler angles)
        double nr = roll  + other.roll;
        double np = pitch + other.pitch;
        double nyaw = yaw  + other.yaw;

        return Pose(nx, ny, nz, nr, np, nyaw);
    }

    // convenience named method
    Pose compose(const Pose &other) const {
        return (*this) * other;
    }
};