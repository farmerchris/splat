//! Orbit camera. Camera space follows the 3DGS/COLMAP convention:
//! +x right, +y down, +z forward (into the screen).

use glam::{Mat3, Vec3};

use crate::scene::Bounds;

/// Which world axis points "up" on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpAxis {
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl UpAxis {
    pub fn vector(self) -> Vec3 {
        match self {
            UpAxis::PosY => Vec3::Y,
            UpAxis::NegY => Vec3::NEG_Y,
            UpAxis::PosZ => Vec3::Z,
            UpAxis::NegZ => Vec3::NEG_Z,
        }
    }

    pub fn next(self) -> Self {
        match self {
            UpAxis::NegY => UpAxis::PosY,
            UpAxis::PosY => UpAxis::PosZ,
            UpAxis::PosZ => UpAxis::NegZ,
            UpAxis::NegZ => UpAxis::NegY,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            UpAxis::PosY => "+Y",
            UpAxis::NegY => "-Y",
            UpAxis::PosZ => "+Z",
            UpAxis::NegZ => "-Z",
        }
    }

    /// Two horizontal axes completing a right-handed frame with `vector()`.
    fn horizontal(self) -> (Vec3, Vec3) {
        let up = self.vector();
        let seed = if up.y.abs() > 0.5 { Vec3::Z } else { Vec3::Y };
        let a = seed.cross(up).normalize();
        let b = up.cross(a).normalize();
        (a, b)
    }
}

impl std::str::FromStr for UpAxis {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "+y" | "y" => Ok(UpAxis::PosY),
            "-y" => Ok(UpAxis::NegY),
            "+z" | "z" => Ok(UpAxis::PosZ),
            "-z" => Ok(UpAxis::NegZ),
            _ => Err(format!("invalid up axis '{s}' (use +y, -y, +z or -z)")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation around the up axis, radians.
    pub yaw: f32,
    /// Elevation above the horizontal plane, radians, clamped to ±89°.
    pub pitch: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub up: UpAxis,
    /// Scene radius; used to scale pan/zoom speeds and the near plane.
    pub scene_radius: f32,
    home: (Vec3, f32, f32, f32),
}

pub const MAX_PITCH: f32 = 89.0 * std::f32::consts::PI / 180.0;

impl OrbitCamera {
    pub fn new(bounds: Bounds, up: UpAxis) -> Self {
        let fov_y = 50f32.to_radians();
        // Fit the robust radius into the vertical field of view with some margin.
        let distance = bounds.radius / (fov_y * 0.5).tan() * 1.1;
        let mut cam = OrbitCamera {
            target: bounds.center,
            distance,
            yaw: 0.0,
            pitch: 15f32.to_radians(),
            fov_y,
            up,
            scene_radius: bounds.radius,
            home: (bounds.center, distance, 0.0, 15f32.to_radians()),
        };
        cam.reset();
        cam
    }

    pub fn reset(&mut self) {
        let (t, d, y, p) = self.home;
        self.target = t;
        self.distance = d;
        self.yaw = y;
        self.pitch = p;
    }

    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw = (self.yaw + dyaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Multiply the orbit distance (values < 1 zoom in).
    pub fn zoom(&mut self, factor: f32) {
        let min = self.scene_radius * 1e-3;
        let max = self.scene_radius * 1e3;
        self.distance = (self.distance * factor).clamp(min, max);
    }

    /// Pan in screen space; units are fractions of the view height.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let v = self.view();
        let scale = self.distance * (self.fov_y * 0.5).tan() * 2.0;
        self.target += (v.right * dx + v.down * dy) * scale;
    }

    pub fn eye(&self) -> Vec3 {
        let (a, b) = self.up.horizontal();
        let up = self.up.vector();
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = (a * sy + b * cy) * cp + up * sp;
        self.target + dir * self.distance
    }

    pub fn view(&self) -> View {
        let eye = self.eye();
        let forward = (self.target - eye).normalize();
        let right = forward.cross(self.up.vector()).normalize();
        let down = forward.cross(right);
        View {
            eye,
            right,
            down,
            forward,
            fov_y: self.fov_y,
            near: self.near(),
        }
    }

    fn near(&self) -> f32 {
        (self.scene_radius * 0.05).max(1e-4)
    }
}

/// Resolved camera pose used by renderers.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: Vec3,
    pub right: Vec3,
    pub down: Vec3,
    pub forward: Vec3,
    pub fov_y: f32,
    pub near: f32,
}

impl View {
    /// World-to-camera rotation (rows are right/down/forward).
    pub fn rotation(&self) -> Mat3 {
        Mat3::from_cols(self.right, self.down, self.forward).transpose()
    }

    /// Focal length in pixels for an image `height` pixels tall.
    pub fn focal(&self, height: u32) -> f32 {
        height as f32 * 0.5 / (self.fov_y * 0.5).tan()
    }

    pub fn to_camera(&self, p: Vec3) -> Vec3 {
        let d = p - self.eye;
        Vec3::new(d.dot(self.right), d.dot(self.down), d.dot(self.forward))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(up: UpAxis) -> OrbitCamera {
        OrbitCamera::new(
            Bounds {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            up,
        )
    }

    #[test]
    fn view_basis_is_orthonormal_right_handed() {
        for up in [UpAxis::PosY, UpAxis::NegY, UpAxis::PosZ, UpAxis::NegZ] {
            let mut c = cam(up);
            c.rotate(0.7, 0.3);
            let v = c.view();
            assert!((v.right.length() - 1.0).abs() < 1e-5);
            assert!((v.down.length() - 1.0).abs() < 1e-5);
            assert!(v.right.dot(v.down).abs() < 1e-5);
            assert!(v.right.dot(v.forward).abs() < 1e-5);
            // x right, y down, z forward is right-handed.
            assert!((v.right.cross(v.down) - v.forward).length() < 1e-5);
            // Screen "up" (-down) has a positive component along world up.
            assert!((-v.down).dot(up.vector()) > 0.0);
        }
    }

    #[test]
    fn target_projects_to_center() {
        let c = cam(UpAxis::NegY);
        let v = c.view();
        let p = v.to_camera(c.target);
        assert!(p.x.abs() < 1e-5 && p.y.abs() < 1e-5);
        assert!((p.z - c.distance).abs() < 1e-4);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut c = cam(UpAxis::PosY);
        c.rotate(0.0, 10.0);
        assert!(c.pitch <= MAX_PITCH);
        c.rotate(0.0, -20.0);
        assert!(c.pitch >= -MAX_PITCH);
    }

    #[test]
    fn reset_restores_home() {
        let mut c = cam(UpAxis::PosY);
        let d = c.distance;
        c.rotate(1.0, 0.5);
        c.zoom(0.5);
        c.pan(0.3, 0.1);
        c.reset();
        assert_eq!(c.distance, d);
        assert_eq!(c.target, Vec3::ZERO);
    }

    #[test]
    fn up_axis_parse() {
        assert_eq!("-y".parse::<UpAxis>().unwrap(), UpAxis::NegY);
        assert_eq!("Z".parse::<UpAxis>().unwrap(), UpAxis::PosZ);
        assert!("w".parse::<UpAxis>().is_err());
    }
}
