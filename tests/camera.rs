//! Orbit camera invariants.

use glam::Vec3;
use splat_tui::camera::{OrbitCamera, UpAxis, MAX_PITCH};
use splat_tui::scene::Bounds;

const ALL: [UpAxis; 4] = [UpAxis::PosY, UpAxis::NegY, UpAxis::PosZ, UpAxis::NegZ];

fn cam(up: UpAxis) -> OrbitCamera {
    OrbitCamera::new(Bounds { center: Vec3::new(1.0, -2.0, 3.0), radius: 2.0 }, up)
}

#[test]
fn orbit_keeps_distance_and_target() {
    for up in ALL {
        let mut c = cam(up);
        let d = c.distance;
        for i in 0..50 {
            c.rotate(0.37 * i as f32, 0.11 * (i as f32).sin());
            assert!(((c.eye() - c.target).length() - d).abs() < 1e-3, "{up:?}");
            let p = c.view().to_camera(c.target);
            assert!(p.x.abs() < 1e-3 && p.y.abs() < 1e-3 && p.z > 0.0);
        }
    }
}

#[test]
fn eye_is_above_horizon_for_positive_pitch() {
    for up in ALL {
        let mut c = cam(up);
        c.rotate(0.0, 0.5 - c.pitch);
        assert!((c.eye() - c.target).dot(up.vector()) > 0.0, "{up:?}");
        c.rotate(0.0, -1.0);
        assert!((c.eye() - c.target).dot(up.vector()) < 0.0, "{up:?}");
    }
}

#[test]
fn view_basis_orthonormal_at_extreme_pitch() {
    for up in ALL {
        for pitch in [10.0, -10.0] {
            let mut c = cam(up);
            c.rotate(1.0, pitch);
            let v = c.view();
            for x in [v.right, v.down, v.forward] {
                assert!(x.is_finite() && (x.length() - 1.0).abs() < 1e-4, "{up:?}");
            }
            assert!(v.right.dot(v.down).abs() < 1e-4);
            assert!(v.down.dot(v.forward).abs() < 1e-4);
            assert!((v.right.cross(v.down) - v.forward).length() < 1e-4);
        }
    }
}

#[test]
fn rotation_matrix_matches_to_camera() {
    for up in ALL {
        let mut c = cam(up);
        c.rotate(2.2, 0.3);
        let v = c.view();
        let p = Vec3::new(0.3, -4.0, 2.5);
        let a = v.to_camera(p);
        let b = v.rotation() * (p - v.eye);
        assert!((a - b).length() < 1e-4);
    }
}

#[test]
fn yaw_wraps_and_full_turn_is_identity() {
    for up in ALL {
        let mut c = cam(up);
        let e0 = c.eye();
        c.rotate(std::f32::consts::TAU, 0.0);
        assert!(c.yaw >= 0.0 && c.yaw < std::f32::consts::TAU + 1e-6);
        assert!((c.eye() - e0).length() < 1e-3);
        c.rotate(-100.0, 0.0);
        assert!(c.yaw >= 0.0 && c.yaw <= std::f32::consts::TAU);
    }
}

#[test]
fn pitch_clamped() {
    let mut c = cam(UpAxis::NegY);
    c.rotate(0.0, 1e9);
    assert_eq!(c.pitch, MAX_PITCH);
    c.rotate(0.0, -1e9);
    assert_eq!(c.pitch, -MAX_PITCH);
}

#[test]
fn zoom_is_multiplicative_and_clamped() {
    let mut c = cam(UpAxis::NegY);
    let d = c.distance;
    c.zoom(0.5);
    assert!((c.distance - d * 0.5).abs() < 1e-5);
    c.zoom(2.0);
    assert!((c.distance - d).abs() < 1e-4);
    c.zoom(1e-12);
    assert!((c.distance - 2.0e-3).abs() < 1e-6, "{}", c.distance);
    c.zoom(1e20);
    assert!((c.distance - 2.0e3).abs() < 1e-1);
    assert!(c.view().eye.is_finite());
}

#[test]
fn pan_moves_target_in_view_plane_and_scales_with_distance() {
    for up in ALL {
        let mut c = cam(up);
        let v0 = c.view();
        let t0 = c.target;
        c.pan(0.25, 0.0);
        let dt = c.target - t0;
        assert!(dt.dot(v0.forward).abs() < 1e-4, "{up:?}");
        assert!(dt.dot(v0.right) > 0.0);
        assert!(dt.dot(v0.down).abs() < 1e-4);
        // Camera orientation is unchanged by panning.
        let v1 = c.view();
        assert!((v1.forward - v0.forward).length() < 1e-5);
        // Panning by a full view height moves the target by one frustum height at the target distance.
        let mut c = cam(up);
        let t0 = c.target;
        c.pan(0.0, 1.0);
        let moved = (c.target - t0).length();
        let h = 2.0 * c.distance * (c.fov_y * 0.5).tan();
        assert!((moved - h).abs() < 1e-3 * h);
        // Panning keeps the target centered on screen.
        let p = c.view().to_camera(c.target);
        assert!(p.x.abs() < 1e-3 && p.y.abs() < 1e-3);
    }
}

#[test]
fn reset_restores_everything() {
    for up in ALL {
        let mut c = cam(up);
        let (e, t, d) = (c.eye(), c.target, c.distance);
        c.rotate(2.0, 0.4);
        c.zoom(3.0);
        c.pan(0.2, -0.3);
        c.reset();
        assert_eq!((c.target, c.distance), (t, d));
        assert!((c.eye() - e).length() < 1e-5);
    }
}

#[test]
fn initial_view_fits_radius_and_near_is_positive() {
    for up in ALL {
        let c = cam(up);
        let v = c.view();
        // A sphere of scene radius at the target should fit vertically.
        let half = c.distance * (c.fov_y * 0.5).tan();
        assert!(half > c.scene_radius);
        assert!(v.near > 0.0 && v.near < c.distance);
        assert!(v.focal(100) > 0.0);
        assert!((v.focal(200) - 2.0 * v.focal(100)).abs() < 1e-3);
    }
}

#[test]
fn degenerate_bounds_still_produce_valid_view() {
    let c = OrbitCamera::new(Bounds { center: Vec3::ZERO, radius: 1.0 }, UpAxis::PosY);
    assert!(c.view().eye.is_finite());
}

#[test]
fn up_axis_cycle_label_parse() {
    let mut a = UpAxis::PosY;
    let mut seen = vec![];
    for _ in 0..4 {
        seen.push(a);
        a = a.next();
    }
    assert_eq!(a, UpAxis::PosY);
    seen.sort_by_key(|u| u.label());
    seen.dedup();
    assert_eq!(seen.len(), 4);
    for u in ALL {
        assert_eq!(u.label().parse::<UpAxis>().unwrap(), u);
        assert!((u.vector().length() - 1.0).abs() < 1e-6);
    }
    assert_eq!("+Y".parse::<UpAxis>().unwrap(), UpAxis::PosY);
    assert_eq!("y".parse::<UpAxis>().unwrap(), UpAxis::PosY);
    assert_eq!("-Z".parse::<UpAxis>().unwrap(), UpAxis::NegZ);
    assert!("".parse::<UpAxis>().is_err());
    assert!("x".parse::<UpAxis>().is_err());
}
