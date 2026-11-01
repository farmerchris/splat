//! Rendering integration tests: backend agreement and basic image semantics.

use std::sync::Arc;

use glam::{Quat, Vec3};
use splat_tui::bench::psnr;
use splat_tui::camera::{OrbitCamera, UpAxis, View};
use splat_tui::gen;
use splat_tui::render::{self, Backend, Frame};
use splat_tui::scene::Scene;

fn view_at(eye: Vec3) -> View {
    // Looking along +z from `eye` (camera space convention: x right, y down).
    View {
        eye,
        right: Vec3::X,
        down: Vec3::Y,
        forward: Vec3::Z,
        fov_y: 50f32.to_radians(),
        near: 0.01,
    }
}

fn scene_of(items: &[(Vec3, f32, f32, Vec3)]) -> Arc<Scene> {
    let mut s = Scene::default();
    for &(p, scale, op, c) in items {
        s.push(p, Vec3::splat(scale), Quat::IDENTITY, op, c);
    }
    s.finalize().unwrap();
    Arc::new(s)
}

fn render_with(b: Backend, s: &Arc<Scene>, v: &View, w: u32, h: u32, bg: Vec3) -> Frame {
    let mut r = render::create(b, s.clone()).unwrap();
    let mut f = Frame::default();
    r.render(v, w, h, bg, &mut f);
    assert_eq!((f.width, f.height), (w, h));
    assert_eq!(f.pixels.len(), (w * h) as usize);
    f
}

const RED: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const BLUE: Vec3 = Vec3::new(0.0, 0.0, 1.0);

#[test]
fn backends_agree_on_odd_sizes_and_poses() {
    for name in ["cube", "knot", "galaxy"] {
        let scene = Arc::new(gen::generate(name).unwrap());
        for (w, h) in [(1, 1), (17, 9), (16, 16), (33, 31), (64, 48), (333, 77)] {
            for (yaw, pitch, up) in [
                (0.37f32, 0.26f32, UpAxis::NegY),
                (1.3, -0.5, UpAxis::NegY),
                (4.0, 0.9, UpAxis::PosY),
                (2.0, 0.1, UpAxis::PosZ),
                (5.5, 0.3, UpAxis::NegZ),
            ] {
                let mut cam = OrbitCamera::new(scene.bounds(), up);
                cam.rotate(yaw, pitch - cam.pitch);
                let v = cam.view();
                let a = render_with(Backend::Cpu, &scene, &v, w, h, Vec3::new(0.1, 0.2, 0.3));
                let b = render_with(Backend::CpuNaive, &scene, &v, w, h, Vec3::new(0.1, 0.2, 0.3));
                let p = psnr(&a, &b);
                assert!(p >= 45.0, "{name} {w}x{h} yaw {yaw} pitch {pitch} {up:?}: psnr {p}");
            }
        }
    }
}

#[test]
fn backends_agree_on_sh_scene() {
    let scene = Arc::new(gen::generate("shballs").unwrap());
    let cam = OrbitCamera::new(scene.bounds(), UpAxis::NegY);
    let a = render_with(Backend::Cpu, &scene, &cam.view(), 97, 53, Vec3::ZERO);
    let b = render_with(Backend::CpuNaive, &scene, &cam.view(), 97, 53, Vec3::ZERO);
    assert!(psnr(&a, &b) >= 45.0);
}

#[test]
fn single_red_splat_center_red_corners_background() {
    let s = scene_of(&[(Vec3::new(0.0, 0.0, 5.0), 0.5, 1.0, RED)]);
    let v = view_at(Vec3::ZERO);
    let bg = Vec3::new(0.0, 0.0, 0.0);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        for (w, h) in [(64u32, 64u32), (65, 33), (200, 100)] {
            let f = render_with(b, &s, &v, w, h, bg);
            let c = f.get(w / 2, h / 2);
            assert!(c[0] > 200 && c[1] < 10 && c[2] < 10, "{b:?} {w}x{h} center {c:?}");
            for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
                assert_eq!(f.get(x, y), [0, 0, 0], "{b:?} {w}x{h} corner {x},{y}");
            }
        }
    }
}

#[test]
fn background_color_shows_through() {
    let s = scene_of(&[(Vec3::new(0.0, 0.0, 5.0), 0.5, 1.0, RED)]);
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let f = render_with(b, &s, &v, 64, 64, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(f.get(0, 0), [0, 255, 0]);
    }
}

#[test]
fn splat_behind_camera_renders_nothing() {
    let s = scene_of(&[(Vec3::new(0.0, 0.0, -5.0), 0.5, 1.0, RED)]);
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let mut r = render::create(b, s.clone()).unwrap();
        let mut f = Frame::default();
        let st = r.render(&v, 40, 30, Vec3::splat(0.5), &mut f);
        assert_eq!(st.visible, 0, "{b:?}");
        assert!(f.pixels.iter().all(|p| *p == [128, 128, 128]), "{b:?}");
    }
}

#[test]
fn off_screen_and_transparent_splats_give_pure_background() {
    let s = scene_of(&[
        (Vec3::new(500.0, 0.0, 5.0), 0.5, 1.0, RED),
        (Vec3::new(0.0, 0.0, 5.0), 0.5, 0.0, RED),
    ]);
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let f = render_with(b, &s, &v, 50, 50, Vec3::new(0.2, 0.4, 0.6));
        let first = f.pixels[0];
        assert!(f.pixels.iter().all(|p| *p == first), "{b:?}");
    }
}

#[test]
fn empty_scene_is_background() {
    let s = Arc::new(Scene::default());
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let f = render_with(b, &s, &v, 20, 10, Vec3::ONE);
        assert!(f.pixels.iter().all(|p| *p == [255, 255, 255]));
    }
}

#[test]
fn near_opaque_blue_hides_far_red_regardless_of_insertion_order() {
    let blue = (Vec3::new(0.0, 0.0, 4.0), 0.5, 1.0, BLUE);
    let red = (Vec3::new(0.0, 0.0, 8.0), 1.0, 1.0, RED);
    let v = view_at(Vec3::ZERO);
    for order in [[blue, red], [red, blue]] {
        let s = scene_of(&order);
        for b in [Backend::Cpu, Backend::CpuNaive] {
            let f = render_with(b, &s, &v, 64, 64, Vec3::ZERO);
            let c = f.get(32, 32);
            assert!(c[2] > 200 && c[0] < 10, "{b:?} {order:?} center {c:?}");
        }
    }
}

#[test]
fn translucent_front_blends_with_back() {
    let s = scene_of(&[
        (Vec3::new(0.0, 0.0, 4.0), 0.5, 0.5, BLUE),
        (Vec3::new(0.0, 0.0, 8.0), 2.0, 1.0, RED),
    ]);
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let c = render_with(b, &s, &v, 64, 64, Vec3::ZERO).get(32, 32);
        assert!(c[0] > 40 && c[2] > 40, "{b:?} {c:?}");
    }
}

#[test]
fn splat_moves_with_camera_translation() {
    // A splat to the right of the view axis appears in the right half (x right).
    let s = scene_of(&[(Vec3::new(1.0, 0.0, 5.0), 0.2, 1.0, RED)]);
    let v = view_at(Vec3::ZERO);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let f = render_with(b, &s, &v, 100, 100, Vec3::ZERO);
        let (mut lo, mut hi) = (0u32, 0u32);
        for y in 0..100 {
            for x in 0..100 {
                if f.get(x, y)[0] > 100 {
                    if x < 50 { lo += 1 } else { hi += 1 }
                }
            }
        }
        assert!(hi > 0 && lo == 0, "{b:?} lo {lo} hi {hi}");
    }
}

#[test]
fn repeated_renders_are_deterministic_and_resize_safe() {
    let scene = Arc::new(gen::generate("cube").unwrap());
    let cam = OrbitCamera::new(scene.bounds(), UpAxis::NegY);
    for b in [Backend::Cpu, Backend::CpuNaive] {
        let mut r = render::create(b, scene.clone()).unwrap();
        let mut f = Frame::default();
        r.render(&cam.view(), 48, 32, Vec3::ZERO, &mut f);
        let first = f.clone();
        r.render(&cam.view(), 7, 5, Vec3::ZERO, &mut f);
        assert_eq!((f.width, f.height, f.pixels.len()), (7, 5, 35));
        r.render(&cam.view(), 48, 32, Vec3::ZERO, &mut f);
        assert_eq!(f.pixels, first.pixels, "{b:?}");
    }
}

#[test]
fn gpu_backend_unavailable_without_feature() {
    if !cfg!(feature = "gpu") {
        assert!(render::create(Backend::Gpu, Arc::new(Scene::default())).is_err());
    }
}

/// The cube lattice viewed exactly down an axis has many splats with bit-identical
/// depth. Both backends use an unstable sort, so blend order among ties (and thus
/// the image) differs: PSNR is only 21-35 dB at yaw 0.
#[test]
#[ignore = "bug: render backends disagree on exact-depth ties (unstable sorts without a tiebreak, src/render/{cpu,naive}.rs)"]
fn backends_agree_with_exact_depth_ties() {
    let scene = Arc::new(gen::generate("cube").unwrap());
    let cam = OrbitCamera::new(scene.bounds(), UpAxis::NegY);
    for (w, h) in [(17, 9), (64, 48), (333, 77)] {
        let a = render_with(Backend::Cpu, &scene, &cam.view(), w, h, Vec3::ZERO);
        let b = render_with(Backend::CpuNaive, &scene, &cam.view(), w, h, Vec3::ZERO);
        let p = psnr(&a, &b);
        assert!(p >= 45.0, "{w}x{h}: psnr {p}");
    }
}
