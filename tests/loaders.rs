//! Integration tests for the PLY and `.splat` loaders.

use glam::{Quat, Vec3};
use splat_tui::scene::{sigmoid, Scene};
use splat_tui::sh::SH_C0;
use splat_tui::{ply, splat_format};
use std::path::Path;

#[derive(Clone, Copy)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::I8 => "char",
            Ty::U8 => "uchar",
            Ty::I16 => "short",
            Ty::U16 => "ushort",
            Ty::I32 => "int",
            Ty::U32 => "uint",
            Ty::F32 => "float",
            Ty::F64 => "double",
        }
    }
    fn encode(self, v: f64, le: bool, out: &mut Vec<u8>) {
        macro_rules! e {
            ($v:expr) => {{
                let b = $v;
                if le {
                    out.extend_from_slice(&b.to_le_bytes())
                } else {
                    out.extend_from_slice(&b.to_be_bytes())
                }
            }};
        }
        match self {
            Ty::I8 => e!(v as i8),
            Ty::U8 => e!(v as u8),
            Ty::I16 => e!(v as i16),
            Ty::U16 => e!(v as u16),
            Ty::I32 => e!(v as i32),
            Ty::U32 => e!(v as u32),
            Ty::F32 => e!(v as f32),
            Ty::F64 => e!(v),
        }
    }
}

/// Build a PLY with one vertex element with the given (name, type) props and rows.
fn build(format: &str, nl: &str, props: &[(&str, Ty)], rows: &[Vec<f64>]) -> Vec<u8> {
    let mut h = format!("ply{nl}format {format} 1.0{nl}element vertex {}{nl}", rows.len());
    for (n, t) in props {
        h.push_str(&format!("property {} {n}{nl}", t.name()));
    }
    h.push_str(&format!("end_header{nl}"));
    let mut out = h.into_bytes();
    for r in rows {
        match format {
            "ascii" => {
                let s: Vec<String> = r.iter().map(|v| format!("{v}")).collect();
                out.extend_from_slice(format!("{}\n", s.join(" ")).as_bytes());
            }
            _ => {
                for ((_, t), v) in props.iter().zip(r) {
                    t.encode(*v, format == "binary_little_endian", &mut out);
                }
            }
        }
    }
    out
}

fn xyz() -> Vec<(&'static str, Ty)> {
    vec![("x", Ty::F32), ("y", Ty::F32), ("z", Ty::F32)]
}

#[test]
fn binary_le_be_ascii_agree() {
    let props: Vec<(&str, Ty)> = xyz()
        .into_iter()
        .chain([
            ("f_dc_0", Ty::F32),
            ("f_dc_1", Ty::F32),
            ("f_dc_2", Ty::F32),
            ("opacity", Ty::F32),
            ("scale_0", Ty::F32),
            ("scale_1", Ty::F32),
            ("scale_2", Ty::F32),
            ("rot_0", Ty::F32),
            ("rot_1", Ty::F32),
            ("rot_2", Ty::F32),
            ("rot_3", Ty::F32),
        ])
        .collect();
    let rows = vec![
        vec![1.0, 2.0, 3.0, 0.5, -0.5, 1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0],
        vec![-4.0, 5.5, 6.25, 0.0, 0.0, 0.0, 2.0, 0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3],
    ];
    let mut scenes = Vec::new();
    for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
        scenes.push(ply::parse(&build(f, "\n", &props, &rows)).unwrap());
    }
    for s in &scenes {
        assert_eq!(s.len(), 2);
        assert_eq!(s.positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.positions[1], Vec3::new(-4.0, 5.5, 6.25));
        let c = Vec3::new(0.5, -0.5, 1.0) * SH_C0 + Vec3::splat(0.5);
        assert!((s.colors[0] - c).length() < 1e-5);
        assert!((s.opacities[0] - sigmoid(0.0)).abs() < 1e-6);
        // scale_* = (-1, 0, 0.5) is exp-activated.
        assert!((s.scales[0] - Vec3::new((-1.0f32).exp(), 1.0, 0.5f32.exp())).length() < 1e-5, "{:?}", s.scales[0]);
        assert_eq!(s.sh_degree, 0);
    }
    for s in &scenes[1..] {
        for i in 0..2 {
            assert!((s.positions[i] - scenes[0].positions[i]).length() < 1e-5);
            assert!((s.colors[i] - scenes[0].colors[i]).length() < 1e-5);
            assert!((s.scales[i] - scenes[0].scales[i]).length() < 1e-5);
            assert!((s.rotations[i] - scenes[0].rotations[i]).length() < 1e-5);
        }
    }
}

#[test]
fn quaternion_is_wxyz() {
    let mut props = xyz();
    props.extend([("rot_0", Ty::F32), ("rot_1", Ty::F32), ("rot_2", Ty::F32), ("rot_3", Ty::F32)]);
    let s = ply::parse(&build("binary_little_endian", "\n", &props, &[vec![0., 0., 0., 0.1, 0.2, 0.3, 0.4]])).unwrap();
    let q = s.rotations[0];
    assert!((q.w - 0.1).abs() < 1e-6 && (q.x - 0.2).abs() < 1e-6 && (q.y - 0.3).abs() < 1e-6 && (q.z - 0.4).abs() < 1e-6);
}

#[test]
fn all_scalar_types_for_position_and_rgb() {
    for t in [Ty::I8, Ty::I16, Ty::U16, Ty::I32, Ty::U32, Ty::F32, Ty::F64] {
        for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
            let props = vec![("x", t), ("y", t), ("z", t)];
            let s = ply::parse(&build(f, "\n", &props, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])).unwrap();
            assert_eq!(s.positions[1], Vec3::new(4.0, 5.0, 6.0), "{} {f}", t.name());
        }
    }
    // Negative values in signed types.
    for t in [Ty::I8, Ty::I16, Ty::I32, Ty::F32, Ty::F64] {
        let props = vec![("x", t), ("y", t), ("z", t)];
        let s = ply::parse(&build("binary_big_endian", "\n", &props, &[vec![-1.0, -2.0, -3.0]])).unwrap();
        assert_eq!(s.positions[0], Vec3::new(-1.0, -2.0, -3.0));
    }
    // uchar rgb normalized to [0,1]; float rgb used as is.
    let mut props = xyz();
    props.extend([("red", Ty::U8), ("green", Ty::U8), ("blue", Ty::U8)]);
    let s = ply::parse(&build("binary_little_endian", "\n", &props, &[vec![0., 0., 0., 255., 0., 51.]])).unwrap();
    assert!((s.colors[0] - Vec3::new(1.0, 0.0, 0.2)).length() < 1e-6);
    let mut props = xyz();
    props.extend([("red", Ty::F32), ("green", Ty::F32), ("blue", Ty::F32)]);
    let s = ply::parse(&build("binary_little_endian", "\n", &props, &[vec![0., 0., 0., 0.25, 0.5, 0.75]])).unwrap();
    assert!((s.colors[0] - Vec3::new(0.25, 0.5, 0.75)).length() < 1e-6);
}

#[test]
fn mixed_type_rows_have_correct_offsets() {
    let props = vec![
        ("x", Ty::F64),
        ("pad", Ty::U8),
        ("y", Ty::I16),
        ("junk", Ty::U32),
        ("z", Ty::F32),
        ("opacity", Ty::F32),
    ];
    let rows = vec![vec![1.5, 7., -2., 99., 3.5, 0.0], vec![-1.5, 8., 20., 98., 4.5, 0.0]];
    for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
        let s = ply::parse(&build(f, "\n", &props, &rows)).unwrap();
        assert_eq!(s.positions[0], Vec3::new(1.5, -2.0, 3.5), "{f}");
        assert_eq!(s.positions[1], Vec3::new(-1.5, 20.0, 4.5), "{f}");
    }
}

#[test]
fn defaults_for_missing_optional_properties() {
    let s = ply::parse(&build("binary_little_endian", "\n", &xyz(), &[vec![1., 2., 3.]])).unwrap();
    assert_eq!(s.opacities[0], 1.0);
    assert_eq!(s.scales[0], Vec3::splat(0.01));
    assert_eq!(s.rotations[0], Quat::IDENTITY);
    assert_eq!(s.colors[0], Vec3::splat(0.8));
    assert_eq!(s.sh_degree, 0);
    assert!(s.sh_rest.is_empty());
}

#[test]
fn missing_position_is_error() {
    let props = vec![("x", Ty::F32), ("y", Ty::F32)];
    assert!(ply::parse(&build("binary_little_endian", "\n", &props, &[vec![1., 2.]])).is_err());
}

#[test]
fn sh_rest_is_channel_major() {
    for (degree, rest_count) in [(1u32, 9usize), (2, 24), (3, 45)] {
        let per = rest_count / 3;
        let mut props = xyz();
        let names: Vec<String> = (0..rest_count).map(|i| format!("f_rest_{i}")).collect();
        for n in &names {
            props.push((n.as_str(), Ty::F32));
        }
        // value = channel*100 + coefficient index
        let mut row = vec![0.0, 0.0, 0.0];
        for i in 0..rest_count {
            row.push(((i / per) * 100 + i % per) as f64);
        }
        for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
            let s = ply::parse(&build(f, "\n", &props, &[row.clone(), row.clone()])).unwrap();
            assert_eq!(s.sh_degree, degree);
            assert_eq!(s.rest_per_splat(), per);
            assert_eq!(s.sh_rest.len(), 2 * per);
            for k in 0..per {
                assert_eq!(s.sh_rest[k], Vec3::new(k as f32, 100.0 + k as f32, 200.0 + k as f32), "deg {degree} k {k} {f}");
                assert_eq!(s.sh_rest[per + k], s.sh_rest[k]);
            }
        }
    }
}

#[test]
fn sh_with_unsupported_rest_count_falls_back_to_degree_zero() {
    let mut props = xyz();
    let names: Vec<String> = (0..10).map(|i| format!("f_rest_{i}")).collect();
    for n in &names {
        props.push((n.as_str(), Ty::F32));
    }
    let mut s = ply::parse(&build("binary_little_endian", "\n", &props, &[vec![0.0; 13]])).unwrap();
    assert_eq!(s.sh_degree, 0);
    s.finalize().unwrap();
}

#[test]
fn extra_unknown_properties_are_ignored() {
    let mut props = xyz();
    props.extend([("nx", Ty::F32), ("weird_prop", Ty::F64), ("opacity", Ty::F32), ("label", Ty::U8)]);
    let s = ply::parse(&build("binary_little_endian", "\n", &props, &[vec![1., 2., 3., 9., 9., 2.0, 7.]])).unwrap();
    assert_eq!(s.positions[0], Vec3::new(1., 2., 3.));
    assert!((s.opacities[0] - sigmoid(2.0)).abs() < 1e-6);
}

#[test]
fn crlf_header() {
    for f in ["binary_little_endian", "ascii"] {
        let s = ply::parse(&build(f, "\r\n", &xyz(), &[vec![1., 2., 3.], vec![4., 5., 6.]])).unwrap();
        assert_eq!(s.len(), 2, "{f}");
        assert_eq!(s.positions[1], Vec3::new(4., 5., 6.), "{f}");
    }
}

#[test]
fn comments_and_obj_info_are_accepted() {
    let mut data = b"ply\nformat ascii 1.0\ncomment hi there\nobj_info foo\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n".to_vec();
    assert_eq!(ply::parse(&data).unwrap().len(), 1);
    data.extend_from_slice(b"trailing garbage ignored? \n");
    assert_eq!(ply::parse(&data).unwrap().len(), 1);
}

#[test]
fn fixed_size_element_before_vertex_is_skipped() {
    let mut data = b"ply\nformat binary_little_endian 1.0\nelement chunk 2\nproperty float a\nproperty uchar b\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n".to_vec();
    data.extend_from_slice(&[0xEE; 10]); // 2 * (4+1)
    for v in [1f32, 2., 3., 4., 5., 6.] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let s = ply::parse(&data).unwrap();
    assert_eq!(s.positions[0], Vec3::new(1., 2., 3.));
    assert_eq!(s.positions[1], Vec3::new(4., 5., 6.));
}

#[test]
fn element_after_vertex_is_ignored() {
    let mut data = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n".to_vec();
    for v in [1f32, 2., 3.] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(ply::parse(&data).unwrap().len(), 1);
}

#[test]
fn element_before_vertex_with_list_or_ascii_is_error_not_panic() {
    let data = b"ply\nformat binary_little_endian 1.0\nelement face 1\nproperty list uchar int vi\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    assert!(ply::parse(data).is_err());
    let data = b"ply\nformat ascii 1.0\nelement chunk 1\nproperty float a\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1\n1 2 3\n";
    assert!(ply::parse(data).is_err());
}

#[test]
fn huge_element_before_vertex_is_error() {
    let data = format!("ply\nformat binary_little_endian 1.0\nelement chunk {}\nproperty double a\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n", usize::MAX);
    assert!(ply::parse(data.as_bytes()).is_err());
    let data = format!("ply\nformat binary_little_endian 1.0\nelement chunk 1000\nproperty double a\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
    assert!(ply::parse(data.as_bytes()).is_err());
}

#[test]
fn truncated_body_is_error() {
    let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0., 0.]).collect();
    for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
        let full = build(f, "\n", &xyz(), &rows);
        for cut in [1usize, 3, 4, 12, 50] {
            let mut d = full.clone();
            d.truncate(d.len() - cut);
            let r = ply::parse(&d);
            if f == "ascii" && cut == 1 {
                continue; // dropping only the final newline is fine
            }
            if f == "ascii" && cut < 3 {
                continue;
            }
            assert!(r.is_err(), "{f} cut {cut}");
        }
    }
}

#[test]
fn lying_vertex_counts_error_without_panic() {
    for f in ["binary_little_endian", "binary_big_endian", "ascii"] {
        for count in [
            "1000000".to_string(),
            "4000000000".to_string(),
            "1000000000000".to_string(),
            u64::MAX.to_string(),
            usize::MAX.to_string(),
        ] {
            let data = format!(
                "ply\nformat {f} 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n"
            );
            // Must neither panic, abort on a huge allocation, nor succeed.
            let r = std::panic::catch_unwind(|| ply::parse(data.as_bytes()));
            assert!(matches!(r, Ok(Err(_))), "{f} count {count}: {:?}", r.map(|r| r.map(|s| s.len())));
        }
    }
}

#[test]
fn bad_counts_and_headers_are_errors() {
    for h in [
        "ply\nformat binary_little_endian 1.0\nelement vertex -3\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nelement vertex abc\nproperty float x\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty banana x\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nproperty float x\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nelement vertex\nend_header\n",
        "ply\nformat utf16 1.0\nelement vertex 1\nproperty float x\nend_header\n",
        "ply\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n\0\0\0\0\0\0\0\0\0\0\0\0",
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty list uchar\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nbogus line\nend_header\n",
        "ply\nformat binary_little_endian 1.0\nelement face 1\nproperty list uchar int vi\nend_header\n\x00",
        "",
        "ply",
        "ply\n",
        "plyx\nformat ascii 1.0\nend_header\n",
    ] {
        assert!(ply::parse(h.as_bytes()).is_err(), "{h:?}");
    }
}

#[test]
fn missing_end_header_is_error() {
    let d = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n1 2 3\n";
    assert!(ply::parse(d).is_err());
    // Only a little binary junk.
    assert!(ply::parse(&[0x70, 0x6c, 0x79, 0x0a, 0xff, 0xfe]).is_err());
}

#[test]
fn non_utf8_bytes() {
    // In header: error.
    let d = b"ply\nformat ascii 1.0\ncomment \xff\xfe\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n";
    assert!(ply::parse(d).is_err());
    // In ASCII body: error.
    let d = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 \xff\n";
    assert!(ply::parse(d).is_err());
    // In binary body: perfectly fine (raw floats).
    let mut d = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n".to_vec();
    d.extend_from_slice(&[0xff, 0xfe, 0x80, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(ply::parse(&d).unwrap().len(), 1);
    // Entirely random bytes.
    let junk: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    assert!(ply::parse(&junk).is_err());
}

#[test]
fn compressed_ply_rejected() {
    let props = vec![("packed_position", Ty::U32), ("x", Ty::F32), ("y", Ty::F32), ("z", Ty::F32)];
    assert!(ply::parse(&build("binary_little_endian", "\n", &props, &[vec![0., 0., 0., 0.]])).is_err());
}

#[test]
fn ascii_garbage_number_is_error() {
    let d = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 two 3\n";
    assert!(ply::parse(d).is_err());
}

#[test]
fn zero_vertices_parses_empty_and_load_rejects() {
    let d = build("binary_little_endian", "\n", &xyz(), &[]);
    assert_eq!(ply::parse(&d).unwrap().len(), 0);
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("empty.ply");
    std::fs::write(&p, &d).unwrap();
    assert!(Scene::load(&p).is_err());
}

#[test]
fn ply_write_parse_roundtrip_with_sh_and_load() {
    let mut s = Scene::default();
    s.sh_degree = 2;
    for i in 0..5 {
        let f = i as f32;
        s.positions.push(Vec3::new(f, -f, 2.0 * f));
        s.scales.push(Vec3::new(0.1, 0.2, 0.3) * (f + 1.0));
        s.rotations.push(Quat::from_rotation_z(f * 0.3));
        s.opacities.push(0.1 + 0.15 * f);
        s.colors.push(Vec3::new(0.1 * f, 0.5, 1.0 - 0.1 * f));
    }
    s.sh_rest = (0..5 * 8).map(|i| Vec3::new(i as f32, -(i as f32), 0.5 * i as f32)).collect();
    let mut bytes = Vec::new();
    ply::write(&s, &mut bytes).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.PLY");
    std::fs::write(&p, &bytes).unwrap();
    let r = Scene::load(&p).unwrap();
    assert_eq!(r.len(), 5);
    assert_eq!(r.sh_degree, 2);
    assert_eq!(r.sh_rest, s.sh_rest);
    assert_eq!(r.cov3d.len(), 5);
    for i in 0..5 {
        assert!((r.opacities[i] - s.opacities[i]).abs() < 1e-5);
        assert!((r.colors[i] - s.colors[i]).length() < 1e-5);
    }
}

// ---------------- .splat ----------------

fn sample_scene(n: usize) -> Scene {
    let mut s = Scene::default();
    for i in 0..n {
        let f = i as f32;
        s.push(
            Vec3::new(f, -f * 0.5, 3.0),
            Vec3::new(0.1, 0.05, 0.2) * (1.0 + f),
            Quat::from_rotation_y(0.2 * f),
            (0.1 * f).min(1.0),
            Vec3::new(0.2, (0.1 * f).min(1.0), 0.9),
        );
    }
    s
}

#[test]
fn splat_roundtrip_many() {
    let s = sample_scene(10);
    let bytes = splat_format::write(&s);
    assert_eq!(bytes.len(), 320);
    let r = splat_format::parse(&bytes).unwrap();
    assert_eq!(r.len(), 10);
    for i in 0..10 {
        assert_eq!(r.positions[i], s.positions[i]);
        assert_eq!(r.scales[i], s.scales[i]);
        assert!((r.opacities[i] - s.opacities[i]).abs() <= 0.5 / 255.0 + 1e-6);
        assert!((r.colors[i] - s.colors[i]).abs().max_element() <= 0.5 / 255.0 + 1e-6);
        assert!(r.rotations[i].normalize().angle_between(s.rotations[i]) < 0.03);
    }
}

#[test]
fn splat_wrong_sizes() {
    assert!(splat_format::parse(&[]).unwrap().is_empty());
    for n in [1usize, 31, 33, 63, 65, 100] {
        assert!(splat_format::parse(&vec![0u8; n]).is_err(), "{n}");
    }
    assert_eq!(splat_format::parse(&[0u8; 64]).unwrap().len(), 2);
}

#[test]
fn splat_clamps_out_of_range_values() {
    let mut s = Scene::default();
    s.push(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 2.0, Vec3::new(-1.0, 3.0, 0.5));
    let r = splat_format::parse(&splat_format::write(&s)).unwrap();
    assert_eq!(r.opacities[0], 1.0);
    assert_eq!(r.colors[0].x, 0.0);
    assert_eq!(r.colors[0].y, 1.0);
}

#[test]
fn splat_identity_rotation_decodes_close_to_identity() {
    let mut s = Scene::default();
    s.push(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
    let r = splat_format::parse(&splat_format::write(&s)).unwrap();
    assert!(r.rotations[0].normalize().angle_between(Quat::IDENTITY) < 0.02);
}

#[test]
fn scene_load_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let s = sample_scene(3);
    let p = dir.path().join("a.splat");
    std::fs::write(&p, splat_format::write(&s)).unwrap();
    assert_eq!(Scene::load(&p).unwrap().len(), 3);
    let up = dir.path().join("UP.SPLAT");
    std::fs::write(&up, splat_format::write(&s)).unwrap();
    assert_eq!(Scene::load(&up).unwrap().len(), 3);
    // Unknown extension and not PLY.
    let q = dir.path().join("a.bin");
    std::fs::write(&q, splat_format::write(&s)).unwrap();
    assert!(Scene::load(&q).is_err());
    // Unknown extension but PLY magic.
    let mut b = Vec::new();
    ply::write(&s, &mut b).unwrap();
    let q = dir.path().join("noext");
    std::fs::write(&q, &b).unwrap();
    assert_eq!(Scene::load(&q).unwrap().len(), 3);
    // Missing file.
    assert!(Scene::load(&dir.path().join("nope.ply")).is_err());
    // Empty .splat.
    let e = dir.path().join("e.splat");
    std::fs::write(&e, []).unwrap();
    assert!(Scene::load(&e).is_err());
}

#[test]
fn finalize_sanitizes_values() {
    let mut s = Scene::default();
    s.push(Vec3::ZERO, Vec3::ONE, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), f32::NAN, Vec3::ONE);
    s.push(Vec3::ZERO, Vec3::ONE, Quat::from_xyzw(0.0, 0.0, 5.0, 0.0), 7.0, Vec3::ONE);
    s.finalize().unwrap();
    assert_eq!(s.rotations[0], Quat::IDENTITY);
    assert!((s.rotations[1].length() - 1.0).abs() < 1e-6);
    assert_eq!(s.opacities, vec![0.0, 1.0]);
    assert!(s.cov3d.iter().all(|c| c.iter().all(|v| v.is_finite())));
}

#[test]
fn bounds_edge_cases() {
    let b = Scene::default().bounds();
    assert_eq!((b.center, b.radius), (Vec3::ZERO, 1.0));
    // All points identical: radius falls back to 1.
    let mut s = Scene::default();
    for _ in 0..10 {
        s.push(Vec3::splat(2.0), Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
    }
    let b = s.bounds();
    assert_eq!(b.center, Vec3::splat(2.0));
    assert_eq!(b.radius, 1.0);
    // Non-finite positions are ignored.
    let mut s = Scene::default();
    s.push(Vec3::splat(f32::NAN), Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
    s.push(Vec3::splat(f32::INFINITY), Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
    assert_eq!(s.bounds().radius, 1.0);
    for i in 0..20 {
        s.push(Vec3::new(i as f32, 0.0, 0.0), Vec3::ONE, Quat::IDENTITY, 1.0, Vec3::ONE);
    }
    let b = s.bounds();
    assert!(b.center.is_finite() && b.radius.is_finite() && b.radius > 1.0);
}

// ---------------- real files ----------------

#[test]
fn real_splat_files_load_with_sane_bounds() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("scenes/downloaded");
    for name in ["nike", "plush", "train"] {
        let p = dir.join(format!("{name}.splat"));
        if !p.exists() {
            eprintln!("skipping {}: not present", p.display());
            continue;
        }
        let s = Scene::load(&p).unwrap();
        assert!(s.len() > 1000, "{name}: {}", s.len());
        assert_eq!(s.cov3d.len(), s.len());
        assert!(s.opacities.iter().all(|o| (0.0..=1.0).contains(o)));
        assert!(s.colors.iter().all(|c| c.is_finite()));
        let b = s.bounds();
        assert!(b.center.is_finite(), "{name}");
        assert!(b.radius.is_finite() && b.radius > 1e-4 && b.radius < 1e5, "{name}: {}", b.radius);
    }
}
