//! End-to-end tests of the `splat-tui` binary.

use std::path::Path;
use std::process::{Command, Output};

use splat_tui::scene::Scene;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_splat-tui"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn splat-tui")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const SCENES: [&str; 5] = ["cube", "knot", "galaxy", "planet", "shballs"];

#[test]
fn gen_list() {
    let o = run(&["gen", "--list"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    for n in SCENES {
        assert!(out.lines().any(|l| l.starts_with(n)), "{n} missing in {out}");
    }
    assert_eq!(out.lines().count(), SCENES.len());
}

#[test]
fn gen_single_scene_both_formats_and_reload() {
    let dir = tempfile::tempdir().unwrap();
    for (fmt, ext) in [("ply", "ply"), ("splat", "splat")] {
        let out_dir = dir.path().join(fmt).join("nested");
        let o = run(&["gen", "cube", "-o", out_dir.to_str().unwrap(), "--format", fmt]);
        assert!(o.status.success(), "{}", stderr(&o));
        let p = out_dir.join(format!("cube.{ext}"));
        assert!(p.exists());
        assert!(stdout(&o).contains("3409 gaussians"), "{}", stdout(&o));
        let s = Scene::load(&p).unwrap();
        assert_eq!(s.len(), 3409);
    }
}

#[test]
fn gen_all_writes_every_scene() {
    let dir = tempfile::tempdir().unwrap();
    let o = run(&["gen", "-o", dir.path().to_str().unwrap(), "--format", "splat"]);
    assert!(o.status.success(), "{}", stderr(&o));
    for n in SCENES {
        let p = dir.path().join(format!("{n}.splat"));
        assert!(Scene::load(&p).unwrap().len() > 0, "{n}");
    }
    // shballs has SH: a note about lost view-dependent color goes to stderr.
    assert!(stderr(&o).contains("shballs"), "{}", stderr(&o));
}

#[test]
fn gen_ply_preserves_sh() {
    let dir = tempfile::tempdir().unwrap();
    let o = run(&["gen", "shballs", "-o", dir.path().to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    let s = Scene::load(&dir.path().join("shballs.ply")).unwrap();
    assert_eq!(s.sh_degree, 3);
}

#[test]
fn gen_unknown_scene_fails() {
    let dir = tempfile::tempdir().unwrap();
    let o = run(&["gen", "nope", "-o", dir.path().to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("unknown scene"), "{}", stderr(&o));
}

fn is_png(p: &Path) -> (u32, u32) {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&b[12..16], b"IHDR");
    let w = u32::from_be_bytes(b[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(b[20..24].try_into().unwrap());
    assert_eq!(&b[b.len() - 8..b.len() - 4], b"IEND");
    (w, h)
}

#[test]
fn render_cube_png() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x.png");
    let o = run(&["render", "gen:cube", "-o", out.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(is_png(&out), (640, 360));
    assert!(stdout(&o).contains("wrote"));
}

#[test]
fn render_options_and_backends() {
    let dir = tempfile::tempdir().unwrap();
    for (i, backend) in ["cpu", "cpu-naive"].iter().enumerate() {
        let out = dir.path().join(format!("{i}.png"));
        let o = run(&[
            "render", "gen:knot", "-o", out.to_str().unwrap(), "--size", "33x17", "--backend", backend,
            "--yaw", "-45", "--pitch", "-20", "--zoom", "2", "--up", "-y",
        ]);
        assert!(o.status.success(), "{backend}: {}", stderr(&o));
        assert_eq!(is_png(&out), (33, 17));
    }
    // Out-of-range pitch is clamped rather than producing NaNs.
    let out = dir.path().join("p.png");
    let o = run(&["render", "gen:cube", "-o", out.to_str().unwrap(), "--size", "8x8", "--pitch", "90"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(is_png(&out), (8, 8));
    // 1x1.
    let o = run(&["render", "gen:cube", "-o", out.to_str().unwrap(), "--size", "1x1"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(is_png(&out), (1, 1));
}

#[test]
fn render_loaded_file() {
    let dir = tempfile::tempdir().unwrap();
    let o = run(&["gen", "cube", "-o", dir.path().to_str().unwrap(), "--format", "splat"]);
    assert!(o.status.success());
    let scene = dir.path().join("cube.splat");
    let out = dir.path().join("o.png");
    let o = run(&["render", scene.to_str().unwrap(), "-o", out.to_str().unwrap(), "--size", "20x20"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(is_png(&out), (20, 20));
}

#[test]
fn render_error_cases() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x.png");
    let out = out.to_str().unwrap();
    let cases: Vec<Vec<&str>> = vec![
        vec!["render", "/definitely/not/here.ply", "-o", out],
        vec!["render", "gen:nope", "-o", out],
        vec!["render", "gen:cube", "-o", out, "--size", "abc"],
        vec!["render", "gen:cube", "-o", out, "--size", "10"],
        vec!["render", "gen:cube", "-o", out, "--size", "0x10"],
        vec!["render", "gen:cube", "-o", out, "--size", "10x0"],
        vec!["render", "gen:cube", "-o", out, "--size", "99999x10"],
        vec!["render", "gen:cube", "-o", out, "--size", "-5x10"],
        vec!["render", "gen:cube", "-o", out, "--zoom", "0"],
        vec!["render", "gen:cube", "-o", out, "--zoom", "-2"],
        vec!["render", "gen:cube", "-o", out, "--up", "w"],
        vec!["render", "gen:cube"],
        vec!["bench", "gen:cube", "--size", "bogus"],
        vec!["bench", "/no/such/file.splat"],
        vec!["bogus-subcommand-or-file.ply"],
    ];
    let mut cases = cases;
    if !cfg!(feature = "gpu") {
        cases.push(vec!["render", "gen:cube", "-o", out, "--backend", "gpu"]);
    }
    for args in cases {
        let o = run(&args);
        assert!(!o.status.success(), "{args:?} should fail; stdout {}", stdout(&o));
        assert!(!stderr(&o).is_empty(), "{args:?}: no error message");
        assert!(!stderr(&o).contains("panicked"), "{args:?}: {}", stderr(&o));
    }
    assert!(!Path::new(out).exists());
}

#[test]
fn render_bad_files_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x.png");
    for (name, bytes) in [
        ("bad.splat", vec![0u8; 33]),
        ("empty.splat", vec![]),
        ("bad.ply", b"ply\nformat ascii 1.0\nelement vertex 99999999999\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n".to_vec()),
        ("garbage.txt", b"hello".to_vec()),
    ] {
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        let o = run(&["render", p.to_str().unwrap(), "-o", out.to_str().unwrap()]);
        assert!(!o.status.success(), "{name}");
        assert!(!stderr(&o).contains("panicked"), "{name}: {}", stderr(&o));
    }
}

#[test]
fn bench_cpu_two_frames() {
    let o = run(&["bench", "gen:cube", "--backend", "cpu", "--frames", "2", "--warmup", "0", "--size", "32x18"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    assert!(out.contains("gen:cube: 3409 gaussians"), "{out}");
    assert!(out.contains("32x18 px, 2 frames"), "{out}");
    let row = out.lines().find(|l| l.starts_with("cpu ")).expect("cpu row");
    assert!(row.trim_end().ends_with("ref"), "{row}");
}

#[test]
fn bench_compares_backends_with_psnr() {
    let o = run(&["bench", "gen:knot", "--backend", "cpu,cpu-naive", "--frames", "1", "--warmup", "1", "--size", "64x36"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    let row = out.lines().find(|l| l.starts_with("cpu-naive")).expect("naive row");
    let psnr = row.split_whitespace().last().unwrap();
    assert!(psnr == "inf" || psnr.parse::<f64>().unwrap() >= 45.0, "{row}");
}

#[test]
fn bench_zero_frames_does_not_panic() {
    let o = run(&["bench", "gen:cube", "--backend", "cpu", "--frames", "0", "--warmup", "0", "--size", "16x16"]);
    assert!(o.status.success(), "{} {}", stdout(&o), stderr(&o));
    assert!(!stdout(&o).contains("NaN"), "{}", stdout(&o));
}

#[test]
fn bench_gpu_reports_unavailable_without_feature() {
    if cfg!(feature = "gpu") {
        return;
    }
    let o = run(&["bench", "gen:cube", "--backend", "gpu", "--frames", "1", "--warmup", "0", "--size", "16x16"]);
    assert!(o.status.success());
    assert!(stdout(&o).contains("unavailable"), "{}", stdout(&o));
}

#[test]
fn view_without_scene_errors_and_help_version_work() {
    let o = run(&[]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("no scene"), "{}", stderr(&o));
    for a in ["--help", "--version", "render --help", "bench --help", "gen --help"] {
        let args: Vec<&str> = a.split(' ').collect();
        let o = run(&args);
        assert!(o.status.success(), "{a}");
        assert!(!stdout(&o).is_empty());
    }
}
