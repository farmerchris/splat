use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use glam::Vec3;

use splat_tui::app::{self, ViewerOptions};
use splat_tui::bench::{self, BenchConfig};
use splat_tui::camera::{OrbitCamera, UpAxis};
use splat_tui::render::{self, Backend, Frame};
use splat_tui::scene::Scene;
use splat_tui::term::ColorMode;
use splat_tui::{gen, ply, splat_format};

/// View 3D Gaussian Splatting scenes in the terminal.
///
/// SCENE is a `.ply` (3DGS training output) or `.splat` file, or
/// `gen:<name>` for a built-in procedural scene (see `splat-tui gen --list`).
#[derive(Parser)]
#[command(version, args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    view: ViewArgs,
}

#[derive(clap::Args)]
struct ViewArgs {
    /// Scene to open (.ply, .splat or gen:<name>).
    scene: Option<String>,
    /// Rasterizer backend.
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    /// World axis pointing up (3DGS/COLMAP captures are usually -y).
    #[arg(long, default_value = "-y", allow_hyphen_values = true)]
    up: UpAxis,
    /// Supersampling factor (1-4) for anti-aliasing.
    #[arg(long, default_value_t = 2)]
    supersample: u32,
    /// Terminal color mode (auto-detected by default).
    #[arg(long, value_enum)]
    color: Option<ColorMode>,
}

#[derive(Subcommand)]
enum Command {
    /// Open the interactive viewer (default when a scene is given).
    View(ViewArgs),
    /// Write procedural example scenes to disk.
    Gen {
        /// Scene name or `all`.
        #[arg(default_value = "all")]
        name: String,
        /// Output directory.
        #[arg(short, long, default_value = "scenes")]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = OutFormat::Ply)]
        format: OutFormat,
        /// List available scenes and exit.
        #[arg(long)]
        list: bool,
    },
    /// Benchmark render backends headlessly.
    Bench {
        scene: String,
        /// Backends to compare (default: all available).
        #[arg(long, value_enum, num_args = 1.., value_delimiter = ',')]
        backend: Vec<Backend>,
        /// Render size in pixels, WIDTHxHEIGHT.
        #[arg(long, default_value = "320x180", value_parser = parse_size)]
        size: (u32, u32),
        #[arg(long, default_value_t = 60)]
        frames: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value = "-y", allow_hyphen_values = true)]
        up: UpAxis,
    },
    /// Render a single image to a PNG file.
    Render {
        scene: String,
        #[arg(short, long)]
        out: PathBuf,
        #[arg(long, default_value = "640x360", value_parser = parse_size)]
        size: (u32, u32),
        #[arg(long, value_enum, default_value_t = Backend::Cpu)]
        backend: Backend,
        /// Camera yaw in degrees.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        yaw: f32,
        /// Camera pitch in degrees.
        #[arg(long, default_value_t = 15.0, allow_hyphen_values = true)]
        pitch: f32,
        /// Zoom factor (>1 moves closer).
        #[arg(long, default_value_t = 1.0)]
        zoom: f32,
        #[arg(long, default_value = "-y", allow_hyphen_values = true)]
        up: UpAxis,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum OutFormat {
    Ply,
    Splat,
}

fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s.split_once(['x', 'X']).ok_or("expected WIDTHxHEIGHT")?;
    let w: u32 = w.parse().map_err(|_| "bad width")?;
    let h: u32 = h.parse().map_err(|_| "bad height")?;
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err("size must be between 1 and 16384".into());
    }
    Ok((w, h))
}

fn load(spec: &str) -> Result<(Arc<Scene>, String)> {
    if let Some(name) = spec.strip_prefix("gen:") {
        return Ok((Arc::new(gen::generate(name)?), spec.to_string()));
    }
    let path = Path::new(spec);
    let scene = Scene::load(path)?;
    let title = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok((Arc::new(scene), title))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => view(cli.view),
        Some(Command::View(v)) => view(v),
        Some(Command::Gen {
            name,
            out,
            format,
            list,
        }) => {
            if list {
                for (n, d) in gen::SCENES {
                    println!("{n:<10} {d}");
                }
                return Ok(());
            }
            std::fs::create_dir_all(&out)?;
            let names: Vec<&str> = if name == "all" {
                gen::SCENES.iter().map(|s| s.0).collect()
            } else {
                vec![name.as_str()]
            };
            for n in names {
                let scene = gen::generate(n)?;
                let (ext, bytes) = match format {
                    OutFormat::Ply => {
                        let mut b = Vec::new();
                        ply::write(&scene, &mut b)?;
                        ("ply", b)
                    }
                    OutFormat::Splat => {
                        if scene.sh_degree > 0 {
                            eprintln!(
                                "note: .splat cannot store SH; {n} loses view-dependent color"
                            );
                        }
                        ("splat", splat_format::write(&scene))
                    }
                };
                let path = out.join(format!("{n}.{ext}"));
                std::fs::write(&path, bytes)
                    .with_context(|| format!("writing {}", path.display()))?;
                println!("wrote {} ({} gaussians)", path.display(), scene.len());
            }
            Ok(())
        }
        Some(Command::Bench {
            scene,
            backend,
            size,
            frames,
            warmup,
            up,
        }) => {
            let t = Instant::now();
            let (scene, title) = load(&scene)?;
            println!(
                "{title}: {} gaussians (SH degree {}), loaded in {:.0} ms; {}x{} px, {frames} frames, {} threads",
                scene.len(),
                scene.sh_degree,
                t.elapsed().as_secs_f64() * 1e3,
                size.0,
                size.1,
                rayon::current_num_threads(),
            );
            let backends = if backend.is_empty() {
                Backend::all().to_vec()
            } else {
                backend
            };
            let cfg = BenchConfig {
                width: size.0,
                height: size.1,
                frames,
                warmup,
                up,
            };
            let results = bench::run(scene, &backends, &cfg);
            bench::print_table(&results, &backends);
            Ok(())
        }
        Some(Command::Render {
            scene,
            out,
            size,
            backend,
            yaw,
            pitch,
            zoom,
            up,
        }) => {
            let (scene, _) = load(&scene)?;
            let mut cam = OrbitCamera::new(scene.bounds(), up);
            cam.yaw = yaw.to_radians();
            cam.pitch = pitch
                .to_radians()
                .clamp(-splat_tui::camera::MAX_PITCH, splat_tui::camera::MAX_PITCH);
            if zoom <= 0.0 {
                bail!("--zoom must be positive");
            }
            cam.zoom(1.0 / zoom);
            let mut r = render::create(backend, scene)?;
            let mut frame = Frame::default();
            let stats = r.render(&cam.view(), size.0, size.1, Vec3::ZERO, &mut frame);
            bench::write_png(&out, &frame)?;
            println!(
                "wrote {} ({} visible, {:.1} ms with {})",
                out.display(),
                stats.visible,
                stats.total_ms,
                r.name()
            );
            Ok(())
        }
    }
}

fn view(args: ViewArgs) -> Result<()> {
    let Some(spec) = args.scene else {
        bail!("no scene given; try `splat-tui gen:galaxy` or `splat-tui --help`");
    };
    let (scene, title) = load(&spec)?;
    app::run(
        scene,
        ViewerOptions {
            title,
            backend: args.backend,
            up: args.up,
            supersample: args.supersample,
            color: args.color.unwrap_or_else(ColorMode::detect),
        },
    )
}
