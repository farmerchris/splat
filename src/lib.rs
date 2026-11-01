//! splat-tui: render 3D Gaussian Splatting scenes in a terminal.
//!
//! The crate is split into:
//! - [`scene`]: in-memory gaussian storage plus loaders for `.ply` and `.splat`
//! - [`camera`]: orbit camera and projection parameters
//! - [`render`]: rasterizer backends that turn a scene + camera into an RGB frame
//! - [`term`]: mapping RGB frames onto terminal cells (half-block glyphs)
//! - [`app`]: the interactive TUI
//! - [`gen`]: procedural example scenes
//! - [`bench`]: headless benchmark harness used to compare backends

pub mod app;
pub mod bench;
pub mod camera;
pub mod gen;
pub mod ply;
pub mod render;
pub mod scene;
pub mod sh;
pub mod splat_format;
pub mod term;
