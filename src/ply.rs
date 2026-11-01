//! Minimal PLY reader/writer for 3D Gaussian Splatting point clouds.
//!
//! Supports `ascii`, `binary_little_endian` and `binary_big_endian` files with
//! the property layout written by the reference 3DGS trainer
//! (`x y z [nx ny nz] f_dc_0..2 [f_rest_*] opacity scale_0..2 rot_0..3`).
//! Plain colored point clouds (`red green blue`) are also accepted and get a
//! small default splat size.

use std::collections::HashMap;
use std::io::Write;

use anyhow::{anyhow, bail, Context, Result};
use glam::{Quat, Vec3};

use crate::scene::{logit, sigmoid, Scene};
use crate::sh::{degree_from_rest_count, SH_C0};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Ascii,
    BinaryLe,
    BinaryBe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => bail!("unsupported PLY property type '{s}'"),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    #[inline]
    fn read(self, b: &[u8], le: bool) -> f64 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b[..$n].try_into().unwrap();
                (if le {
                    <$t>::from_le_bytes(a)
                } else {
                    <$t>::from_be_bytes(a)
                }) as f64
            }};
        }
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => rd!(i16, 2),
            Self::U16 => rd!(u16, 2),
            Self::I32 => rd!(i32, 4),
            Self::U32 => rd!(u32, 4),
            Self::F32 => rd!(f32, 4),
            Self::F64 => rd!(f64, 8),
        }
    }
}

#[derive(Debug)]
struct Property {
    name: String,
    ty: ScalarType,
    /// `Some(count_type)` for list properties.
    list: Option<ScalarType>,
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Property>,
}

struct Header {
    format: Format,
    elements: Vec<Element>,
    body_offset: usize,
}

fn parse_header(data: &[u8]) -> Result<Header> {
    let end_marker = b"end_header";
    let pos = data
        .windows(end_marker.len())
        .take(1 << 20)
        .position(|w| w == end_marker)
        .ok_or_else(|| anyhow!("PLY header missing end_header"))?;
    // Body starts after the newline that terminates `end_header`.
    let mut body_offset = pos + end_marker.len();
    while body_offset < data.len() && data[body_offset] != b'\n' {
        body_offset += 1;
    }
    body_offset += 1;

    let text = std::str::from_utf8(&data[..pos]).context("PLY header is not UTF-8")?;
    let mut lines = text.lines().map(str::trim);
    if lines.next() != Some("ply") {
        bail!("not a PLY file (missing 'ply' magic)");
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in lines {
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("format") => {
                format = Some(match tok.next() {
                    Some("ascii") => Format::Ascii,
                    Some("binary_little_endian") => Format::BinaryLe,
                    Some("binary_big_endian") => Format::BinaryBe,
                    other => bail!("unsupported PLY format {other:?}"),
                });
            }
            Some("element") => {
                let name = tok.next().ok_or_else(|| anyhow!("element without name"))?;
                let count = tok
                    .next()
                    .ok_or_else(|| anyhow!("element without count"))?
                    .parse()
                    .context("bad element count")?;
                elements.push(Element {
                    name: name.to_string(),
                    count,
                    props: Vec::new(),
                });
            }
            Some("property") => {
                let el = elements
                    .last_mut()
                    .ok_or_else(|| anyhow!("property before element"))?;
                let t = tok.next().ok_or_else(|| anyhow!("property without type"))?;
                let prop = if t == "list" {
                    let ct = ScalarType::parse(tok.next().unwrap_or(""))?;
                    let it = ScalarType::parse(tok.next().unwrap_or(""))?;
                    let name = tok.next().ok_or_else(|| anyhow!("list without name"))?;
                    Property {
                        name: name.to_string(),
                        ty: it,
                        list: Some(ct),
                    }
                } else {
                    let name = tok.next().ok_or_else(|| anyhow!("property without name"))?;
                    Property {
                        name: name.to_string(),
                        ty: ScalarType::parse(t)?,
                        list: None,
                    }
                };
                el.props.push(prop);
            }
            Some("comment") | Some("obj_info") | None => {}
            Some(other) => bail!("unexpected PLY header keyword '{other}'"),
        }
    }
    Ok(Header {
        format: format.ok_or_else(|| anyhow!("PLY header has no format line"))?,
        elements,
        body_offset,
    })
}

/// Column indices of the properties we understand.
struct Columns {
    pos: [usize; 3],
    dc: Option<[usize; 3]>,
    rgb: Option<[usize; 3]>,
    rest: Vec<usize>,
    opacity: Option<usize>,
    scale: Option<[usize; 3]>,
    rot: Option<[usize; 4]>,
}

impl Columns {
    fn resolve(props: &[Property]) -> Result<Self> {
        let idx: HashMap<&str, usize> = props
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.as_str(), i))
            .collect();
        let get = |n: &str| idx.get(n).copied();
        let need = |n: &str| get(n).ok_or_else(|| anyhow!("PLY vertex missing '{n}'"));
        let all3 = |a: &str, b: &str, c: &str| Some([get(a)?, get(b)?, get(c)?]);

        let mut rest = Vec::new();
        while let Some(i) = get(&format!("f_rest_{}", rest.len())) {
            rest.push(i);
        }
        if props.iter().any(|p| p.name.starts_with("packed_")) {
            bail!(
                "compressed PLY (SuperSplat packed format) is not supported; decompress it first"
            );
        }
        let rot = (|| Some([get("rot_0")?, get("rot_1")?, get("rot_2")?, get("rot_3")?]))();
        Ok(Self {
            pos: [need("x")?, need("y")?, need("z")?],
            dc: all3("f_dc_0", "f_dc_1", "f_dc_2"),
            rgb: all3("red", "green", "blue"),
            rest,
            opacity: get("opacity"),
            scale: all3("scale_0", "scale_1", "scale_2"),
            rot,
        })
    }
}

/// Parse a PLY file into an (un-finalized) [`Scene`].
pub fn parse(data: &[u8]) -> Result<Scene> {
    let header = parse_header(data)?;
    let vi = header
        .elements
        .iter()
        .position(|e| e.name == "vertex")
        .ok_or_else(|| anyhow!("PLY has no vertex element"))?;
    // Elements stored before `vertex` must be skipped; only fixed-size ones are supported.
    let mut offset = header.body_offset;
    for el in &header.elements[..vi] {
        if header.format == Format::Ascii || el.props.iter().any(|p| p.list.is_some()) {
            bail!("PLY element '{}' before vertex is not supported", el.name);
        }
        let stride: usize = el.props.iter().map(|p| p.ty.size()).sum();
        offset = offset
            .checked_add(
                stride
                    .checked_mul(el.count)
                    .ok_or_else(|| anyhow!("PLY too large"))?,
            )
            .ok_or_else(|| anyhow!("PLY too large"))?;
    }
    let el = &header.elements[vi];
    let cols = Columns::resolve(&el.props)?;
    let n = el.count;
    let nprops = el.props.len();

    // Decode every vertex into a row of f64 values, then map columns.
    let mut row = vec![0f64; nprops];
    let mut scene = Scene::default();
    let degree = degree_from_rest_count(cols.rest.len()).unwrap_or(0);
    let rest_used = 3 * (crate::sh::coeffs_for_degree(degree) - 1);
    let per_channel = rest_used / 3;
    scene.sh_degree = degree;
    scene.positions.reserve(n);
    scene.scales.reserve(n);
    scene.rotations.reserve(n);
    scene.opacities.reserve(n);
    scene.colors.reserve(n);
    scene.sh_rest.reserve(n * per_channel);

    let emit = |row: &[f64], scene: &mut Scene| {
        let f = |i: usize| row[i] as f32;
        scene
            .positions
            .push(Vec3::new(f(cols.pos[0]), f(cols.pos[1]), f(cols.pos[2])));
        let color = if let Some(dc) = cols.dc {
            Vec3::new(f(dc[0]), f(dc[1]), f(dc[2])) * SH_C0 + Vec3::splat(0.5)
        } else if let Some(rgb) = cols.rgb {
            let scale = if el.props[rgb[0]].ty == ScalarType::U8 {
                1.0 / 255.0
            } else {
                1.0
            };
            Vec3::new(f(rgb[0]), f(rgb[1]), f(rgb[2])) * scale
        } else {
            Vec3::splat(0.8)
        };
        scene.colors.push(color);
        scene
            .opacities
            .push(cols.opacity.map(|i| sigmoid(f(i))).unwrap_or(1.0));
        scene.scales.push(
            cols.scale
                .map(|s| Vec3::new(f(s[0]).exp(), f(s[1]).exp(), f(s[2]).exp()))
                .unwrap_or(Vec3::splat(0.01)),
        );
        scene.rotations.push(
            cols.rot
                // 3DGS stores quaternions as (w, x, y, z).
                .map(|r| Quat::from_xyzw(f(r[1]), f(r[2]), f(r[3]), f(r[0])))
                .unwrap_or(Quat::IDENTITY),
        );
        // f_rest is channel-major: all R coefficients, then G, then B.
        for k in 0..per_channel {
            scene.sh_rest.push(Vec3::new(
                f(cols.rest[k]),
                f(cols.rest[per_channel + k]),
                f(cols.rest[2 * per_channel + k]),
            ));
        }
    };

    match header.format {
        Format::Ascii => {
            let text = std::str::from_utf8(&data[offset.min(data.len())..])
                .context("ASCII PLY body is not UTF-8")?;
            let mut tokens = text.split_ascii_whitespace();
            for v in 0..n {
                for (pi, p) in el.props.iter().enumerate() {
                    let mut next = || -> Result<f64> {
                        tokens
                            .next()
                            .ok_or_else(|| anyhow!("ASCII PLY truncated at vertex {v}"))?
                            .parse::<f64>()
                            .context("bad number in ASCII PLY")
                    };
                    if p.list.is_some() {
                        let cnt = next()? as usize;
                        for _ in 0..cnt {
                            next()?;
                        }
                        row[pi] = 0.0;
                    } else {
                        row[pi] = next()?;
                    }
                }
                emit(&row, &mut scene);
            }
        }
        Format::BinaryLe | Format::BinaryBe => {
            if el.props.iter().any(|p| p.list.is_some()) {
                bail!("list properties on vertices are not supported");
            }
            let le = header.format == Format::BinaryLe;
            let offs: Vec<usize> = el
                .props
                .iter()
                .scan(0, |acc, p| {
                    let o = *acc;
                    *acc += p.ty.size();
                    Some(o)
                })
                .collect();
            let stride: usize = el.props.iter().map(|p| p.ty.size()).sum();
            let need = stride
                .checked_mul(n)
                .and_then(|b| b.checked_add(offset))
                .ok_or_else(|| anyhow!("PLY vertex count overflows"))?;
            if data.len() < need {
                bail!(
                    "PLY body truncated: need {need} bytes, file has {}",
                    data.len()
                );
            }
            for rec in data[offset..need].chunks_exact(stride.max(1)).take(n) {
                for (pi, p) in el.props.iter().enumerate() {
                    row[pi] = p.ty.read(&rec[offs[pi]..], le);
                }
                emit(&row, &mut scene);
            }
        }
    }
    Ok(scene)
}

/// Write a scene in the standard 3DGS binary little-endian layout.
pub fn write<W: Write>(scene: &Scene, mut w: W) -> Result<()> {
    let n = scene.len();
    let per_channel = scene.rest_per_splat();
    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str("comment generated by splat-tui\n");
    header.push_str(&format!("element vertex {n}\n"));
    for p in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ] {
        header.push_str(&format!("property float {p}\n"));
    }
    for i in 0..per_channel * 3 {
        header.push_str(&format!("property float f_rest_{i}\n"));
    }
    for p in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {p}\n"));
    }
    header.push_str("end_header\n");
    w.write_all(header.as_bytes())?;

    let mut buf: Vec<u8> = Vec::with_capacity(n * (17 + per_channel * 3) * 4);
    for i in 0..n {
        let mut put = |v: f32| buf.extend_from_slice(&v.to_le_bytes());
        let p = scene.positions[i];
        let dc = (scene.colors[i] - Vec3::splat(0.5)) / SH_C0;
        let s = scene.scales[i];
        let q = scene.rotations[i];
        for v in [p.x, p.y, p.z, 0.0, 0.0, 0.0, dc.x, dc.y, dc.z] {
            put(v);
        }
        let rest = &scene.sh_rest[i * per_channel..(i + 1) * per_channel];
        for c in 0..3 {
            for r in rest {
                put(r[c]);
            }
        }
        put(logit(scene.opacities[i]));
        for v in [s.x, s.y, s.z] {
            put(v.max(1e-12).ln());
        }
        for v in [q.w, q.x, q.y, q.z] {
            put(v);
        }
    }
    w.write_all(&buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scene() -> Scene {
        let mut s = Scene::default();
        s.push(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::from_rotation_y(0.5),
            0.75,
            Vec3::new(0.2, 0.4, 0.6),
        );
        s.push(
            Vec3::ZERO,
            Vec3::splat(0.05),
            Quat::IDENTITY,
            0.25,
            Vec3::ONE,
        );
        s
    }

    #[test]
    fn roundtrip_binary() {
        let s = sample_scene();
        let mut bytes = Vec::new();
        write(&s, &mut bytes).unwrap();
        let r = parse(&bytes).unwrap();
        assert_eq!(r.len(), 2);
        assert!((r.positions[0] - s.positions[0]).length() < 1e-6);
        assert!((r.scales[0] - s.scales[0]).length() < 1e-5);
        assert!((r.opacities[0] - 0.75).abs() < 1e-5);
        assert!((r.colors[0] - s.colors[0]).length() < 1e-5);
        assert!(r.rotations[0].angle_between(s.rotations[0]) < 1e-3);
    }

    #[test]
    fn roundtrip_with_sh() {
        let mut s = sample_scene();
        s.sh_degree = 1;
        s.sh_rest = (0..6)
            .map(|i| Vec3::new(i as f32, 10.0 + i as f32, 20.0 + i as f32))
            .collect();
        let mut bytes = Vec::new();
        write(&s, &mut bytes).unwrap();
        let r = parse(&bytes).unwrap();
        assert_eq!(r.sh_degree, 1);
        assert_eq!(r.sh_rest, s.sh_rest);
    }

    #[test]
    fn ascii_point_cloud() {
        let txt = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 255 0 0\n1 1 1 0 255 0\n";
        let s = parse(txt).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.colors[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(s.opacities[1], 1.0);
    }

    #[test]
    fn truncated_binary_is_error() {
        let mut bytes = Vec::new();
        write(&sample_scene(), &mut bytes).unwrap();
        bytes.truncate(bytes.len() - 10);
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn garbage_is_error() {
        assert!(parse(b"hello world").is_err());
        assert!(parse(b"ply\nformat binary_little_endian 1.0\nend_header\n").is_err());
    }
}
