//! Terminal helper tests.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;
use splat_tui::render::Frame;
use splat_tui::term::{downsample, rgb_to_ansi256, ColorMode, HalfBlockImage};

fn gradient(w: u32, h: u32) -> Frame {
    let mut f = Frame::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.pixels[(y * w + x) as usize] = [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8];
        }
    }
    f
}

#[test]
fn downsample_non_divisible_truncates() {
    let src = gradient(7, 5);
    let mut dst = Frame::default();
    downsample(&src, 2, &mut dst);
    assert_eq!((dst.width, dst.height), (3, 2));
    assert_eq!(dst.pixels.len(), 6);
    // Top-left block averages (0,0),(1,0),(0,1),(1,1): r=(0+1+0+1)/4 = 0, g=(0+0+1+1)/4 = 0.
    assert_eq!(dst.pixels[0], [0, 0, 1]);
    // Block (2,1) covers x 4..6, y 2..4.
    assert_eq!(dst.pixels[5], [(4 + 5 + 4 + 5) / 4, (2 + 2 + 3 + 3) / 4, (6 + 7 + 7 + 8) / 4]);
}

#[test]
fn downsample_factor_larger_than_image_gives_empty() {
    let src = gradient(3, 3);
    let mut dst = Frame::new(10, 10);
    downsample(&src, 4, &mut dst);
    assert_eq!((dst.width, dst.height, dst.pixels.len()), (0, 0, 0));
}

#[test]
fn downsample_factor_one_and_zero() {
    let src = gradient(5, 4);
    for f in [0, 1] {
        let mut dst = Frame::default();
        downsample(&src, f, &mut dst);
        assert_eq!((dst.width, dst.height), (5, 4));
        assert_eq!(dst.pixels, src.pixels);
    }
}

#[test]
fn downsample_constant_and_white_do_not_overflow() {
    let mut src = Frame::new(9, 9);
    src.pixels.fill([255, 255, 255]);
    for f in 2..=4 {
        let mut dst = Frame::default();
        downsample(&src, f, &mut dst);
        assert_eq!(dst.width, 9 / f);
        assert!(dst.pixels.iter().all(|p| *p == [255, 255, 255]));
    }
}

fn palette(i: u8) -> [i32; 3] {
    const L: [i32; 6] = [0, 95, 135, 175, 215, 255];
    if i >= 232 {
        let g = 8 + (i as i32 - 232) * 10;
        [g; 3]
    } else {
        let i = i as usize - 16;
        [L[i / 36], L[(i / 6) % 6], L[i % 6]]
    }
}

#[test]
fn ansi256_exact_cube_colors_map_exactly() {
    const L: [u8; 6] = [0, 95, 135, 175, 215, 255];
    for (ri, &r) in L.iter().enumerate() {
        for (gi, &g) in L.iter().enumerate() {
            for (bi, &b) in L.iter().enumerate() {
                let idx = rgb_to_ansi256([r, g, b]);
                let got = palette(idx);
                assert_eq!(got, [r as i32, g as i32, b as i32], "{:?} -> {idx} (cube {ri}{gi}{bi})", [r, g, b]);
            }
        }
    }
}

#[test]
fn ansi256_gray_ramp_exact() {
    for i in 232u8..=255 {
        let g = palette(i)[0] as u8;
        assert_eq!(rgb_to_ansi256([g, g, g]), i, "gray {g}");
    }
}

#[test]
fn ansi256_is_nearest_within_reason() {
    // The result must never be farther from the input than the nearest of
    // all 240 candidates by more than a small slack, and always in 16..=255.
    let mut worst = 0;
    for r in (0..=255u16).step_by(7) {
        for g in (0..=255u16).step_by(11) {
            for b in (0..=255u16).step_by(13) {
                let c = [r as u8, g as u8, b as u8];
                let idx = rgb_to_ansi256(c);
                assert!(idx >= 16);
                let d = |p: [i32; 3]| (0..3).map(|k| (p[k] - c[k] as i32).pow(2)).sum::<i32>();
                let best = (16..=255u16).map(|i| d(palette(i as u8))).min().unwrap();
                let got = d(palette(idx));
                worst = worst.max(got - best);
                assert!(got - best <= 1500, "{c:?}: idx {idx} dist {got}, best {best}");
            }
        }
    }
    eprintln!("worst excess squared distance {worst}");
}

#[test]
fn color_mode_color_mapping() {
    assert_eq!(ColorMode::Truecolor.color([1, 2, 3]), Color::Rgb(1, 2, 3));
    assert_eq!(ColorMode::Ansi256.color([255, 0, 0]), Color::Indexed(196));
}

#[test]
fn half_block_image_area_larger_than_frame() {
    let f = gradient(3, 4); // 3 cols, 2 cell rows
    let area = Rect::new(2, 1, 10, 8);
    let full = Rect::new(0, 0, 20, 12);
    let mut buf = Buffer::empty(full);
    HalfBlockImage { frame: &f, mode: ColorMode::Truecolor }.render(area, &mut buf);
    for cy in 0..8u16 {
        for cx in 0..10u16 {
            let cell = &buf[(2 + cx, 1 + cy)];
            if cx < 3 && cy < 2 {
                assert_eq!(cell.symbol(), "▀");
                let t = f.get(cx as u32, cy as u32 * 2);
                let b = f.get(cx as u32, cy as u32 * 2 + 1);
                assert_eq!(cell.fg, Color::Rgb(t[0], t[1], t[2]));
                assert_eq!(cell.bg, Color::Rgb(b[0], b[1], b[2]));
            } else {
                assert_eq!(cell.symbol(), " ", "{cx},{cy}");
            }
        }
    }
    // Nothing outside `area` was touched.
    assert_eq!(buf[(0, 0)].symbol(), " ");
    assert_eq!(buf[(1, 1)].symbol(), " ");
}

#[test]
fn half_block_image_area_smaller_than_frame_and_odd_height() {
    let f = gradient(10, 7); // 3 full cell rows, last pixel row dropped
    let area = Rect::new(0, 0, 4, 10);
    let mut buf = Buffer::empty(area);
    HalfBlockImage { frame: &f, mode: ColorMode::Ansi256 }.render(area, &mut buf);
    for cy in 0..10u16 {
        for cx in 0..4u16 {
            let drawn = buf[(cx, cy)].symbol() == "▀";
            assert_eq!(drawn, cy < 3, "{cx},{cy}");
        }
    }
    assert!(matches!(buf[(0, 0)].fg, Color::Indexed(_)));
}

#[test]
fn half_block_image_empty_frame_and_zero_area() {
    let f = Frame::default();
    let area = Rect::new(0, 0, 5, 5);
    let mut buf = Buffer::empty(area);
    HalfBlockImage { frame: &f, mode: ColorMode::Truecolor }.render(area, &mut buf);
    assert!(buf.content.iter().all(|c| c.symbol() == " "));
    let f = gradient(4, 4);
    HalfBlockImage { frame: &f, mode: ColorMode::Truecolor }.render(Rect::new(0, 0, 0, 0), &mut buf);
}
