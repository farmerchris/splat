//! Mapping RGB frames onto terminal cells.
//!
//! Each terminal cell shows two vertically stacked pixels using the upper
//! half block `▀` (foreground = top pixel, background = bottom pixel). Since
//! cells are roughly twice as tall as wide this yields square pixels.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

use crate::render::Frame;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ColorMode {
    /// 24-bit RGB escape codes.
    Truecolor,
    /// xterm 256-color palette (for terminals without truecolor).
    Ansi256,
}

impl ColorMode {
    /// Guess from `$COLORTERM` / `$TERM_PROGRAM`.
    pub fn detect() -> Self {
        let ct = std::env::var("COLORTERM")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let tp = std::env::var("TERM_PROGRAM").unwrap_or_default();
        if ct.contains("truecolor") || ct.contains("24bit") {
            ColorMode::Truecolor
        } else if tp == "Apple_Terminal" {
            ColorMode::Ansi256
        } else {
            ColorMode::Truecolor
        }
    }

    #[inline]
    pub fn color(self, c: [u8; 3]) -> Color {
        match self {
            ColorMode::Truecolor => Color::Rgb(c[0], c[1], c[2]),
            ColorMode::Ansi256 => Color::Indexed(rgb_to_ansi256(c)),
        }
    }
}

/// Nearest xterm-256 color, choosing between the 6×6×6 cube and the gray ramp.
pub fn rgb_to_ansi256(c: [u8; 3]) -> u8 {
    const LEVELS: [i32; 6] = [0, 95, 135, 175, 215, 255];
    let cube_idx = |v: u8| -> usize {
        let v = v as i32;
        if v < 48 {
            0
        } else if v < 115 {
            1
        } else {
            ((v - 35) / 40).min(5) as usize
        }
    };
    let (ri, gi, bi) = (cube_idx(c[0]), cube_idx(c[1]), cube_idx(c[2]));
    let cube = [LEVELS[ri], LEVELS[gi], LEVELS[bi]];
    let avg = (c[0] as i32 + c[1] as i32 + c[2] as i32) / 3;
    let gray_idx = ((avg - 3) / 10).clamp(0, 23);
    let gray = 8 + gray_idx * 10;
    let dist = |a: [i32; 3]| (0..3).map(|k| (a[k] - c[k] as i32).pow(2)).sum::<i32>();
    if dist([gray; 3]) < dist(cube) {
        (232 + gray_idx) as u8
    } else {
        (16 + 36 * ri + 6 * gi + bi) as u8
    }
}

/// Box-filter `src` down by an integer factor into `dst`.
pub fn downsample(src: &Frame, factor: u32, dst: &mut Frame) {
    let factor = factor.max(1);
    let w = src.width / factor;
    let h = src.height / factor;
    dst.resize(w, h);
    if factor == 1 {
        dst.pixels.copy_from_slice(&src.pixels[..(w * h) as usize]);
        return;
    }
    let n = factor * factor;
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0u32; 3];
            for sy in 0..factor {
                let row = ((y * factor + sy) * src.width + x * factor) as usize;
                for p in &src.pixels[row..row + factor as usize] {
                    acc[0] += p[0] as u32;
                    acc[1] += p[1] as u32;
                    acc[2] += p[2] as u32;
                }
            }
            dst.pixels[(y * w + x) as usize] =
                [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8];
        }
    }
}

/// Widget drawing a frame with half blocks. The frame must be
/// `area.width` × `2 * area.height` pixels (extra pixels are ignored).
pub struct HalfBlockImage<'a> {
    pub frame: &'a Frame,
    pub mode: ColorMode,
}

impl Widget for HalfBlockImage<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let f = self.frame;
        let cols = (area.width as u32).min(f.width);
        let rows = (area.height as u32).min(f.height / 2);
        for cy in 0..rows {
            for cx in 0..cols {
                let top = f.get(cx, cy * 2);
                let bot = f.get(cx, cy * 2 + 1);
                let cell = &mut buf[(area.x + cx as u16, area.y + cy as u16)];
                cell.set_char('▀')
                    .set_fg(self.mode.color(top))
                    .set_bg(self.mode.color(bot));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi256_extremes() {
        assert_eq!(rgb_to_ansi256([0, 0, 0]), 16);
        assert_eq!(rgb_to_ansi256([255, 255, 255]), 231);
        assert_eq!(rgb_to_ansi256([255, 0, 0]), 196);
        // Mid gray maps onto the gray ramp.
        let g = rgb_to_ansi256([128, 128, 128]);
        assert!((232..=255).contains(&g), "{g}");
    }

    #[test]
    fn downsample_averages() {
        let mut src = Frame::new(2, 2);
        src.pixels = vec![[0, 0, 0], [100, 100, 100], [200, 200, 200], [100, 100, 100]];
        let mut dst = Frame::default();
        downsample(&src, 2, &mut dst);
        assert_eq!((dst.width, dst.height), (1, 1));
        assert_eq!(dst.pixels[0], [100, 100, 100]);
    }

    #[test]
    fn widget_draws_half_blocks() {
        let mut f = Frame::new(2, 2);
        f.pixels = vec![[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]];
        let area = Rect::new(0, 0, 2, 1);
        let mut buf = Buffer::empty(area);
        HalfBlockImage {
            frame: &f,
            mode: ColorMode::Truecolor,
        }
        .render(area, &mut buf);
        let c = &buf[(0, 0)];
        assert_eq!(c.symbol(), "▀");
        assert_eq!(c.fg, Color::Rgb(255, 0, 0));
        assert_eq!(c.bg, Color::Rgb(0, 0, 255));
    }
}
