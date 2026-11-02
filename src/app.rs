//! Interactive terminal viewer.

use std::io::{stdout, BufWriter, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, BeginSynchronizedUpdate, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use glam::Vec3;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Terminal;

use crate::camera::{OrbitCamera, UpAxis};
use crate::render::{self, Backend, Frame, RenderStats, Renderer};
use crate::scene::Scene;
use crate::term::{downsample, ColorMode, HalfBlockImage};

const ROTATE_STEP: f32 = 5.0 * std::f32::consts::PI / 180.0;
const MOUSE_YAW_PER_CELL: f32 = 0.03;
const MOUSE_PITCH_PER_CELL: f32 = 0.06;
const PAN_STEP: f32 = 0.05;
const ZOOM_STEP: f32 = 1.15;
const BACKGROUNDS: [(&str, [f32; 3]); 4] = [
    ("black", [0.0, 0.0, 0.0]),
    ("gray", [0.12, 0.12, 0.14]),
    ("white", [1.0, 1.0, 1.0]),
    ("sky", [0.45, 0.6, 0.8]),
];

pub struct ViewerOptions {
    pub title: String,
    pub backend: Backend,
    pub up: UpAxis,
    pub supersample: u32,
    pub color: ColorMode,
}

/// Large output buffer: a full truecolor frame is hundreds of KB of escape
/// codes, and stdout's default line buffering would split it into many writes.
type Term = Terminal<CrosstermBackend<BufWriter<Stdout>>>;

fn init_terminal() -> Result<Term> {
    // Restore the terminal before the default panic message is printed.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prev(info);
    }));
    enable_raw_mode()?;
    let setup = execute!(stdout(), EnterAlternateScreen, EnableMouseCapture);
    let term = setup.map_err(anyhow::Error::from).and_then(|_| {
        Ok(Terminal::new(CrosstermBackend::new(
            BufWriter::with_capacity(1 << 20, stdout()),
        ))?)
    });
    if term.is_err() {
        restore_terminal();
    }
    term
}

fn restore_terminal() {
    let _ = execute!(
        stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    let _ = disable_raw_mode();
}

struct Drag {
    button: MouseButton,
    col: u16,
    row: u16,
}

struct App {
    scene: Arc<Scene>,
    title: String,
    renderer: Box<dyn Renderer>,
    backend: Backend,
    camera: OrbitCamera,
    hi_res: Frame,
    display: Frame,
    supersample: u32,
    color: ColorMode,
    background: usize,
    auto_rotate: bool,
    show_help: bool,
    drag: Option<Drag>,
    dirty: bool,
    quit: bool,
    view_rows: u16,
    stats: RenderStats,
    present_ms: f64,
    fps: f64,
    last_frame: Option<Instant>,
    message: Option<(String, Instant)>,
}

pub fn run(scene: Arc<Scene>, opts: ViewerOptions) -> Result<()> {
    let renderer = render::create(opts.backend, scene.clone())?;
    let camera = OrbitCamera::new(scene.bounds(), opts.up);
    let mut app = App {
        scene,
        title: opts.title,
        renderer,
        backend: opts.backend,
        camera,
        hi_res: Frame::default(),
        display: Frame::default(),
        supersample: opts.supersample.clamp(1, 4),
        color: opts.color,
        background: 0,
        auto_rotate: false,
        show_help: false,
        drag: None,
        dirty: true,
        quit: false,
        view_rows: 1,
        stats: RenderStats::default(),
        present_ms: 0.0,
        fps: 0.0,
        last_frame: None,
        message: None,
    };

    let mut terminal = init_terminal()?;
    let result = app.event_loop(&mut terminal);
    restore_terminal();
    result
}

impl App {
    fn event_loop(&mut self, terminal: &mut Term) -> Result<()> {
        let mut last_tick = Instant::now();
        while !self.quit {
            if self.auto_rotate {
                let dt = last_tick.elapsed().as_secs_f32().min(0.1);
                self.camera.rotate(dt * 0.6, 0.0);
                self.dirty = true;
            }
            last_tick = Instant::now();
            if self
                .message
                .as_ref()
                .is_some_and(|(_, t)| t.elapsed() > Duration::from_secs(3))
            {
                self.message = None;
                self.dirty = true;
            }
            if self.dirty {
                self.dirty = false;
                self.draw(terminal)?;
            } else {
                self.last_frame = None;
            }
            // Auto-rotate is capped at ~60 fps; otherwise sleep until input arrives.
            let timeout = if self.auto_rotate {
                Duration::from_millis(16).saturating_sub(last_tick.elapsed())
            } else {
                Duration::from_millis(250)
            };
            if event::poll(timeout)? {
                // Drain everything queued so drags don't lag behind rendering.
                loop {
                    let ev = event::read()?;
                    self.handle(ev);
                    if self.quit || !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, terminal: &mut Term) -> Result<()> {
        let size = terminal.size()?;
        let [view_area, _] = split(Rect::new(0, 0, size.width, size.height));
        self.view_rows = view_area.height.max(1);
        // Stay within the renderer's frame size limit on huge terminals.
        let largest = (view_area.width as u32).max(view_area.height as u32 * 2).max(1);
        let ss = self.supersample.min(render::MAX_DIM / largest).max(1);
        let (w, h) = (
            view_area.width as u32 * ss,
            view_area.height as u32 * 2 * ss,
        );
        let bg = Vec3::from_array(BACKGROUNDS[self.background].1);
        self.stats = self
            .renderer
            .render(&self.camera.view(), w, h, bg, &mut self.hi_res);
        downsample(&self.hi_res, ss, &mut self.display);

        let now = Instant::now();
        if let Some(prev) = self.last_frame {
            let inst = 1.0 / (now - prev).as_secs_f64().max(1e-6);
            self.fps = if self.fps == 0.0 {
                inst
            } else {
                self.fps * 0.8 + inst * 0.2
            };
        } else {
            self.fps = 0.0;
        }
        self.last_frame = Some(now);

        let t = Instant::now();
        let status = self.status_line();
        let help = self.show_help;
        let image = HalfBlockImage {
            frame: &self.display,
            mode: self.color,
        };
        // Synchronized output (DEC 2026) avoids tearing on supporting terminals
        // and is ignored elsewhere.
        execute!(terminal.backend_mut(), BeginSynchronizedUpdate)?;
        terminal.draw(|f| {
            // Re-derive the layout: the terminal may have been resized while
            // rendering, and ratatui resizes its buffer inside `draw`.
            let [view_area, status_area] = split(f.area());
            f.render_widget(image, view_area);
            f.render_widget(Paragraph::new(status), status_area);
            if help {
                draw_help(f, view_area);
            }
        })?;
        execute!(terminal.backend_mut(), EndSynchronizedUpdate)?;
        self.present_ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    fn status_line(&self) -> Line<'static> {
        let s = &self.stats;
        let key = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::Gray).bg(Color::DarkGray);
        // Most important information first: narrow terminals truncate the tail.
        let mut text = format!(
            " {} │ {}/{} visible │ {:.1}ms",
            self.title,
            human(s.visible),
            human(self.scene.len()),
            s.total_ms + self.present_ms,
        );
        if self.fps > 0.0 {
            text.push_str(&format!(" {:.0}fps", self.fps));
        }
        if let Some((m, _)) = &self.message {
            text.push_str(&format!(" │ {m}"));
        }
        text.push_str(&format!(
            " │ yaw {:.0}° pitch {:.0}° │ {} ss{} up{} │ proj {:.1} sort {:.1} rast {:.1} draw {:.1} ",
            self.camera.yaw.to_degrees(),
            self.camera.pitch.to_degrees(),
            self.renderer.name(),
            self.supersample,
            self.camera.up.label(),
            s.project_ms,
            s.sort_ms,
            s.raster_ms,
            self.present_ms,
        ));
        Line::from(vec![Span::styled(" ? help ", key), Span::styled(text, dim)]).style(dim)
    }

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => self.on_key(k),
            Event::Mouse(m) => self.on_mouse(m),
            Event::Resize(..) => self.dirty = true,
            _ => {}
        }
    }

    fn on_key(&mut self, k: KeyEvent) {
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let cam = &mut self.camera;
        // An unmapped key must not cancel a redraw requested earlier in the
        // same batch of events.
        let was_dirty = self.dirty;
        self.dirty = true;
        let code = match k.code {
            KeyCode::Char(c) if !ctrl => KeyCode::Char(c.to_ascii_lowercase()),
            other => other,
        };
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Left if shift => cam.pan(PAN_STEP, 0.0),
            KeyCode::Right if shift => cam.pan(-PAN_STEP, 0.0),
            KeyCode::Up if shift => cam.pan(0.0, PAN_STEP),
            KeyCode::Down if shift => cam.pan(0.0, -PAN_STEP),
            KeyCode::Left => cam.rotate(-ROTATE_STEP, 0.0),
            KeyCode::Right => cam.rotate(ROTATE_STEP, 0.0),
            KeyCode::Up => cam.rotate(0.0, -ROTATE_STEP),
            KeyCode::Down => cam.rotate(0.0, ROTATE_STEP),
            KeyCode::Char('a') => cam.pan(PAN_STEP, 0.0),
            KeyCode::Char('d') => cam.pan(-PAN_STEP, 0.0),
            KeyCode::Char('w') => cam.pan(0.0, PAN_STEP),
            KeyCode::Char('s') => cam.pan(0.0, -PAN_STEP),
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::PageUp => cam.zoom(1.0 / ZOOM_STEP),
            KeyCode::Char('-') | KeyCode::Char('_') | KeyCode::PageDown => cam.zoom(ZOOM_STEP),
            KeyCode::Char('[') => cam.fov_y = (cam.fov_y * 1.1).min(150f32.to_radians()),
            KeyCode::Char(']') => cam.fov_y = (cam.fov_y / 1.1).max(5f32.to_radians()),
            KeyCode::Char('r') | KeyCode::Home => cam.reset(),
            KeyCode::Char('u') => {
                cam.up = cam.up.next();
                self.flash(format!("up axis {}", self.camera.up.label()));
            }
            KeyCode::Char(' ') => self.auto_rotate = !self.auto_rotate,
            KeyCode::Char('b') => self.background = (self.background + 1) % BACKGROUNDS.len(),
            KeyCode::Char('x') => {
                self.supersample = self.supersample % 4 + 1;
                self.flash(format!("supersample {}x", self.supersample));
            }
            KeyCode::Char('m') => {
                self.color = match self.color {
                    ColorMode::Truecolor => ColorMode::Ansi256,
                    ColorMode::Ansi256 => ColorMode::Truecolor,
                };
                self.flash(format!("color mode {:?}", self.color));
            }
            KeyCode::Char('n') => self.next_backend(),
            KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::F(1) => {
                self.show_help = !self.show_help
            }
            _ => self.dirty = was_dirty,
        }
    }

    fn on_mouse(&mut self, m: MouseEvent) {
        match m.kind {
            MouseEventKind::Down(button) => {
                self.drag = Some(Drag {
                    button,
                    col: m.column,
                    row: m.row,
                });
            }
            MouseEventKind::Drag(button) => {
                let Some(d) = self.drag.as_mut().filter(|d| d.button == button) else {
                    self.drag = Some(Drag {
                        button,
                        col: m.column,
                        row: m.row,
                    });
                    return;
                };
                let dx = m.column as f32 - d.col as f32;
                let dy = m.row as f32 - d.row as f32;
                d.col = m.column;
                d.row = m.row;
                let rows = self.view_rows as f32;
                if button == MouseButton::Left && !m.modifiers.contains(KeyModifiers::SHIFT) {
                    self.camera
                        .rotate(dx * MOUSE_YAW_PER_CELL, dy * MOUSE_PITCH_PER_CELL);
                } else {
                    // One cell is 1 px wide and 2 px tall; view height is 2*rows px.
                    self.camera.pan(-dx / (2.0 * rows), -dy / rows);
                }
                self.dirty = true;
            }
            MouseEventKind::Up(_) => self.drag = None,
            MouseEventKind::ScrollUp => {
                self.camera.zoom(1.0 / 1.1);
                self.dirty = true;
            }
            MouseEventKind::ScrollDown => {
                self.camera.zoom(1.1);
                self.dirty = true;
            }
            _ => {}
        }
    }

    fn next_backend(&mut self) {
        let all = Backend::all();
        let start = all.iter().position(|b| *b == self.backend).unwrap_or(0);
        for off in 1..=all.len() {
            let b = all[(start + off) % all.len()];
            if b == self.backend {
                break;
            }
            match render::create(b, self.scene.clone()) {
                Ok(r) => {
                    self.renderer = r;
                    self.backend = b;
                    self.flash(format!("backend {}", self.renderer.name()));
                    return;
                }
                Err(_) => continue,
            }
        }
        self.flash("no other backend available".into());
    }

    fn flash(&mut self, msg: String) {
        self.message = Some((msg, Instant::now()));
    }
}

/// Viewport above a one-line status bar.
fn split(area: Rect) -> [Rect; 2] {
    Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(area)
}

fn draw_help(f: &mut ratatui::Frame, area: Rect) {
    let lines = [
        ("mouse drag", "rotate (orbit)"),
        ("right/shift drag", "pan"),
        ("scroll", "zoom"),
        ("← → ↑ ↓", "rotate"),
        ("shift+arrows, wasd", "pan"),
        ("+ / - , PgUp/PgDn", "zoom"),
        ("[ / ]", "wider / narrower field of view"),
        ("r", "reset camera"),
        ("u", "cycle up axis (+Y -Y +Z -Z)"),
        ("space", "toggle auto-rotate"),
        ("b", "cycle background"),
        ("x", "cycle supersampling 1x-4x"),
        ("m", "toggle truecolor / 256 colors"),
        ("n", "next render backend"),
        ("? / h", "toggle this help"),
        ("q / esc", "quit"),
    ];
    let text: Vec<Line> = lines
        .iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(format!("{k:>20}  "), Style::default().fg(Color::Cyan)),
                Span::raw(*v),
            ])
        })
        .collect();
    let w = 58.min(area.width);
    let h = (text.len() as u16 + 2).min(area.height);
    let r = Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    );
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" splat-tui controls "),
        ),
        r,
    );
}

/// Format a count like `1.2M`.
pub fn human(n: usize) -> String {
    match n {
        0..=9_999 => n.to_string(),
        10_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.2}M", n as f64 / 1e6),
    }
}

#[cfg(test)]
mod tests {
    use super::human;

    #[test]
    fn human_counts() {
        assert_eq!(human(12), "12");
        assert_eq!(human(12_345), "12.3k");
        assert_eq!(human(1_234_567), "1.23M");
    }
}
