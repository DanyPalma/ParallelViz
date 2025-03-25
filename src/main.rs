use clap::Parser;
use eframe::egui;
use hound::WavReader;
use rayon::prelude::*;
use rodio::{Decoder, OutputStream, Sink};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::thread;

#[derive(Parser, Debug)]
#[clap(about = "Audio visualizer that displays frequency spectrum")]
struct Args {
    #[clap(value_parser)]
    audio_file: String,
}

#[allow(dead_code)]
fn rust_fft(input: &mut [Complex<f32>]) {
    let mut planner = FftPlanner::<f32>::new();
    let f = planner.plan_fft_forward(input.len());
    f.process(input)
}

#[allow(dead_code)]
fn our_fft(input: &mut [Complex<f32>]) {
    let len = input.len();
    if len == 1 {
        return;
    }
    assert!(len.is_power_of_two());
    let mut evens = input.iter().step_by(2).copied().collect::<Vec<_>>();
    let mut odds = input.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
    our_fft(&mut evens);
    our_fft(&mut odds);
    let principle_angle = core::f32::consts::TAU / len as f32;
    let principle_root = Complex::<f32> {
        re: principle_angle.cos(),
        im: principle_angle.sin(),
    };
    let mut cur_root = Complex::<f32> { re: 1.0, im: 0.0 };
    for i in 0..len / 2 {
        input[i] = evens[i] + cur_root * odds[i];
        input[i + len / 2] = evens[i] - cur_root * odds[i];
        cur_root *= principle_root;
    }
}

fn fft(input: &mut [Complex<f32>]) {
    our_fft(input);
}

#[allow(dead_code)]
struct AudioVisualizer {
    audio_data: Vec<f32>,
    block_size: usize,
    current_position: usize,
    processed_chunks: Vec<Vec<f32>>,
}

impl AudioVisualizer {
    fn new(audio_file: &str, block_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = WavReader::open(audio_file)?;

        let audio_data: Vec<f32> = reader
            .samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / 32768.0)
            .collect();

        let processed_chunks: Vec<Vec<f32>> = audio_data
            .par_chunks(block_size)
            .map(|chunk| {
                let mut input_buffer = chunk.iter().map(|&x| Complex { re: x, im: 0.0 }).collect::<Vec<_>>();
                if input_buffer.len().is_power_of_two() {
                    fft(&mut input_buffer);
                }
                input_buffer.iter().map(|c| (c.norm() + 1.0).ln()).collect::<Vec<f32>>()
            })
            .collect();

        Ok(AudioVisualizer {
            audio_data,
            block_size,
            current_position: 0,
            processed_chunks,
        })
    }

    fn play_audio(&self, audio_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        let audio_file = audio_file.to_string();
        thread::spawn(move || {
            let (_stream, stream_handle) = OutputStream::try_default().unwrap();
            let sink = Sink::try_new(&stream_handle).unwrap();
            let file = File::open(audio_file).unwrap();
            let source = Decoder::new(BufReader::new(file)).unwrap();
            sink.append(source);
            sink.play();
            sink.sleep_until_end();
        });
        Ok(())
    }
}

struct VisualizerApp {
    visualizer: Rc<RefCell<AudioVisualizer>>,
    spectrum: Vec<f32>,
}

impl VisualizerApp {
    fn new(visualizer: Rc<RefCell<AudioVisualizer>>) -> Self {
        Self {
            visualizer: visualizer.clone(),
            spectrum: vec![0.0; visualizer.borrow().block_size / 2 + 1],
        }
    }
}

impl eframe::App for VisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut visualizer = self.visualizer.borrow_mut();

        if visualizer.current_position < visualizer.processed_chunks.len() {
            self.spectrum = visualizer.processed_chunks[visualizer.current_position].clone();
            visualizer.current_position += 1;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let bar_width = 5.0;
            for (i, &y) in self.spectrum.iter().enumerate().step_by(1) {
                let x_pos = i as f32 * (5.0 + 2.0);
                let height = y * 60.0;
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(
                        egui::Pos2::new(x_pos, 600.0 - height),
                        egui::vec2(bar_width, height),
                    ),
                    0.0,
                    egui::Color32::RED,
                );
            }
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let visualizer = AudioVisualizer::new(&args.audio_file, 2048)?;

    visualizer.play_audio(&args.audio_file)?;

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Audio Visualizer",
        options,
        Box::new(|_| Box::new(VisualizerApp::new(Rc::new(RefCell::new(visualizer))))),
    )?;
    Ok(())
}
