use clap::Parser;
use eframe::egui;
use hound::WavReader;
use realfft::RealFftPlanner;
use rodio::{Decoder, OutputStream, Sink};
use rustfft::num_complex::Complex;
use std::fs::File;
use std::io::BufReader;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;

#[derive(Parser, Debug)]
#[clap(about = "Audio visualizer that displays frequency spectrum")]
struct Args {
    #[clap(value_parser)]
    audio_file: String,
}

struct AudioVisualizer {
    audio_data: Vec<f32>,
    block_size: usize,
    current_position: AtomicUsize,
}

impl AudioVisualizer {
    fn new(audio_file: &str, block_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = WavReader::open(audio_file)?;

        let audio_data: Vec<f32> = reader
            .samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / 32768.0)
            .collect();

        Ok(AudioVisualizer {
            audio_data,
            block_size,
            current_position: AtomicUsize::new(0),
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
    visualizer: Arc<AudioVisualizer>,
    spectrum: Vec<f32>,
}

impl VisualizerApp {
    fn new(visualizer: Arc<AudioVisualizer>) -> Self {
        Self {
            visualizer: visualizer.clone(),
            spectrum: vec![0.0; visualizer.block_size / 2 + 1],
        }
    }
}

impl eframe::App for VisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(self.visualizer.block_size);
        let mut input_buffer = vec![0.0f32; self.visualizer.block_size];
        let mut spectrum_output = vec![Complex::new(0.0, 0.0); self.visualizer.block_size / 2 + 1];

        let pos = self.visualizer.current_position.load(Ordering::Relaxed);
        if pos + self.visualizer.block_size <= self.visualizer.audio_data.len() {
            input_buffer.copy_from_slice(
                &self.visualizer.audio_data[pos..pos + self.visualizer.block_size],
            );
            // We don't need to CAS because we're the only writer.
            self.visualizer
                .current_position
                .store(pos + self.visualizer.block_size, Ordering::Relaxed);
            r2c.process(&mut input_buffer, &mut spectrum_output).ok();
            self.spectrum = spectrum_output
                .iter()
                .map(|c| (c.norm() + 1.0).ln())
                .collect();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let bar_width = 5.0;
            // let gap = 1.0;
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
    let visualizer = Arc::new(AudioVisualizer::new(&args.audio_file, 2048)?);
    visualizer.play_audio(&args.audio_file)?;

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Audio Visualizer",
        options,
        Box::new(|_| Box::new(VisualizerApp::new(visualizer))),
    )?;
    Ok(())
}
