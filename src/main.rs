use clap::Parser;
use hound::WavReader;
use realfft::RealFftPlanner;
use rodio::{Decoder, OutputStream, Sink};
use rustfft::num_complex::Complex;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::thread;
use eframe::egui;

#[derive(Parser, Debug)]
#[clap(about = "Audio visualizer that displays frequency spectrum")]
struct Args {
    #[clap(value_parser)]
    audio_file: String,
}

struct AudioVisualizer {
    audio_data: Vec<f32>,
    block_size: usize,
    current_position: Arc<Mutex<usize>>,
}

impl AudioVisualizer {
    fn new(audio_file: &str, block_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut reader = WavReader::open(audio_file)?;

        let audio_data: Vec<f32> = reader.samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / 32768.0)
            .collect();

        Ok(AudioVisualizer {
            audio_data,
            block_size,
            current_position: Arc::new(Mutex::new(0)),
        })
    }

    fn play_audio(&self, audio_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (_stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;
        let file = File::open(audio_file)?;
        let source = Decoder::new(BufReader::new(file))?;
        sink.append(source);
        sink.play();

        thread::spawn(move || {
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

        let mut pos = self.visualizer.current_position.lock().unwrap();
        if *pos + self.visualizer.block_size <= self.visualizer.audio_data.len() {
            input_buffer.copy_from_slice(&self.visualizer.audio_data[*pos..*pos + self.visualizer.block_size]);
            *pos += self.visualizer.block_size;
            r2c.process(&mut input_buffer, &mut spectrum_output).ok();
            self.spectrum = spectrum_output.iter().map(|c| (c.norm() + 1.0).ln()).collect();
        }
        
        egui::CentralPanel::default().show(ctx, |ui| {
            let points: Vec<egui::Pos2> = self.spectrum.iter().enumerate()
                .map(|(i, &y)| egui::Pos2::new(i as f32, 600.0 - y * 20.0))
                .collect();
            ui.painter().add(egui::epaint::Shape::line(points, egui::Stroke::new(1.5, egui::Color32::GREEN)));
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let visualizer = Arc::new(AudioVisualizer::new(&args.audio_file, 2048)?);
    visualizer.play_audio(&args.audio_file)?;
    
    let options = eframe::NativeOptions::default();
    eframe::run_native("Audio Visualizer", options, Box::new(|_| Box::new(VisualizerApp::new(visualizer))))?;
    Ok(())
}