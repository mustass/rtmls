use eframe::{
    egui::{CentralPanel, RichText, SidePanel},
    epaint::Color32,
    App,
};
use rand::seq::SliceRandom;
use std::fmt;

use burn::backend::{Autodiff, Wgpu};

use hotnotdog::model::squeezed_classifier::HotNotDogClassifier;
use hotnotdog::model::squeezed_classifier::load_image;

#[derive(Default)]
pub struct HotNotDogApp {
    stream: Vec<HotNotDogsData>,
    model: HotNotDogClassifier<Autodiff<Wgpu>>,
    true_label: TrueLabel,
    show_prediction: bool,
    prediction: Option<TrueLabel>,
    show_training: bool,
    current_image: usize,
}

struct HotNotDogsData {
    image_path: String,
    label: bool,
}

#[derive(PartialEq, Default)]
pub enum TrueLabel {
    #[default]
    NotHotDog,
    HotDog,
}

impl fmt::Display for TrueLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrueLabel::NotHotDog => write!(f, "NotHotDog"),
            TrueLabel::HotDog => write!(f, "HotDog"),
        }
    }
}

impl App for HotNotDogApp {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Hot or Not Dog");
                // concat
                let mut path = String::new();
                path.push_str("file://");
                path.push_str(&self.stream[self.current_image].image_path);

                ui.image(path);
            })
        });
        SidePanel::right("side_panel").show(ctx, |ui| {
            ui.heading("Play the Hotdog Game");
            ui.label("Predict!");
            // add button to run prediction on displayed image

            ui.horizontal(|ui| {
                if ui.button("Predict").clicked() {
                    println!("Predicting");

                    let image: burn::tensor::Tensor<Autodiff<Wgpu>, 4> = load_image(&self.stream[self.current_image].image_path);
                    let prediction = self.model.predict(image);
                    self.prediction = Some(match prediction {
                        "hot_dog" => TrueLabel::HotDog,
                        _ => TrueLabel::NotHotDog,
                    });
                    self.show_prediction = true;
                }
                if ui.button("Train Me").clicked() {
                    println!("Training enabeled");
                    self.show_training = true;
                }
            });

            // add separator
            if self.show_prediction {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Prediction:");
                    ui.label(self.prediction.as_ref().unwrap().to_string());
                });
            }

            if self.show_training {
                ui.separator();
                ui.label("Train Me!");
                // checkbox with two options
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.true_label, TrueLabel::HotDog, "HotDog");
                    ui.selectable_value(&mut self.true_label, TrueLabel::NotHotDog, "NotHotDog");
                });

                if ui
                    .button(RichText::new("Submit").color(Color32::DARK_BLUE))
                    .clicked()
                {
                    println!("Submitting");

                    let image: burn::tensor::Tensor<Autodiff<Wgpu>, 4> = load_image(&self.stream[self.current_image].image_path);
                    let label = match self.true_label {
                        TrueLabel::HotDog => burn::tensor::Tensor::<Autodiff<Wgpu>, 1, burn::tensor::Int>::from_data([1]),
                        TrueLabel::NotHotDog => burn::tensor::Tensor::<Autodiff<Wgpu>, 1, burn::tensor::Int>::from_data([0]),
                    };

                    self.model.train(image, label);

                    println!("True label: {}", self.true_label);
                }
            }
            // add separator
            ui.separator();

            // add button to get next image
            if ui.button("Next").clicked() {
                println!("Next");
                self.next_image();
            }

            // add separator
            ui.separator();

            //
        });
    }
}

impl HotNotDogApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            stream: load_data(),
            model: HotNotDogClassifier::new(),
            true_label: TrueLabel::HotDog,
            show_prediction: false,
            prediction: None,
            show_training: false,
            current_image: 0,
        }
    }

    fn next_image(&mut self) {
        self.current_image += 1;
        if self.current_image >= self.stream.len() {
            self.current_image = 0;
        }

        self.show_prediction = false;
        self.show_training = false;
    }
}

fn load_data() -> Vec<HotNotDogsData> {

    let mut stream: Vec<HotNotDogsData> = Vec::new();

    let hotdog_path = "./artifacts/seefood_imgs/train/hot_dog/";
    let not_hotdog_path = "./artifacts/seefood_imgs/train/not_hot_dog/";

    let hotdog_files = std::fs::read_dir(hotdog_path).unwrap();
    let not_hotdog_files = std::fs::read_dir(not_hotdog_path).unwrap();

    for entry in hotdog_files {
        let entry = entry.unwrap();
        let path = entry.path();
        let path_str = path.to_str().unwrap();
        stream.push(HotNotDogsData {
            image_path: path_str.to_string(),
            label: true,
        });
    }

    for entry in not_hotdog_files {
        let entry = entry.unwrap();
        let path = entry.path();
        let path_str = path.to_str().unwrap();
        stream.push(HotNotDogsData {
            image_path: path_str.to_string(),
            label: false,
        });
    }

    // shuffle the stream
    stream.shuffle(&mut rand::thread_rng());

    stream
}
