use eframe::{
    egui::{CentralPanel, RichText, SidePanel},
    epaint::Color32,
    App,
};
use std::fmt;

use burn::{backend::{Autodiff, Wgpu}, tensor::{backend::Backend, Data, Int, Tensor}};

use hotnotdog::utils::{load_data, HotNotDogsData, Command};
use hotnotdog::model::squeezed_classifier::load_image;

#[derive(Default)]
pub struct HotNotDogApp<B:Backend> {
    stream: Vec<HotNotDogsData>,
    true_label: TrueLabel,
    prediction: TrueLabel,
    show_prediction: bool,
    show_training: bool,
    current_image: usize,
    _marker: std::marker::PhantomData<B>,
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

impl<B: Backend> App for HotNotDogApp<B> {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame, sender, reciever )
    {

        let no_tensor: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::from_data(Data::<i64, 1>::from([0 as i64]).convert());
        let yes_tensor: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::from_data(Data::<i64, 1>::from([1 as i64]).convert());


        SidePanel::right("side_panel").show(ctx, |ui| {
            ui.heading("Play the Hotdog Game");
            ui.label("Predict!");
            // add button to run prediction on displayed image

            ui.horizontal(|ui| {
                if ui.button("Predict").clicked() {
                    println!("Predicting");

                    let image: burn::tensor::Tensor<B, 4> = load_image(&self.stream[self.current_image].image_path);
                    sender.send(Command::Predict{image}).unwrap();
                    let prediction = reciever.recv().unwrap();
                    // access the prediction
                    let prediction = match prediction {
                        Command::Prediction{value} => value,
                        _ => panic!("Wrong command"),
                    };

                    self.prediction = match prediction {
                        "hotdog" => TrueLabel::HotDog,
                        "not_hotdog" => TrueLabel::NotHotDog,
                        _ => panic!("Wrong prediction"),
                    };

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
                    ui.label(self.prediction.to_string());
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

                    let image: burn::tensor::Tensor<B, 4> = load_image(&self.stream[self.current_image].image_path);
                    let label = match self.true_label {
                        TrueLabel::HotDog => Tensor::<B, 1, Int>::from_data(Data::<i64, 1>::from([1 as i64]).convert()),
                        TrueLabel::NotHotDog => Tensor::<B, 1, Int>::from_data(Data::<i64, 1>::from([0 as i64]).convert())
                    };

                    sender.send(Command::Train{image, label}).unwrap();

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

impl<B:Backend> HotNotDogApp<B> {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            stream: load_data(),
            true_label: TrueLabel::HotDog,
            prediction: TrueLabel::HotDog,
            show_prediction: false,
            show_training: false,
            current_image: 0,
            _marker: std::marker::PhantomData,
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

