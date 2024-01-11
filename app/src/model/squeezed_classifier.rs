use burn::{
    nn::loss::CrossEntropyLoss,
    optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, Sgd, SgdConfig},
    tensor::{Data, Float},
};
use crossbeam_channel::{Receiver, TryRecvError, Sender};
use crate::model::label::LABELS_DOG;
use crate::model::normalizer::Normalizer;
use crate::model::squeezenet;
use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    Int, Tensor,
};
use image::{self, GenericImageView, Pixel};

use super::squeezenet::Classifier;
use super::super::utils::Command;
use num_traits::cast::ToPrimitive;
pub struct HotNotDogClassifier<B: AutodiffBackend> {
    model: Classifier<B>,
    normalizer: Normalizer<B>,
    pub optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, Classifier<B>, B>,
    shutdown: bool,
}

impl<B: AutodiffBackend> Default for HotNotDogClassifier<B> {
    fn default() -> Self {
        Self {
            model: squeezenet::Classifier::<B>::default(),
            normalizer: Normalizer::<B>::default(),
            optimizer: SgdConfig::new().init(),
            shutdown: false,
        }
    }
}

impl<B: AutodiffBackend> HotNotDogClassifier<B> {
    pub fn new() -> Self {
        let squeezenet_imported = squeezenet::Model::<B>::from_embedded();
        let model = squeezenet::Classifier::<B>::new_from_squeezenet(&squeezenet_imported);
        drop(squeezenet_imported);
        let normalizer = Normalizer::<B>::new();
        let optim = SgdConfig::new().init();
        Self {
            model,
            normalizer,
            optimizer: optim,
            shutdown: false,
        }
    }

    pub fn predict(&self, image: Tensor<B, 4>) -> &'static str
    {
        let image = self.normalizer.normalize(image);
        let output = self.model.forward(image);

        let arg_max = output.argmax(1).into_scalar().to_usize().unwrap() as usize;

        let label = LABELS_DOG[arg_max];

        label
    }

    pub fn train(&mut self, image: Tensor<B, 4>, label: Tensor<B, 1, Int>) -> () {
        let image = self.normalizer.normalize(image);
        let prediction = self.model.forward(image);
        let loss = CrossEntropyLoss::new(None).forward(prediction.clone(), label.clone());
        // print the loss please
        println!("Loss: {}", loss);

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &self.model);
        
        // Update the parameters of the model.
        let updated_model = self.optimizer.step(0.10, self.model.clone(), grads);
        self.model = updated_model;
    }

    pub fn shutdown(&mut self) {}

    pub fn run_loop(&mut self, command_receiver: Receiver<Command<B>>, command_sender: Sender<Command<B>>) {
        while !self.shutdown {
            loop {
                let result: Result<Command<B>, TryRecvError> = command_receiver.try_recv();
                if result.is_err() {
                    break;
                }
                let command: Command<B> = result.unwrap();
                match command {
                    Command::Predict{image} => {
                       let label = self.predict(image);
                          command_sender.send(Command::Prediction{value: label}).unwrap();
                    }
                    Command::Train{image, label} => {
                        self.train(image, label);
                    }
                    Command::Shutdown { value } => {
                        if value {
                            self.shutdown = true;
                        }
                    }
                    Command::Prediction {value: _ } => {}
                }
            }
        }

        self.shutdown();
    }
}

pub fn load_image<B: Backend>(path: &str) -> Tensor<B, 4>
{
    let img = image::open(&path).unwrap_or_else(|_| panic!("Failed to load image: {path}"));
    let resized_img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let mut img_array = [[[0.0; 224]; 224]; 3];
    for y in 0..224usize {
        for x in 0..224usize {
            let pixel = resized_img.get_pixel(x as u32, y as u32);
            let rgb = pixel.to_rgb();

            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }
    let data = Data::<f32, 3>::from(img_array);
    let tensor = Tensor::<B, 3>::from_data(data.convert());
    let image_input = tensor.reshape([1, 3, 224, 224]);
            
    image_input
}
