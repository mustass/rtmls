
use burn::{backend::NdArray, nn::loss::CrossEntropyLoss};

use crate::model::normalizer::Normalizer;
use crate::model::squeezenet;
use crate::model::label::LABELS_DOG;
use image::{self, GenericImageView, Pixel};
use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    ElementConversion, Int, Tensor,
};


#[derive(Default)]
pub struct HotNotDogClassifier<B: AutodiffBackend> {
    model: squeezenet::Classifier<B>,
    normalizer: Normalizer<B>,
}

impl<B: AutodiffBackend> HotNotDogClassifier<B> {
    pub fn new() -> Self {
        let squeezenet_imported = squeezenet::Model::<B>::from_embedded();
        let model = squeezenet::Classifier::<B>::new_from_squeezenet(&squeezenet_imported);
        drop(squeezenet_imported);
        let normalizer = Normalizer::<B>::new();
        Self { model, normalizer }
    }

    
}
