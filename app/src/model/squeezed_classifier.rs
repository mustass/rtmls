
use burn::backend::NdArray;
use burn::tensor::Tensor;

use crate::model::normalizer::Normalizer;
use crate::model::squeezenet;
use crate::model::label::LABELS_DOG;

type Backend = NdArray<f32>;


#[derive(Default)]
pub struct HotNotDogClassifier {
    model: squeezenet::Classifier<Backend>,
    normalizer: Normalizer<Backend>,
}

impl HotNotDogClassifier {
    pub fn new() -> Self {
        let squeezenet_imported = squeezenet::Model::<Backend>::from_embedded();
        let model = squeezenet::Classifier::<Backend>::new_from_squeezenet(&squeezenet_imported);
        drop(squeezenet_imported);
        let normalizer = Normalizer::new();
        Self { model, normalizer }
    }

    pub fn predict(&self, image: Tensor<Backend, 4>) -> &'static str {
        let image = self.normalizer.normalize(image);
        let output = self.model.forward(image);
        // Get the argmax of the output
        let arg_max = output.argmax(1).into_scalar() as usize;

        // Get the label from the argmax
        let label = LABELS_DOG[arg_max];
        label
    }
}
