pub struct HotNotDogClassifier {
    model: squeezenet::Classifier<Backend>,
    normalizer: Normalizer,
}

impl HotNotDogClassifier {
    pub fn new() -> Self {
        let squeezenet_imported = squeezenet::Model::<Backend>::from_embedded();
        let model = squeezenet::Classifier::<Backend>::new_from_squeezenet(&squeezenet_imported);
        drop(squeezenet_imported);
        let normalizer = Normalizer::new();
        Self { model, normalizer }
    }

    pub fn predict(&self, image: Tensor<Backend, 4>) -> Tensor<Backend, 1> {
        let image = self.normalizer.normalize(image);
        self.model.forward(&image);
        // Get the argmax of the output
        let arg_max = output.argmax(1).into_scalar() as usize;

        // Get the label from the argmax
        let label = LABELS[arg_max];
    }
}
