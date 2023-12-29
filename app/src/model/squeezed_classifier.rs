
use burn::backend::NdArray;
use burn::tensor::Tensor;

use crate::model::normalizer::Normalizer;
use crate::model::squeezenet;
use crate::model::label::LABELS_DOG;
use image::{self, GenericImageView, Pixel};
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

    pub fn load_image(&self, path: &str) -> Tensor<Backend, 4> {
        let img = image::open(&path).unwrap_or_else(|_| panic!("Failed to load image: {path}"));
        let resized_img = img.resize_exact( 224, 224, image::imageops::FilterType::Lanczos3);
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
        let image_input = Tensor::<Backend, 3>::from_data(img_array).reshape([1, 3, 224, 224]);
        image_input
    }
}
