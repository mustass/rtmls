pub mod label;
pub mod normalizer;
pub mod squeezed_ckassifier;
pub mod squeezenet;

use burn::backend::NdArray;
use burn::tensor::Tensor;

use normalizer::Normalizer;

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

type Backend = NdArray<f32>;
