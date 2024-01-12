use burn::tensor::{Tensor, backend::Backend};
use rand::seq::SliceRandom;

pub struct HotNotDogsData {
    pub image_path: String,
    pub label: bool,
}

pub fn load_data() -> Vec<HotNotDogsData> {

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



#[derive(Debug, Clone)]
pub enum Command<B: Backend>{
    Predict{image: Tensor<B, 4>},
    Train{image: Tensor<B, 4>, label: Tensor<B, 1, burn::tensor::Int>},
    Prediction{value: &'static str},
    Shutdown{value: bool},
}

// implement default

impl<B: Backend> Default for Command<B> {
    fn default() -> Self {
        Command::Shutdown{value: false}
    }
}

