// Generated from ONNX "src/model/squeezenet1.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::PaddingConfig2d;
//use burn::nn::LinearConfig;
//use burn::nn::Linear;
use burn::record::BinBytesRecorder;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    dropout1: Dropout,
    conv2d26: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    //linear: Linear<B>,
    phantom: core::marker::PhantomData<B>,
}

static EMBEDDED_STATES: &[u8] = include_bytes!("/home/sm/Dropbox/DTU/rtc/rtmls/hotdog/squeezenet/squeezenet1.bin");

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_embedded()
    }
}

impl<B: Backend> Model<B> {
    pub fn from_embedded() -> Self {
        let record:ModelRecord<B> = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(EMBEDDED_STATES.to_vec())
            .expect("Failed to decode state");
        Self::new_with(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new_with(record: ModelRecord<B>) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d1);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d2 = Conv2dConfig::new([64, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d2);
        let conv2d3 = Conv2dConfig::new([16, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d3);
        let conv2d4 = Conv2dConfig::new([16, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d4);
        let conv2d5 = Conv2dConfig::new([128, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d5);
        let conv2d6 = Conv2dConfig::new([16, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d6);
        let conv2d7 = Conv2dConfig::new([16, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d7);
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d8 = Conv2dConfig::new([128, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d8);
        let conv2d9 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d9);
        let conv2d10 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d10);
        let conv2d11 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d11);
        let conv2d12 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d12);
        let conv2d13 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d13);
        let maxpool2d3 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d14 = Conv2dConfig::new([256, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d14);
        let conv2d15 = Conv2dConfig::new([48, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d15);
        let conv2d16 = Conv2dConfig::new([48, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d16);
        let conv2d17 = Conv2dConfig::new([384, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d17);
        let conv2d18 = Conv2dConfig::new([48, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d18);
        let conv2d19 = Conv2dConfig::new([48, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d19);
        let conv2d20 = Conv2dConfig::new([384, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d20);
        let conv2d21 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d21);
        let conv2d22 = Conv2dConfig::new([64, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d22);
        let conv2d23 = Conv2dConfig::new([512, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d23);
        let conv2d24 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d24);
        let conv2d25 = Conv2dConfig::new([64, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d25);
        let dropout1 = DropoutConfig::new(0.5).init();
        let conv2d26 = Conv2dConfig::new([512, 1000], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init_with(record.conv2d26);
        let averagepool2d1 = AvgPool2dConfig::new([13, 13])
            .with_strides([13, 13])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .init();
        //let linear = LinearConfig::new(1000, 2)
        //    .init();

        Self {
            conv2d1,
            maxpool2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d2,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            maxpool2d3,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            dropout1,
            conv2d26,
            averagepool2d1,
            //linear,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d2 = Conv2dConfig::new([64, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d3 = Conv2dConfig::new([16, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d4 = Conv2dConfig::new([16, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d5 = Conv2dConfig::new([128, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d6 = Conv2dConfig::new([16, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d7 = Conv2dConfig::new([16, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d8 = Conv2dConfig::new([128, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d9 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d10 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d11 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d12 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d13 = Conv2dConfig::new([32, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let maxpool2d3 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d14 = Conv2dConfig::new([256, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d15 = Conv2dConfig::new([48, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d16 = Conv2dConfig::new([48, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d17 = Conv2dConfig::new([384, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d18 = Conv2dConfig::new([48, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d19 = Conv2dConfig::new([48, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d20 = Conv2dConfig::new([384, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d21 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d22 = Conv2dConfig::new([64, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d23 = Conv2dConfig::new([512, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d24 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let conv2d25 = Conv2dConfig::new([64, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let dropout1 = DropoutConfig::new(0.5).init();
        let conv2d26 = Conv2dConfig::new([512, 1000], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init();
        let averagepool2d1 = AvgPool2dConfig::new([13, 13])
            .with_strides([13, 13])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .init();
        //let linear = LinearConfig::new(1000, 2)
        //    .with_bias(true)
        //    .init();

        Self {
            conv2d1,
            maxpool2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d2,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            maxpool2d3,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            dropout1,
            conv2d26,
            averagepool2d1,
            //linear,
            phantom: core::marker::PhantomData,
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);
        let conv2d2_out1 = self.conv2d2.forward(maxpool2d1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1.clone());
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu2_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([relu3_out1, relu4_out1].into(), 1);
        let conv2d5_out1 = self.conv2d5.forward(concat1_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu5_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([relu6_out1, relu7_out1].into(), 1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(concat2_out1);
        let conv2d8_out1 = self.conv2d8.forward(maxpool2d2_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu8_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([relu9_out1, relu10_out1].into(), 1);
        let conv2d11_out1 = self.conv2d11.forward(concat3_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1.clone());
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu11_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([relu12_out1, relu13_out1].into(), 1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(concat4_out1);
        let conv2d14_out1 = self.conv2d14.forward(maxpool2d3_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu14_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let concat5_out1 = burn::tensor::Tensor::cat([relu15_out1, relu16_out1].into(), 1);
        let conv2d17_out1 = self.conv2d17.forward(concat5_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu17_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let concat6_out1 = burn::tensor::Tensor::cat([relu18_out1, relu19_out1].into(), 1);
        let conv2d20_out1 = self.conv2d20.forward(concat6_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu20_out1.clone());
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu20_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let concat7_out1 = burn::tensor::Tensor::cat([relu21_out1, relu22_out1].into(), 1);
        let conv2d23_out1 = self.conv2d23.forward(concat7_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu23_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu23_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let concat8_out1 = burn::tensor::Tensor::cat([relu24_out1, relu25_out1].into(), 1);
        let dropout1_out1 = self.dropout1.forward(concat8_out1);
        let conv2d26_out1 = self.conv2d26.forward(dropout1_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(relu26_out1);
        let reshape1_out1 = averagepool2d1_out1.reshape([0, -1]);
        //let final_output = self.linear.forward(reshape1_out1);
        reshape1_out1
    }
}
