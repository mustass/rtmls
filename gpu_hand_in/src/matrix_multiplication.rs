use crate::{utility::{GPUHandles, mean_square_error, are_vectors_equivalent, Uniform, run_compute_shader}, gpu_vector::GPUVector};
use core::cmp::max;

fn matrix_multiplication_cpu(
    left_matrix: &Vec<f32>,
    right_matrix: &Vec<f32>,
    outer_dimension_left_length: usize,
    inner_dimension_length: usize,
    outer_dimension_right_length: usize,
) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; outer_dimension_left_length * outer_dimension_right_length];
    for row_output in 0..outer_dimension_left_length {
        for column_output in 0..outer_dimension_right_length {
            for inner_dimension in 0..inner_dimension_length {
                output[row_output * outer_dimension_right_length + column_output] += left_matrix
                    [row_output * inner_dimension_length + inner_dimension]
                    * right_matrix[inner_dimension * outer_dimension_right_length + column_output];
            }
        }
    }
    output
}

fn test_ground_truth() -> bool {
    let outer_dimension_left: usize = 4;
    let inner_dimension: usize = 3;
    let outer_dimension_right: usize = 3;
    let left: Vec<f32> = vec![ 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0];
    let right: Vec<f32> = vec![ 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 2.0];
    let ground_truth_output: Vec<f32> = vec![ 5.0, 4.0, 3.0, 8.0, 9.0, 5.0, 6.0, 5.0, 3.0, 11.0, 9.0, 6.0];

    let output: Vec<f32> = matrix_multiplication_cpu(&left, &right, outer_dimension_left, inner_dimension, outer_dimension_right);

    assert!(output.len() == ground_truth_output.len());
    for index in 0..ground_truth_output.len() {
        if 0.00001 < (output[index] - ground_truth_output[index]).abs() {
            return false;
        }
    }

    true
}

pub fn matrix_multiplication(handles: &GPUHandles) -> bool {
    // A small test to ensure that the matrix_multiplication_cpu function is actually correct.
    let ground_truth_is_correct: bool = test_ground_truth();
    println!("Matrix multiplication ground truth function is correct: {}", ground_truth_is_correct);
    assert!(ground_truth_is_correct);


    // Use big data dimensions to make sure the cost of transferring
    // doesn't dominate the time spent in the function.
    let outer_dimension_left: usize = 1710; // M
    let inner_dimension: usize = 241; // N
    let outer_dimension_right: usize = 3512;// K
    let left_matrix: Vec<f32> = (0..outer_dimension_left*inner_dimension).map(|x| x as f32 * 1.0).collect();
    let right_matrix: Vec<f32> = (0..inner_dimension*outer_dimension_right).map(|x| x as f32 * -0.1).collect();
    let ground_truth: Vec<f32> = matrix_multiplication_cpu(&left_matrix, &right_matrix, outer_dimension_left, inner_dimension, outer_dimension_right);
    
    //
    // 1) Make one version of matrix multiplication using the GPU. Ensure that it is correct.
    //
    // 2) Make another version which uses tiling through shared memory and local accumulation in a register.
    // A tiling reference: http://www.csce.uark.edu/~mqhuang/courses/4643/s2016/lecture/GPU_Lecture_3.pdf
    //
    // 3) After ensuring correctness - time the two functions.
    // 
    // 4) How big do the matrices have to be before you see a big performance difference?
    //
    // 5) What happens when you set the block size to different multiples of 32? Why do you think that is?
    //
    // 6) What is the optimal tile size?
    //
    // 7) Make a third version starting from the tiled version, but pads the matrices with 0's to the nearest
    // multiple of the block size? So, if you had a block size of 32 and a 30x30 * 28x29
    // multiplication you padded both with 0's to get 32x32 * 32x32.
    // HINT - You can now remove some if-guards.
    //
    // HINT - You need a run_compute_shader() call per type of compute shader.
    // Figure out what the arguments are supposed to be (see vector_add.rs) and
    // call the correct shader function in the correct shader file.
    //


    let left_padded_dim:i32 = (outer_dimension_left as f32 / 16.0).ceil() as i32 * 16;
    let right_padded_dim:i32 = (outer_dimension_right as f32 / 16.0).ceil() as i32 * 16;
    let inner_padded_dim:i32 = (inner_dimension as f32 / 16.0).ceil() as i32 * 16;

    let padded_dim:i32 = max(max(left_padded_dim, right_padded_dim), inner_padded_dim) ;
    let padded_dim:usize = padded_dim as usize;


    // Pad left matrix
    let mut left_matrix_padded: Vec<f32> = vec![0.0; (padded_dim*padded_dim) as usize];
    for row in 0..outer_dimension_left {
        for col in 0..inner_dimension {
            left_matrix_padded[(row*padded_dim + col) as usize] = left_matrix[(row*inner_dimension + col) as usize];
        }
    }

    // Pad right matrix
    let mut right_matrix_padded: Vec<f32> = vec![0.0; (padded_dim*padded_dim) as usize];
    for row in 0..inner_dimension {
        for col in 0..outer_dimension_right {
            right_matrix_padded[(row*padded_dim + col) as usize] = right_matrix[(row*outer_dimension_right + col) as usize];
        }
    }



    let uniform: Uniform = Uniform::new(handles, outer_dimension_right, outer_dimension_left, inner_dimension, 0);
    let left_matrix_gpu: GPUVector = GPUVector::new(&handles, left_matrix.clone(), "left_matrix", false);
    let right_matrix_gpu: GPUVector = GPUVector::new(&handles, right_matrix.clone(), "right_matrix", false);
    let mut output_gpu_naive: GPUVector = GPUVector::new(&handles, vec![0.0; outer_dimension_left*outer_dimension_right], "output", true);
    let mut output_gpu_tiled: GPUVector = GPUVector::new(&handles, vec![0.0; outer_dimension_left*outer_dimension_right], "output", true);

    let uniform_padded: Uniform = Uniform::new(handles, padded_dim, padded_dim, padded_dim, 0);
    let left_matrix_padded_gpu: GPUVector = GPUVector::new(&handles, left_matrix_padded.clone(), "left_matrix", false);
    let right_matrix_padded_gpu: GPUVector = GPUVector::new(&handles, right_matrix_padded.clone(), "right_matrix", false);
    let mut output_gpu_padded: GPUVector = GPUVector::new(&handles, vec![0.0; padded_dim*padded_dim], "output", true);


    let block_size_x: usize = 16;
    let max_outer_dims = max(outer_dimension_left, outer_dimension_right);
    let launch_blocks_x: u32 = ((max_outer_dims + block_size_x - 1) / block_size_x) as u32;
    let block_size_y: usize = 16;
    let launch_blocks_y: u32 = ((max_outer_dims + block_size_y - 1) / block_size_y) as u32;
    let shader_file_naive: &'static str = include_str!("matrix_multiplication_naive.wgsl");
    let shader_function_naive: &str = "matmul_naive";


    run_compute_shader(
        handles,
        block_size_x, 
        launch_blocks_x,
        block_size_y,
        launch_blocks_y,
        shader_file_naive,
        shader_function_naive,
        &uniform,
        &left_matrix_gpu,
        &right_matrix_gpu,
        &mut output_gpu_naive,
    );

    let shader_file_tiled: &'static str = include_str!("matrix_multiplication_tiled.wgsl");
    let shader_function_tiled: &str = "matmul_tiled";

    run_compute_shader(
        handles,
        block_size_x, 
        launch_blocks_x,
        block_size_y,
        launch_blocks_y,
        shader_file_tiled,
        shader_function_tiled,
        &uniform,
        &left_matrix_gpu,
        &right_matrix_gpu,
        &mut output_gpu_tiled,
    );

    // 
    
    let shader_file_padded: &'static str = include_str!("matrix_multiplication_padded.wgsl");
    let shader_function_padded: &str = "matmul_padded";

    run_compute_shader(
        handles,
        block_size_x, 
        launch_blocks_x,
        block_size_y,
        launch_blocks_y,
        shader_file_padded,
        shader_function_padded,
        &uniform_padded,
        &left_matrix_padded_gpu,
        &right_matrix_padded_gpu,
        &mut output_gpu_padded,
    );

    let mut output_gpu_padded_cpu: Vec<f32> = vec![0.0; outer_dimension_left*outer_dimension_right];
    for row in 0..outer_dimension_left {
        for col in 0..outer_dimension_right {
            output_gpu_padded_cpu[(row*outer_dimension_right + col) as usize] = output_gpu_padded.cpu_data[(row*padded_dim + col) as usize];
        }
    }



    //
    // YOUR CODE HERE
    let data_naive: Vec<f32> = output_gpu_naive.cpu_data; // Remove this and replace with your own data
    let data_tiled: Vec<f32> = output_gpu_tiled.cpu_data; // Remove this and replace with your own data
    let data_padded: Vec<f32> = output_gpu_padded_cpu; // Remove this and replace with your own data
    //

    // Naive
    println!("matrix multiplication naive MSE: {}", mean_square_error(&ground_truth, &data_naive));
    let success: bool = are_vectors_equivalent(&ground_truth, &data_naive);
    println!("matrix multiplication naive success: {}!", success);

    // Tiled
    println!("matrix multiplication tiled MSE: {}", mean_square_error(&ground_truth, &data_tiled));
    let success: bool = are_vectors_equivalent(&ground_truth, &data_tiled);
    println!("matrix multiplication tiled success: {}!", success);

    // Padded
    println!("matrix multiplication padded MSE: {}", mean_square_error(&ground_truth, &data_padded));
    let success: bool = are_vectors_equivalent(&ground_truth, &data_padded);
    println!("matrix multiplication padded success: {}!", success);

    success
}