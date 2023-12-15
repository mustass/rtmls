struct Uniform {
    outer_right : i32,
    outer_left : i32,
    inner : i32,
    not_used : u32,
};

@group(0) @binding(0)
var<uniform> dimensions : Uniform;

//Bind a read only array of 32-bit floats
@group(0) @binding(1)
var<storage, read> left_matrix : array<f32>;

@group(0) @binding(2)
var<storage, read> right_matrix : array<f32>;

//Bind a read/write array
@group(0) @binding(3)
var<storage, read_write> output : array<f32>;

@compute @workgroup_size(16, 16, 1)

fn matmul_naive(
@builtin(local_invocation_id) local_id : vec3 <u32>,
@builtin(workgroup_id) group_id : vec3 <u32>,
)
{
    let row = i32(group_id.y) * i32(16) + i32(local_id.y);
    let col = i32(group_id.x) * i32(16) + i32(local_id.x);

    var sum : f32 = 0.0;
    if (row < dimensions.outer_left && col < dimensions.outer_right) {
        for (var i = 0; i < dimensions.inner; i = i + 1) {
            sum += left_matrix[row * dimensions.inner+ i] * right_matrix[i * dimensions.outer_right + col];
        }
        output[row * dimensions.outer_right + col] = sum;
    }
}