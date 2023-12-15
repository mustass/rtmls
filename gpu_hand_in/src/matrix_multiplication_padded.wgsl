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

var<workgroup> A : array<f32,256>;
var<workgroup> B : array<f32,256>;

@compute @workgroup_size(16, 16, 1)

fn matmul_padded(
@builtin(local_invocation_id) local_id : vec3 <u32>,
@builtin(workgroup_id) group_id : vec3 <u32>,
)
{
    let block_size: i32 = 16;
    let tx: i32 = i32(local_id.x);
    let ty: i32 = i32(local_id.y);
    let bx: i32 = i32(group_id.x);
    let by: i32 = i32(group_id.y);
    
    let row = by * block_size + ty;
    let col = bx * block_size + tx;

    let num_tiles =  dimensions.outer_right / block_size;

    var sum : f32 = 0.0;

    for (var i = 0; i < num_tiles; i = i + 1) {
  
        A[ty * 16 + tx] = left_matrix[row * dimensions.inner + i * 16 + tx];
        B[ty * 16 + tx] = right_matrix[(i * 16 + ty) * dimensions.outer_right + col];

        workgroupBarrier();

        for (var k = 0; k < 16; k = k + 1) {
            sum += A[ty * 16 + k] * B[k * 16 + tx];
        }

        workgroupBarrier();

        output[row * dimensions.outer_right + col] = sum;
    } 


}