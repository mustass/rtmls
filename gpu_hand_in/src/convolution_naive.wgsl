struct Uniform {
    element_count : u32,
    filter_size : i32,
    not_used : u32,
    not_used : u32,
};

@group(0) @binding(0)
var<uniform> dimensions : Uniform;

//Bind a read only array of 32-bit floats
@group(0) @binding(1)
var<storage, read> input_signal : array<f32>;

@group(0) @binding(2)
var<storage, read> filter_signal : array<f32>;

//Bind a read/write array
@group(0) @binding(3)
var<storage, read_write> output : array<f32>;

@compute @workgroup_size(32, 1, 1)

fn conv_naive(
@builtin(global_invocation_id) global_id : vec3 <u32>
)
{
    let thread_id: u32 = global_id.x;
    let filter_offset: i32 = dimensions.filter_size / 2;
    
    // for loop to iterate over the filter
    for (var i: i32 = 0; i < dimensions.filter_size; i = i + 1) {
        let offset: i32 = i32(thread_id ) + i - filter_offset;
        if (-1 < offset && offset < i32(dimensions.element_count)) {
            output[thread_id] = output[thread_id] + input_signal[offset] * filter_signal[i];
        }
    }

}