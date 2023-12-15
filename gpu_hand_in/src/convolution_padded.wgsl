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

var<workgroup> shared_signal : array<f32, 100>;
var<workgroup> shared_filter : array<f32, 100>;

@compute @workgroup_size(32, 1, 1)

fn conv_padded(
@builtin(global_invocation_id) global_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>,
@builtin(workgroup_id) group_id : vec3 <u32>,
)
{

    let thread_id: u32 = global_id.x;
    let filter_offset: i32 = dimensions.filter_size / 2;
    let tot_num_elemen: i32 = 32 + filter_offset * 2;
    
    let offset: i32 = i32(local_id.x) + 32;
    let global_offset:i32 = 32 * i32(group_id.x) + offset;

    shared_signal[local_id.x] = input_signal[thread_id];

    if (offset < tot_num_elemen) {
        shared_signal[offset] = input_signal[global_offset];
    }

    if (i32(local_id.x) < dimensions.filter_size) {
        shared_filter[i32(local_id.x)] = filter_signal[i32(local_id.x)];
    }

    workgroupBarrier();
    
    var sum: f32 = 0.0;
    for (var i: i32 = 0; i < dimensions.filter_size; i = i + 1) {
        sum += shared_signal[i32(local_id.x) + i] * shared_filter[i];
    }

    output[thread_id] = sum;
}
    