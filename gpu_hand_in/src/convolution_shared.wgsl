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

var<workgroup> shared_signal : array<f32,100>;
var<workgroup> shared_filter : array<f32,100>;

@compute @workgroup_size(32, 1, 1)

fn conv_shared(
@builtin(global_invocation_id) global_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>,
@builtin(workgroup_id) group_id : vec3 <u32>,
)
{

    let tid: i32 = i32(global_id.x);
    let lid: i32 = i32(local_id.x);
    let filter_offset: i32 = dimensions.filter_size / 2;
    
    if (lid == 0){
        for (var i: i32 = filter_offset; i > -1 ; i = i - 1) {
            if (tid - i < 0) {
                shared_signal[filter_offset-i] = 0.0;
            } else {
                shared_signal[filter_offset-i] = input_signal[tid - i];
            }
        }
    } else if (lid == 31) {
        for (var i: i32 = 0; i <= filter_offset ; i = i + 1) {
            if (tid + i > i32(dimensions.element_count)) {
                shared_signal[lid + filter_offset +i] = 0.0;
            } else {
                shared_signal[lid + filter_offset +i] = input_signal[tid + i];
            }
        }
    }else {
        shared_signal[filter_offset + lid] = input_signal[tid];
    } 

    if (lid < dimensions.filter_size){
        shared_filter[lid] = filter_signal[lid];
        }
    
    workgroupBarrier();
    
    
    
    var sum: f32 = 0.0;
        for (var i: i32 = 0; i < dimensions.filter_size; i = i + 1) {
            sum += shared_signal[lid+i] * shared_filter[i];
        }

        output[tid] = sum;

}
    