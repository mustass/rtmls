# GPU handin

## 1D Convolution

Status: All the versions work in their current state. 



However, there are things left to do. 

### To do:

- Run timing: For shared memory> signal in shared memory vs. filter in shared memory vs. both
- Shared memory conv --> block size to different multiples of 32
- Padded memory conv --> pad with 0s to ensure that the signal is always a multiple of my block size

### Learnings:

- Allocating a shared array of size x and adding elements to indices x+n does not give an error. 

## Matrix multiplication
Status: All the versions work in their current state. 

### To do:

- Run timing: naive vs. tiled vs. padded

### Learnings and Q's:

It is more difficult if one is launching different amount of blocks in each axis. Is this in the scope of the handin to handle or can we just launch max*max?