# GPU handin

## 1D Convolution

Status: All the versions work in their current state. 

Learnings:

- Allocating a shared array of size x and adding elements to indices x+n does not give an error. 

## Matrix multiplication
Status: All the versions work in their current state. 

### Learnings and Q's:

It is more difficult if one is launching different amount of blocks in each axis. Is this in the scope of the handin to handle or can we just launch max*max?