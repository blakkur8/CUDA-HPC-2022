# CUDA-HPC-2022

1. compile the program
2. run the program ./final_cuda <number range> (eg. 256 or 4096)
3. profit

### Parameter

1. upper number range (eg. 256 or 4096), default value 256
2. (optional) -r: row output (only prints results). Good to use this when piping to file

### Example of extension:

```c
./cuda_brute_force 256
```

### Example of extension with -r:

```c
./cuda_brute_force 256 -r
```
