## GSRast: Gaussian Rasterizer

This repo rasterizes gaussians trained from the Gaussian Splatting paper: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

### Building

```bash
mkdir build
cd build
cmake .. -G Ninja
```

The project needs [GLFW3](https://www.glfw.org/) to build.

### Running

```bash
cd build
./apps/gsrast/gsrast[.exe]
```
