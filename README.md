Vishwajeet Waghmare- 241080076
Yash Wadhwa- 241080078




# Parallel Edge Detection with OpenMP (V0-V6)

Files
  edge.cpp        main project code (about 300 lines, all 7 versions, tests and tables)
  edge_full.cpp   same code + extras (about 380 lines): correctness check on all 6 photos,
                  edge pictures (edges_1.png ...), results.csv for plots, V5/V6 in the
                  thread-scaling test, and a loop-schedule comparison
  stb_image.h / stb_image_write.h   image load / save libraries (no install needed)
  photos/         6 real photos from the BSDS500 test set (timing uses 100007.jpg)

Build and run (open a terminal in this folder)
  g++ -O3 -march=native -fno-math-errno -fopenmp edge.cpp -o edge.exe
  .\edge.exe quick        (1080p + 4K)        .\edge.exe   (also 8K, needs about 1 GB RAM)

  g++ -O3 -march=native -fno-math-errno -fopenmp edge_full.cpp -o edge_full.exe
  .\edge_full.exe quick
(On Linux / Colab use ./edge and ./edge_full, and add nothing else.)

Optional: pin threads to cores
  Linux/Colab: OMP_PROC_BIND=true ./edge       PowerShell: $env:OMP_PROC_BIND="true"; .\edge.exe

Notes
  - Speedup is measured against the faster of V0 and V1 (best sequential time).
  - "output OK" means the edge image is identical to V0's.
  - 1080p/4K/8K inputs are made by enlarging a photo (timing does not depend on image content).
  - Run on mains power with other apps closed. Colab has only about 2 cores.
  - Photos: BSDS500 (Arbelaez et al., 2011).


Compile: 
g++ -O3 -march=native -fno-math-errno -fopenmp edge_full.cpp -o edge_full.exe

Run: 
./edge_full.exe
