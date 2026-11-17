/*
  Parallel Edge Detection with OpenMP - Parallel Computing Mini Project, VJTI
  Steps: grayscale -> 5x5 Gaussian blur -> Sobel gradient -> threshold

  V0 scalar (1 thread)     V1 compiler auto-vectorization (1 thread)    V2 OpenMP threads
  V3 threads + omp simd    V4 separable blur (10 multiplies, not 25)    V5 all steps fused
  V6 fused with small tiles (data stays in cache)

  Build: g++ -O3 -march=native -fno-math-errno -fopenmp edge.cpp -o edge
  Run  : ./edge quick   (1080p and 4K)      ./edge   (also 8K)
*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;

#define THRESHOLD 100.0f   // gradient above this = edge
#define WARMUP 2           // runs we ignore
#define RUNS 10            // runs we measure (we report the median)

// Only for V0 and V2: switch OFF auto-vectorization (GCC: whole function, Clang: the loop)
#ifdef __clang__
  #define NO_VEC_FUNC
  #define NO_VEC_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
  #define NO_VEC_FUNC __attribute__((optimize("no-tree-vectorize")))
  #define NO_VEC_LOOP
#endif

// Gaussian weights (sigma = 1.4). 2D weight of a pixel = G[ky] * G[kx]
const float G[5] = { 0.11020946f, 0.23691201f, 0.30575706f, 0.23691201f, 0.11020946f };

// Data shared by all versions (global, so functions need no arguments)
int W, H;                        // image width and height
vector<unsigned char> rgb;       // input photo
vector<float> gray, blur, tmp, mag;
vector<unsigned char> edge;      // final answer: 255 = edge, 0 = not an edge

// ---------- one-row functions (used by V3 to V6). "omp simd" = many pixels per instruction ----------
void gray_row(const unsigned char *in, float *out) {
  #pragma omp simd
  for (int x = 0; x < W; x++)
    out[x] = 0.299f * in[3*x] + 0.587f * in[3*x+1] + 0.114f * in[3*x+2];
}
void blur25_row(const float *in, float *out) {        // full 5x5 blur, "in" = start of this row
  #pragma omp simd
  for (int x = 2; x < W - 2; x++) {
    float sum = 0;
    for (int ky = 0; ky < 5; ky++)
      for (int kx = 0; kx < 5; kx++)
        sum += G[ky] * G[kx] * in[(ky - 2) * W + x + kx - 2];
    out[x] = sum;
  }
}
void hblur_row(const float *in, float *out) {         // blur left-right
  #pragma omp simd
  for (int x = 2; x < W - 2; x++)
    out[x] = G[0]*in[x-2] + G[1]*in[x-1] + G[2]*in[x] + G[3]*in[x+1] + G[4]*in[x+2];
}
void vblur_row(const float *in, float *out) {         // blur up-down, "in" = start of this row
  #pragma omp simd
  for (int x = 2; x < W - 2; x++)
    out[x] = G[0]*in[x-2*W] + G[1]*in[x-W] + G[2]*in[x] + G[3]*in[x+W] + G[4]*in[x+2*W];
}
void sobel_row(const float *in, float *out) {         // gradient size, "in" = start of this row
  const float *up = in - W, *dn = in + W;
  #pragma omp simd
  for (int x = 3; x < W - 3; x++) {
    float gx = (up[x+1] + 2*in[x+1] + dn[x+1]) - (up[x-1] + 2*in[x-1] + dn[x-1]);
    float gy = (dn[x-1] + 2*dn[x] + dn[x+1]) - (up[x-1] + 2*up[x] + up[x+1]);
    out[x] = sqrtf(gx * gx + gy * gy);
  }
}
void thresh_row(const float *in, unsigned char *out) {
  #pragma omp simd
  for (int x = 0; x < W; x++)
    out[x] = (in[x] >= THRESHOLD) ? 255 : 0;
}

// ---------- V0 and V2: simple loops, vectorization OFF (V0 = 1 thread, V2 = many threads) ----------
NO_VEC_FUNC
void scalar_version() {
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) {                       // 1. grayscale
    NO_VEC_LOOP
    for (int x = 0; x < W; x++) {
      int i = y * W + x;
      gray[i] = 0.299f * rgb[3*i] + 0.587f * rgb[3*i+1] + 0.114f * rgb[3*i+2];
    }
  }
  #pragma omp parallel for schedule(static)
  for (int y = 2; y < H - 2; y++) {                   // 2. blur
    NO_VEC_LOOP
    for (int x = 2; x < W - 2; x++) {
      float sum = 0;
      for (int ky = 0; ky < 5; ky++)
        for (int kx = 0; kx < 5; kx++)
          sum += G[ky] * G[kx] * gray[(y + ky - 2) * W + x + kx - 2];
      blur[y * W + x] = sum;
    }
  }
  #pragma omp parallel for schedule(static)
  for (int y = 3; y < H - 3; y++) {                   // 3. Sobel
    NO_VEC_LOOP
    for (int x = 3; x < W - 3; x++) {
      int i = y * W + x;
      float gx = (blur[i-W+1] + 2*blur[i+1] + blur[i+W+1]) - (blur[i-W-1] + 2*blur[i-1] + blur[i+W-1]);
      float gy = (blur[i+W-1] + 2*blur[i+W] + blur[i+W+1]) - (blur[i-W-1] + 2*blur[i-W] + blur[i-W+1]);
      mag[i] = sqrtf(gx * gx + gy * gy);
    }
  }
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) {                       // 4. threshold
    NO_VEC_LOOP
    for (int x = 0; x < W; x++)
      edge[y * W + x] = (mag[y * W + x] >= THRESHOLD) ? 255 : 0;
  }
}

// ---------- V1: the SAME loops, but now the compiler is allowed to vectorize them ----------
void autovec_version() {
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      int i = y * W + x;
      gray[i] = 0.299f * rgb[3*i] + 0.587f * rgb[3*i+1] + 0.114f * rgb[3*i+2];
    }
  #pragma omp parallel for schedule(static)
  for (int y = 2; y < H - 2; y++)
    for (int x = 2; x < W - 2; x++) {
      float sum = 0;
      for (int ky = 0; ky < 5; ky++)
        for (int kx = 0; kx < 5; kx++)
          sum += G[ky] * G[kx] * gray[(y + ky - 2) * W + x + kx - 2];
      blur[y * W + x] = sum;
    }
  #pragma omp parallel for schedule(static)
  for (int y = 3; y < H - 3; y++)
    for (int x = 3; x < W - 3; x++) {
      int i = y * W + x;
      float gx = (blur[i-W+1] + 2*blur[i+1] + blur[i+W+1]) - (blur[i-W-1] + 2*blur[i-1] + blur[i+W-1]);
      float gy = (blur[i+W-1] + 2*blur[i+W] + blur[i+W+1]) - (blur[i-W-1] + 2*blur[i-W] + blur[i-W+1]);
      mag[i] = sqrtf(gx * gx + gy * gy);
    }
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      edge[y * W + x] = (mag[y * W + x] >= THRESHOLD) ? 255 : 0;
}

// ---------- V3: threads share the ROWS, SIMD shares the PIXELS inside a row ----------
void simd_version() {
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) gray_row(&rgb[3*y*W], &gray[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 2; y < H - 2; y++) blur25_row(&gray[y*W], &blur[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 3; y < H - 3; y++) sobel_row(&blur[y*W], &mag[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) thresh_row(&mag[y*W], &edge[y*W]);
}

// ---------- V4: same as V3, but the blur is two 1D passes (5 + 5 multiplies, not 25) ----------
void separable_version() {
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) gray_row(&rgb[3*y*W], &gray[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) hblur_row(&gray[y*W], &tmp[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 2; y < H - 2; y++) vblur_row(&tmp[y*W], &blur[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 3; y < H - 3; y++) sobel_row(&blur[y*W], &mag[y*W]);
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) thresh_row(&mag[y*W], &edge[y*W]);
}

// ---------- V5 and V6: all steps fused ----------
// Work on a few rows (a "band") at a time. The in-between rows live in small buffers that
// stay in the CPU cache; only the final edge image is written to big memory.
// Output rows y0..y1-1 need gray rows y0-3..y1+2 and blurred rows y0-1..y1.
void fused_version(int band) {
  int num_bands = (H - 6 + band - 1) / band;
  #pragma omp parallel
  {
    vector<float> g((band + 6) * W), h((band + 6) * W), bl((band + 2) * W), m(W);   // per-thread buffers
    #pragma omp for schedule(static)
    for (int k = 0; k < num_bands; k++) {
      int y0 = 3 + k * band;
      int y1 = min(y0 + band, H - 3);
      int n = y1 - y0;                                        // rows in this band
      for (int r = 0; r < n + 6; r++) {                       // gray, then blur left-right
        gray_row(&rgb[3 * (y0 - 3 + r) * W], &g[r*W]);
        hblur_row(&g[r*W], &h[r*W]);
      }
      for (int r = 0; r < n + 2; r++) vblur_row(&h[(r+2)*W], &bl[r*W]);   // blur up-down
      for (int r = 0; r < n; r++) {                           // Sobel + threshold
        sobel_row(&bl[(r+1)*W], &m[0]);
        thresh_row(&m[0], &edge[(y0 + r) * W]);
      }
    }
  }
}
void fused_big()   { fused_version(64); }   // V5: bands of 64 rows
void fused_tiled() { fused_version(8); }    // V6: small tiles of 8 rows (fit in cache)

// ---------- table of versions (threads: 1 = one thread, 0 = all threads) ----------
struct Version { const char *name; void (*run)(); int threads; };
Version versions[7] = {
  { "V0 scalar",       scalar_version,    1 },
  { "V1 auto-vec",     autovec_version,   1 },
  { "V2 threads",      scalar_version,    0 },
  { "V3 threads+simd", simd_version,      0 },
  { "V4 separable",    separable_version, 0 },
  { "V5 fused",        fused_big,         0 },
  { "V6 tiled",        fused_tiled,       0 }
};
int max_threads;

// Make a W x H test image from the photo by repeating pixels, and allocate the buffers
void set_size(const unsigned char *photo, int pw, int ph, int w, int h) {
  W = w; H = h;
  rgb.resize(3 * W * H);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      memcpy(&rgb[3 * (y * W + x)], photo + 3 * ((y * ph / H) * pw + x * pw / W), 3);
  for (vector<float> *v : { &gray, &blur, &tmp, &mag }) v->assign(W * H, 0);
  edge.assign(W * H, 0);
}

// Run a version WARMUP + RUNS times and return the median time in ms
double time_it(const Version &v, int threads) {
  double t[RUNS];
  omp_set_num_threads(threads);
  for (int i = 0; i < WARMUP; i++) v.run();
  for (int i = 0; i < RUNS; i++) {
    auto start = chrono::steady_clock::now();      // high-resolution timer
    v.run();
    auto end = chrono::steady_clock::now();
    t[i] = chrono::duration<double, milli>(end - start).count();   // time in ms
  }
  sort(t, t + RUNS);
  return t[RUNS / 2];
}

void print_row(const char *name, int threads, double ms, double base_ms) {
  printf("  %-16s %2d thr %9.2f ms %8.1f MP/s  speedup %5.2fx", name, threads, ms, W * H / 1e3 / ms, base_ms / ms);
}

// ---------- experiments ----------
void all_versions(const char *label) {                  // E1 + E3
  printf("\n=== All versions at %s (%d x %d) ===\n", label, W, H);
  double t[7];
  bool ok[7];
  vector<unsigned char> ref;
  for (int v = 0; v < 7; v++) {
    t[v] = time_it(versions[v], versions[v].threads ? 1 : max_threads);
    if (v == 0) ref = edge;                             // V0 gives the correct answer
    ok[v] = (edge == ref);                              // same as V0?
  }
  double best_seq = min(t[0], t[1]);                    // fastest 1-thread version
  for (int v = 0; v < 7; v++) {
    print_row(versions[v].name, versions[v].threads ? 1 : max_threads, t[v], best_seq);
    printf("  %s\n", ok[v] ? "output OK" : "OUTPUT DIFFERS!");
  }
}

void scaling() {                                         // E2: more and more threads
  printf("\n=== Thread scaling at %d x %d ===\n", W, H);
  double best_seq = time_it(versions[1], 1);
  for (int k = 2; k <= 3; k++) {                         // V2 (no SIMD) and V3 (SIMD)
    double t1 = time_it(versions[k], 1);
    for (int th = 1; th <= max_threads; th *= 2) {
      double ms = (th == 1) ? t1 : time_it(versions[k], th);
      print_row(versions[k].name, th, ms, best_seq);
      printf("  efficiency %5.1f%%\n", 100 * t1 / (th * ms));
    }
  }
}

int main(int argc, char **argv) {
  bool quick = (argc > 1 && strcmp(argv[1], "quick") == 0);
  max_threads = omp_get_max_threads();
  printf("Threads: %d | time = median of %d runs\n", max_threads, RUNS);

  int pw, ph, ch;
  unsigned char *photo = stbi_load("photos/100007.jpg", &pw, &ph, &ch, 3);   // 3 = RGB
  if (!photo) { printf("Cannot open photos/100007.jpg\n"); return 1; }

  set_size(photo, pw, ph, 1920, 1080); all_versions("1080p");
  set_size(photo, pw, ph, 3840, 2160); all_versions("4K");
  if (!quick) { set_size(photo, pw, ph, 7680, 4320); all_versions("8K"); }
  scaling();                                              // uses the biggest size made so far

  printf("\n=== Image size sweep: V1 (1 thread) vs V6 (all threads) ===\n");   // E4
  for (int s = 256; s <= 4096; s *= 2) {
    set_size(photo, pw, ph, s, s);
    double t1 = time_it(versions[1], 1);
    print_row("V1 auto-vec", 1, t1, t1);                                     printf("\n");
    print_row("V6 tiled", max_threads, time_it(versions[6], max_threads), t1); printf("\n");
  }
  stbi_image_free(photo);
}
