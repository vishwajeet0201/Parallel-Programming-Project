/*
  Parallel Edge Detection with OpenMP - Parallel Computing Mini Project, VJTI
  Steps: grayscale -> 5x5 Gaussian blur -> Sobel gradient -> threshold

  V0 scalar (1 thread)     V1 compiler auto-vectorization (1 thread)    V2 OpenMP threads
  V3 threads + omp simd    V4 separable blur (10 multiplies, not 25)    V5 all steps fused
  V6 fused with small tiles (data stays in cache)

  FULL VERSION = edge.cpp + extras: correctness check on all 6 photos, edge pictures (edges_N.png),
  results.csv for plots, V5/V6 in the thread-scaling test, and a loop-schedule comparison.

  Build: g++ -O3 -march=native -fno-math-errno -fopenmp edge_full.cpp -o edge_full
  Run  : ./edge_full quick   (1080p and 4K)      ./edge_full   (also 8K)
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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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
    #pragma omp for schedule(runtime)       // schedule is chosen in main() with omp_set_schedule
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

FILE *csv;                    // results.csv (for plots)
const char *exp_name = "";    // name of the running experiment, written in the csv

// Print one result line and save it in results.csv. eff = thread efficiency (-1 = not used)
void print_row(const char *name, int threads, double ms, double base_ms, double eff = -1) {
  double mp = W * H / 1e3 / ms;
  printf("  %-16s %2d thr %9.2f ms %8.1f MP/s  speedup %5.2fx", name, threads, ms, mp, base_ms / ms);
  if (eff >= 0) printf("  efficiency %5.1f%%", eff * 100);
  printf("\n");
  fprintf(csv, "%s,%s,%d,%d,%d,%.3f,%.2f,%.3f,%.3f\n", exp_name, name, W, H, threads, ms, mp, base_ms / ms, eff);
}

// ---------- experiments ----------

// Correctness: every version must give the same edges as V0, on every photo (original size)
void check_photos(const char **files, int n) {
  printf("\n=== Correctness: all versions vs V0 on each original photo ===\n");
  for (int f = 0; f < n; f++) {
    int pw, ph, ch;
    unsigned char *photo = stbi_load(files[f], &pw, &ph, &ch, 3);
    if (!photo) { printf("  cannot open %s\n", files[f]); continue; }
    set_size(photo, pw, ph, pw, ph);                    // original size, no enlarging
    omp_set_num_threads(1);
    versions[0].run();
    vector<unsigned char> ref = edge;                   // V0 = correct answer
    int bad = 0;
    for (int v = 1; v < 7; v++) {
      omp_set_num_threads(versions[v].threads ? 1 : max_threads);
      versions[v].run();
      if (edge != ref) bad++;
    }
    printf("  %-20s %dx%d  edge pixels %6d  %s\n", files[f], W, H,
           (int)count(ref.begin(), ref.end(), 255), bad == 0 ? "all versions OK" : "SOME VERSIONS DIFFER!");
    char name[50];
    snprintf(name, sizeof(name), "edges_%d.png", f + 1);
    stbi_write_png(name, W, H, 1, ref.data(), W);       // save the edge picture
    stbi_image_free(photo);
  }
  printf("  (edge pictures saved as edges_1.png, edges_2.png, ...)\n");
}

void all_versions(const char *label) {                  // E1 + E3
  printf("\n=== All versions at %s (%d x %d) ===\n", label, W, H);
  exp_name = "E1E3";
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
  exp_name = "E2";
  double best_seq = time_it(versions[1], 1);
  int which[4] = { 2, 3, 5, 6 };                         // V2, V3, V5, V6
  for (int k = 0; k < 4; k++) {
    Version &v = versions[which[k]];
    double t1 = time_it(v, 1);
    for (int th = 1; th <= max_threads; th *= 2) {
      double ms = (th == 1) ? t1 : time_it(v, th);
      print_row(v.name, th, ms, best_seq, t1 / (th * ms));   // efficiency = t1 / (threads * time)
    }
  }
}

void schedules() {                                       // V6 with different loop schedules
  printf("\n=== V6 with different schedules at %d x %d ===\n", W, H);
  exp_name = "SCHED";
  double best_seq = time_it(versions[1], 1);
  const char *names[3] = { "V6 static", "V6 dynamic", "V6 guided" };
  omp_sched_t kinds[3] = { omp_sched_static, omp_sched_dynamic, omp_sched_guided };
  for (int s = 0; s < 3; s++) {
    omp_set_schedule(kinds[s], 0);                       // 0 = default chunk size
    print_row(names[s], max_threads, time_it(versions[6], max_threads), best_seq);
  }
  omp_set_schedule(omp_sched_static, 0);                 // back to normal
}

void size_sweep(const unsigned char *photo, int pw, int ph) {   // E4
  printf("\n=== Image size sweep: V1 (1 thread) vs V6 (all threads) ===\n");
  exp_name = "E4";
  for (int s = 256; s <= 4096; s *= 2) {
    set_size(photo, pw, ph, s, s);
    double t1 = time_it(versions[1], 1);
    print_row("V1 auto-vec", 1, t1, t1);
    print_row("V6 tiled", max_threads, time_it(versions[6], max_threads), t1);
  }
}

int main(int argc, char **argv) {
  bool quick = (argc > 1 && strcmp(argv[1], "quick") == 0);
  max_threads = omp_get_max_threads();
  omp_set_schedule(omp_sched_static, 0);
  printf("Threads: %d | time = median of %d runs\n", max_threads, RUNS);

  csv = fopen("results.csv", "w");
  fprintf(csv, "experiment,version,width,height,threads,time_ms,mpix_per_s,speedup,efficiency\n");

  // the 6 test photos (BSDS500)
  const char *files[6] = { "photos/100007.jpg", "photos/103029.jpg", "photos/107045.jpg",
                           "photos/160067.jpg", "photos/188025.jpg", "photos/196088.jpg" };
  check_photos(files, 6);

  int pw, ph, ch;
  unsigned char *photo = stbi_load(files[0], &pw, &ph, &ch, 3);   // photo used for timing
  if (!photo) { printf("Cannot open %s\n", files[0]); return 1; }

  set_size(photo, pw, ph, 1920, 1080); all_versions("1080p");
  set_size(photo, pw, ph, 3840, 2160); all_versions("4K");
  if (!quick) { set_size(photo, pw, ph, 7680, 4320); all_versions("8K"); }
  scaling();                                              // uses the biggest size made so far
  set_size(photo, pw, ph, 3840, 2160); schedules();
  size_sweep(photo, pw, ph);

  stbi_image_free(photo);
  fclose(csv);
  printf("\nResults saved in results.csv\n");
}
