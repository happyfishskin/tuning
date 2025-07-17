import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import csv
import argparse

# CUDA dummy kernel for synthetic workload
kernel_code = """
__global__ void dummyKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

mod = SourceModule(kernel_code)
dummy_kernel = mod.get_function("dummyKernel")

#每組 block/grid 配置會重複執行 5 次 kernel，並平均這些 GPU 純運算時間，以避免單次測量的偶然誤差，確保評估準確性。
def measure_time(N, block_size_x, block_size_y, grid_size_x, grid_size_y, repeat=5):
    total_threads = block_size_x * block_size_y * grid_size_x * grid_size_y
    N = min(N, total_threads)  # 保證不超出 alloc 空間

    output = np.zeros(N, dtype=np.float32)
    output_gpu = drv.mem_alloc(output.nbytes)

    block = (block_size_x, block_size_y, 1)
    grid = (grid_size_x, grid_size_y)

    start = drv.Event()
    end = drv.Event()

    start.record()
    for _ in range(repeat):
        dummy_kernel(output_gpu, np.int32(N), block=block, grid=grid)
    end.record()
    end.synchronize()

    elapsed_time = start.time_till(end) / repeat  # ms
    output_gpu.free()
    return elapsed_time

def run_tuning(N, output_csv="dummy_tuning.csv"):
    block_sizes = [32, 64, 128, 256]
    grid_sizes = [16, 32, 64, 128]

    results = []
    best_time = float("inf")
    best_config = None

    print("\n開始 dummy kernel tuning for N =", N)

    for block_x in block_sizes:
        for block_y in [1, 2, 4, 8, 16, 32]:
            if block_x * block_y > 1024:
                continue
            grid_x = min((N + block_x - 1) // block_x, 1024)
            grid_y = min((N + block_y - 1) // block_y, 1024)
            try:
                t = measure_time(N, block_x, block_y, grid_x, grid_y)
                results.append((block_x, block_y, grid_x, grid_y, t))
                print(f"Block=({block_x},{block_y}), Grid=({grid_x},{grid_y}) -> {t:.3f} ms")
                if t < best_time:
                    best_time = t
                    best_config = (block_x, block_y, grid_x, grid_y)
            except Exception as e:
                print(f"Block=({block_x},{block_y}) Failed: {e}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Block X", "Block Y", "Grid X", "Grid Y", "Time (ms)"])
        writer.writerows(results)

    print("\n最佳配置:")
    print(f"Block=({best_config[0]},{best_config[1]}), Grid=({best_config[2]},{best_config[3]}) -> {best_time:.3f} ms")
    print(f"結果已儲存到 {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256, help="總 thread 數 N")
    parser.add_argument("--output", type=str, default="dummy_tuning.csv", help="輸出 CSV 檔案名稱")
    args = parser.parse_args()
    run_tuning(args.size, output_csv=args.output)
