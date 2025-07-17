
CUDA Dummy Kernel Tuning

作者：613k0007c 余品誼

本專案透過 PyCUDA 實作一個模擬神經網路運算負載的 dummyKernel，並系統性測試不同的 CUDA block/grid 配置，找出最佳效能組合。此程式專為 auto-tuning 而設計，可用於分析並行計算結構對效能的影響。

---

核心特色

- 每組 block/grid 執行多次取平均（預設 5 次）
- 只計算純 kernel 執行時間（不含資料搬移）
- 結果輸出為 CSV 檔，含所有參數與時間
- 可延伸畫圖分析或結合真實模型 kernel

---

執行環境需求

請先確認以下環境已安裝：

- Python 3.6 以上（建議使用 Python 3.8）
- CUDA（CUDA Version: 12.2）
- NVIDIA 顯示卡（建議支援 compute capability 6.1 以上）
- PyCUDA 套件

安裝 PyCUDA：
建議使用 pip 安裝：

pip install pycuda
pip install numpy

---

Kernel 模擬說明

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

kernel 設計：模擬矩陣乘法計算（每個 thread 執行一個 N×N 的矩陣元素乘加）

---

執行方式

python3 dummy_kernel_tuning.py --size 8192 --output result.csv

可用參數：

參數         說明                                   預設值
--size       總 thread 數 N，決定 output 陣列大小    8192
--output     結果輸出 CSV 檔案名稱                  tuning_results.csv

---

範例輸出（CSV）

Block X,Block Y,Grid X,Grid Y,Time (ms)
64,4,16,250,0.01564
128,2,8,500,0.01484
...

---

專案結構

.
├── tuning_module.py    主程式
├── dummy_tuning.csv    範例輸出
├── README.md           本說明文件

---

未來可擴充方向

- 自動產生 Heatmap / 折線圖
- 支援多 GPU 環境的比較
- 整合 TensorRT / cuBLAS 對照分析
- 換用真實卷積 / matmul kernel 進行 tuning

---

影片連結
https://youtu.be/QOlcWtvoD4w

---
