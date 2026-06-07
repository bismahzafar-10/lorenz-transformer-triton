# Physics-Informed Causal Transformer with Custom Triton Kernel Fusion

A high-performance, hardware-accelerated **Physics-Informed Neural Network (PINN) Transformer** designed to predict non-linear trajectories within the chaotic Lorenz Attractor system. This framework leverages multi-objective optimization to minimize empirical tracking errors under explicit physical constraints, utilizing low-level custom **Triton kernels** to optimize inference execution speed on GPU hardware.

## 🚀 Core Engineering Highlights
* **Custom Low-Level Kernel Fusion:** Engineered a vectorized, hardware-fused GELU activation function using **Triton**. By managing memory layout paths directly on the GPU execution grid, this approach reduces global VRAM memory round-trips and improves arithmetic intensity.
* **Pareto Frontier Optimization:** Balanced structural representation and hardware constraints by expanding channel width (`d_model=256`) and keeping layer depth shallow (`num_layers=2`), ensuring maximum parallel performance across GPU Streaming Multiprocessors.
* **Physics-Informed Architecture:** Embedded continuous differential equation local constraints natively inside the loss loop with a linear dynamic weight-annealing schedule, preventing model drift down to an elite validation **MSE of 0.028**.
* **Tensor Core Acceleration:** Integrated native FP16 Automatic Mixed Precision (`torch.amp`) benchmarking to achieve an optimal microsecond inference footprint of **~2.01 ms**.

## 📊 Evaluation & Verification Results

### 1. 3D Chaotic Trajectory Invariant Mapping
In chaotic manifolds, minor evaluation variances trigger cascading prediction failures over long sequences. Driven by physics-informed local constraints, our Transformer closely matches the global geometric invariants of the dual-lobe butterfly manifold computed by fourth-order scientific solvers:

<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/43f45b19-a1d6-481f-8f1f-37a3c6704d30" />

<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/0a9606c7-0548-448a-8fcb-180624f41005" />


### 2. Operational Benchmark Progression
Our development progression systematically cleared critical latency boundaries while scaling up model parameter budgets:

| Workload Layout Variant | Execution Topology Profile | System Inference Latency | Validation Tracking MSE | Primary Bottleneck Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **01. Global Pooling Base** | 2-Layer / Full Sequence Mean | 11.66 ms (Host Bound) | `0.9404` (Unbounded Drift) | Framework launch overhead. |
| **02. CUDA Mini-Batching** | 2-Layer / Sequence Mean | 1.00 ms (Compute Safe) | `0.0423` (Precision Plateau) | Eradicated host-device memory thrashing. |
| **03. Causal Edge Slicing** | 2-Layer / Token Context Slice | 0.97 ms (Compute Safe) | `0.0366` (High Precision) | Removed lossy temporal average pooling. |
| **04. Capacity Scaling** | 4-Layer / Token Context Slice | 2.63 ms (Layer Bound) | `0.0292` (High Precision) | Sequential operator depth exceeded latency limits. |
| **05. Triton Fused (Final)** | **2-Layer Wide / Triton Gelu** | **2.01 ms (Tensor Core Optimal)** | **`0.0288` (Elite Tracking)** | **Fused activation kernels via Triton.** |

---

## 🛠️ Installation & Reproduction Lifecycle

### Prerequisites
Ensure your host machine features an NVIDIA GPU with functional CUDA drivers and support for the Triton runtime environment.

### 1. Clone and Install Dependencies
```bash
git clone [https://github.com/bismahzafar-10/lorenz-transformer-triton.git](https://github.com/bismahzafar-10/lorenz-transformer-triton.git)
cd lorenz-transformer-triton
pip install -r requirements.txt
```



### 2. Execute Training and Hardware Benchmarking
Run the main pipeline loop to simulate data, optimize the model layout, execute the custom Triton compilation grid, and export the tracking plots:

```bash
python main.py  
```


---

### 📤 Update your repository:
Once you update the text in VS Code, run these quick commands in your terminal to sync the clean version straight up to your live link:

```bash
git add README.md
git commit -m "docs: finalize showcase layout and remove setup guide"
git push origin main
