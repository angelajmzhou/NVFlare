# ExecuTorch Simulated/Real Cross-Edge Federated Learning

This guide details how to set up and run the ExecuTorch-based cross-edge federated learning example with support for real iOS devices and simulated clients.

## 1. Setup & Provisioning

If you are starting fresh or need to recreate the workspace, follow these steps.

### Prerequisites
Ensure NVFlare and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Optional: Build ExecuTorch with MPS Backend (for GPU Acceleration)

By default, ExecuTorch uses CPU-only execution. To enable GPU acceleration on macOS/iOS via Metal Performance Shaders (MPS):

```bash
cd /Users/angel/code/nvflare_exectorch_trial/executorch
CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh
```

**Requirements:**
- macOS ≥ 12.4 or iOS ≥ 15.4
- Xcode ≥ 14.1
- Build time: ~15-20 minutes

**What this enables:**
- GPU-accelerated model training on iPhone/iPad
- 3-4x faster training for convolutional networks
- Better performance during development on macOS

### Provision the Project
Generate the workspace using the project configuration. This creates the necessary certificates and startup scripts.

```bash
# If a previous workspace exists, remove it to avoid conflicts
rm -rf /tmp/nvflare/workspaces/edge_example

# Provision the workspace
cd /Users/angel/code/nvflare_exectorch_trial/NVFlare/examples/advanced/edge
nvflare provision -s -p project.yml
./post_provision.sh
```

This creates the workspace at `/tmp/nvflare/workspaces/edge_example/prod_00`.

## 2. Start the Infrastructure

You need to start the FL Server and the Routing Proxy (RP) to enable communication with edge devices.

### Step A: Start the FL Server & Aggregators
Term 1:
```bash
cd /tmp/nvflare/workspaces/edge_example/prod_00
./start_all.sh
```
*Wait until you see "Server started" and clients connecting.*

### Step B: Start the Routing Proxy
Term 2:
```bash
cd /tmp/nvflare/workspaces/edge_example/prod_00/scripts
./start_rp.sh
```
*Wait until you see "Routing Proxy started on port 4321".*

### Step C: Start Isolated Phone Environment (Alternative)
To prevent simulated clients from stealing the task from your real phone, use `start_phone.sh` instead of `start_all.sh`.

```bash
# Term 1 (Replace Step A)
conda activate executorch
cd /tmp/nvflare/workspaces/edge_example/prod_00
chmod +x start_phone.sh
./start_phone.sh
```
*This starts the Server, Proxy, and a single waiting client (C11) for your phone.*


## 3. iOS Client Setup

1.  Open `NVFlare/nvflare/edge/device/ios/ExampleProject/ExampleProject.xcodeproj` in Xcode.
2.  Navigate to `TrainerController.swift`.
3.  Ensure the `serverURL` matches your Routing Proxy address (likely your computer's IP):
    ```swift
    // Use HTTP if RP is not configured for SSL (default in this example)
    @Published var serverURL = "http://192.168.1.180:4321" 
    ```
4.  Build and Run the app on your iOS device.
5.  Watch the Xcode logs for `Successfully registered client`.

## 4. Submitting Jobs

You can run jobs with or without Hardware Acceleration (MPS - Metal Performance Shaders for GPU).

### Option A: With Hardware Acceleration (MPS) - **Recommended for Real Devices**

If you built ExecuTorch with MPS support (see Setup section above), GPU acceleration is **enabled by default**.

```bash
# Term 3
conda activate executorch
cd /Users/angel/code/nvflare_exectorch_trial/NVFlare/examples/advanced/edge/jobs
export NVFLARE_EDGE_ENABLE_MPS=1
python3 et_job.py
```

**What happens:**
- Model export includes MPS partitioning (adds ~5-10s overhead)
- Training uses iPhone/iPad GPU via Metal
- ~3-4x faster training per epoch compared to CPU

**To verify MPS is active:** Check worker logs for `"NVFlare: Partitioning implementation for MPS..."` and `"MPS Partitioning successful."`

### Option B: CPU-Only Mode (No GPU Acceleration)

To disable MPS and use pure CPU execution, set the environment variable before running:

```bash
# Term 3
conda activate executorch
cd /Users/angel/code/nvflare_exectorch_trial/NVFlare/examples/advanced/edge/jobs
export NVFLARE_EDGE_ENABLE_MPS=0
python3 et_job.py
```

**When to use CPU-only:**
- Troubleshooting MPS issues
- Testing on devices without GPU support
- **Fixing 'Error 32':** If the iOS app is missing `portable_kernels`, disabling MPS will fallback to **XNNPACK**, which is supported.
- Comparing CPU vs GPU performance

**IMPORTANT:**
To change this setting, you must **restart the NVFlare Server** (kill `start_phone.sh` and run it again) if the variable was exported in the same terminal, OR explicitly export it before running `start_phone.sh`.

**Performance comparison:**
| Mode | Model Export | Training (1 epoch) |
|------|-------------|-------------------|
| CPU-only (XNNPACK) | ~5-10s | ~10-20s (Fast!) |
| MPS (GPU) | ~10-15s | ~5-15s |

### Job Arguments

Both modes support the same arguments:

```bash
python3 et_job.py \
  --total_num_of_devices 1 \                    # Total clients expected
  --num_of_simulated_devices_on_each_leaf 0    # 0 for real phone only
```

## Troubleshooting

### General Issues

-   **Client Not Registered**: Ensure `start_all.sh` is running. The "Bridge Clients" (C1, C11, etc.) must be active to forward traffic.
-   **SSL Mismatch**: If the iOS app fails to connect, check if it's using `https` vs `http`. The Routing Proxy default is often `http` unless `ssl_cert` args are passed.
-   **Model Not Ready**: The first time a job runs, the server may take 30-60s to initialize the model. The client will retry automatically.

### MPS / Hardware Acceleration Issues

-   **"Backend MPSBackend is not registered"**: ExecuTorch was not built with MPS support. Rebuild using `CMAKE_ARGS="-DEXECUTORCH_BUILD_MPS=ON" ./install_executorch.sh`.
-   **MPS takes too long**: First-time MPS partitioning can take 10-15 seconds. Subsequent runs use cached partitions.
-   **Want to test without MPS**: Set `export NVFLARE_EDGE_ENABLE_MPS=0` before running `et_job.py`.
-   **Check if MPS is active**: Look for `"NVFlare: Partitioning implementation for MPS..."` in client worker logs at `/tmp/nvflare/workspaces/edge_example/prod_00/C*/*/log.txt`.
-   **iOS device doesn't support MPS**: Requires iOS ≥ 15.4. Older devices will fall back to CPU automatically.

### Performance Tips

- Use MPS for real device deployment (3-4x faster)
- Use CPU-only for quick debugging and testing
- Monitor training time differences to verify GPU acceleration is working
