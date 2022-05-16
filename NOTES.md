# PyTorch

-   Install from [this guide](https://pytorch.org/tutorials/). To get [CUDA](https://developer.nvidia.com/cuda-zone) drivers pre-bundled.
-   Windows & Linux:
    -   https://pytorch.org/get-started/locally/
    -   `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
    -   check successful installation
        ```
        import torch
        x = torch.rand(5, 3)
        print(x) # should yield:    tensor([...])
        torch.cuda.is_available()
        torch.cuda.get_device_properties(torch.cuda.current_device())
        ```

# JAX

-   Only runs on Linux (may run on WSL, DOES NOT WORK WITH WIN 10, ONLY 11)
-   Install from [this guide](https://github.com/google/jax#pip-installation-gpu-cuda).
<!--
-   WSL (Ubuntu 20.04):

        -   install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) (command line instruction here)
            -   ```
                wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
                sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
                wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-wsl-ubuntu-11-6-local_11.6.2-1_amd64.deb
                sudo dpkg -i cuda-repo-wsl-ubuntu-11-6-local_11.6.2-1_amd64.deb
                sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-6-local/7fa2af80.pub
                sudo apt-get update
                sudo apt-get -y install cuda
                ```
            -   cuda version check & system variables:
                ```
                export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
                export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
                nvcc --version
                ```
        -   install [CuDNN](https://developer.nvidia.com/CUDNN):
            -   Nvidia Dev account and survey
            -   download
            -   ```
                sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb
                ```
        -   link GPU functionality
            -   `sudo ln -s /path/to/cuda /usr/local/cuda-11.6` (Was already linked for me)
        -   install jax
            ```
            pip install --upgrade pip
            pip install --upgrade --force-reinstall --no-cache  "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
            ```

    -->

-   Linux (PopOs22.04):

    -   install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) (command line instruction here)
        -   ```
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
            sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
            sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo apt-get update
            sudo apt-get -y install cuda
            ```
        -   cuda version check & system variables:
            ```
            export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
            export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
            nvcc --version
            nvidia-smi
            ```
    -   install [CuDNN](https://developer.nvidia.com/CUDNN):
        -   download [this](https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb)
        -   ```
            sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb
            ```
    -   link GPU functionality
        -   `sudo ln -s /path/to/cuda /usr/local/cuda-11.7` (Was already linked for me)
    -   install jax
        ```
        pip install --upgrade pip
        pip install --upgrade --force-reinstall --no-cache  "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
        ```

-   Check:

    -   check SAMPLES:
        -   download: `git clone https://github.com/NVIDIA/cuda-samples.git`
        -   `cd cuda-samples/Samples/0_Introduction/matrixMul`
        -   `make`
        -   `./matrixMul`
    -   check successful installation

        ```
        import jax
        from jax import grad
        import jax.numpy as jnp

        def tanh(x):  # Define a function
            y = jnp.exp(-2.0 * x)
            return (1.0 - y) / (1.0 + y)

        grad_tanh = grad(tanh)  # Obtain its gradient function
        print(grad_tanh(1.0))   # Evaluate it at x = 1.0
        print(jax.devices())
        ```

# Datasets

-   [Image test datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
-   [Pre-Trained vision Transformer](https://github.com/facebookresearch/dino)
