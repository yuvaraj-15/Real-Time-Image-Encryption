# Secure Image Encryption & Tamper Detection

This project addresses the secure, real-time transmission of sensitive image data in domains like military surveillance, telemedicine, and cloud-based storage. It presents a lightweight image encryption algorithm that balances strong security with computational efficiency. The project includes two versions of the algorithm, with the improved version showcasing advanced features like block-wise encryption, watermarking, and network transmission simulation.

---
## Features

This section highlights the key functionalities of your application, particularly the improved Version 2.

* **Block-wise Encryption:** The algorithm divides images into $4 \times 4$ blocks and uses a unique, position-based seed for each block. This approach enhances security by localizing encryption and increasing randomness.
* **Permutation-based Scrambling:** Instead of geometric transforms, this method uses a pseudo-random permutation to scramble pixels within each $4 \times 4$ block. This avoids distortion and helps preserve image quality.
* **Digital Watermarking:** A binary watermark is embedded in the encrypted image. This watermark can be extracted later to verify data authenticity and integrity.
* **Tamper Detection:** The system can detect unauthorized modifications by comparing the hash of the recovered image to the original. It adds a layer of robustness by combining hash checks with watermark verification.
* **Network Transmission Simulation:** This feature uses the Edmonds-Karp algorithm to simulate secure image transmission over a network graph. It ensures that only untampered images are permitted to pass, maintaining end-to-end integrity.
* **Comprehensive Security Analysis:** The system is analyzed using a suite of metrics including **PSNR**, **MSE**, **SSIM**, and **Entropy** to evaluate encryption strength and image reconstruction quality.

---
## Comparison: Version 1 vs. Version 2

The initial version used a global Logistic Map for XOR encryption and Arnold's Cat Map for scrambling. While fast, it was more prone to visual degradation and had lower security metrics due to global operations.

The **improved version** enhances both security and reconstruction quality by:
* Using **block-wise encryption**, which localizes the encryption process and increases randomness, resulting in higher entropy values.
* Implementing **permutation-based scrambling**, which avoids the distortion caused by geometric transforms and leads to a higher PSNR and SSIM.
* Adding **watermark verification**, providing an extra layer of integrity validation beyond simple hash checking.
* Including a **network simulation**, which models secure transmission over a network and ensures only untampered images are permitted to pass, demonstrating a real-world application.

---
## Getting Started

### Prerequisites
Make sure you have Python 3.x installed on your system. You'll also need the following libraries:
* `opencv-python`
* `numpy`
* `scikit-image`
* `matplotlib`
* `networkx`
* `Pillow`

You can install all required libraries using pip:
`pip install opencv-python numpy scikit-image matplotlib networkx Pillow`

### How to Run
1.  Download or clone the repository to your local machine.
2.  Navigate to the project directory in your terminal or command prompt.
3.  Run the main application using the following commands:
    `python finalUnimproved.py` and `python finalImproved.py` 
4.  The GUI will launch, allowing you to load an image and use the various encryption and analysis features.
