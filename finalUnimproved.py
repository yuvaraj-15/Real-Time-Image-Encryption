import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import hashlib

def logistic_map_sequence(length, seed=0.5, r=3.999):
    x = seed
    sequence = []
    for _ in range(length):
        x = r * x * (1 - x)
        sequence.append(int(x * 256) % 256)
    return sequence

def arnold_cat_map(image, iterations):
    n, _ = image.shape[:2]
    result = np.copy(image)
    for _ in range(iterations):
        temp = np.copy(result)
        for x in range(n):
            for y in range(n):
                new_x = (x + y) % n
                new_y = (x + 2 * y) % n
                result[new_x, new_y] = temp[x, y]
    return result

def inverse_arnold_cat_map(image, iterations):
    n, _ = image.shape[:2]
    result = np.copy(image)
    for _ in range(iterations):
        temp = np.copy(result)
        for x in range(n):
            for y in range(n):
                new_x = (2 * x - y) % n
                new_y = (-x + y) % n
                result[new_x, new_y] = temp[x, y]
    return result

def encrypt_image(img_path, logistic_seed=0.6, logistic_r=3.99, arnold_iter=10):
    img = cv2.imread(img_path)
    original_shape = img.shape
    img = cv2.resize(img, (256, 256))
    key_stream = logistic_map_sequence(256*256*3, seed=logistic_seed, r=logistic_r)
    key_stream = np.array(key_stream).reshape((256, 256, 3)).astype(np.uint8)
    encrypted_img = cv2.bitwise_xor(img, key_stream)
    scrambled_img = np.zeros_like(encrypted_img)
    for c in range(3):
        scrambled_img[:, :, c] = arnold_cat_map(encrypted_img[:, :, c], arnold_iter)
    return scrambled_img, original_shape

def decrypt_image(encrypted_img, logistic_seed=0.6, logistic_r=3.99, arnold_iter=10):
    unscrambled_img = np.zeros_like(encrypted_img)
    for c in range(3):
        unscrambled_img[:, :, c] = inverse_arnold_cat_map(encrypted_img[:, :, c], arnold_iter)
    key_stream = logistic_map_sequence(256*256*3, seed=logistic_seed, r=logistic_r)
    key_stream = np.array(key_stream).reshape((256, 256, 3)).astype(np.uint8)
    decrypted_img = cv2.bitwise_xor(unscrambled_img, key_stream)
    return decrypted_img

def calculate_image_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    return float('inf') if m == 0 else 20 * np.log10(255.0 / np.sqrt(m))

def compute_entropy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    prob = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in prob if p > 0])



# Encrypt
path = "sampleNORMAL.jpg"
original_img = cv2.imread("sampleNORMAL.jpg")
encrypted, original_shape = encrypt_image(path)
decrypted = decrypt_image(encrypted, logistic_seed=0.6, logistic_r=3.99, arnold_iter=10)
decrypted_resized = cv2.resize(decrypted, (original_shape[1], original_shape[0]))

# Simulated secure transmission using hash verification
original_hash = calculate_image_hash(encrypted)
cv2.imwrite("encrypted_output.png", encrypted)

# Simulate receiver loading the file
received = cv2.imread("encrypted_output.png")
# # Simulate tampering by modifying a single pixel
# received[0, 0, 0] = np.uint8((int(received[0, 0, 0]) + 1) % 256)
received_hash = calculate_image_hash(received)

#Performance
gray_original = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
gray_decrypted = cv2.cvtColor(decrypted_resized, cv2.COLOR_RGB2GRAY)
mse_val = mse(original_img, decrypted_resized)
psnr_val = psnr(original_img, decrypted_resized)
ssim_val = ssim(gray_original, gray_decrypted)
entropy_original = compute_entropy(original_img)
entropy_encrypted = compute_entropy(encrypted)
entropy_decrypted = compute_entropy(decrypted_resized)

# Hash comparison
if original_hash == received_hash:
    print("Transmission Verified: No Tampering Detected.")
else:
    print("Alert! Transmission Tampered.")

#Metrics
print(f"\n[Image Similarity Metrics]")
print(f"MSE: {mse_val:.4f}")   # lower is better. Zero means perfect match
print(f"PSNR: {psnr_val:.2f} dB")   # above 30 is good quality
print(f"SSIM: {ssim_val:.4f}")  # 0 - 1.1 means pefect structural integrity
print(f"Entropy (Original): {entropy_original:.4f}")    # higher entropy in encrypted image = more randomness (more secure)
print(f"Entropy (Encrypted): {entropy_encrypted:.4f}")  # original and decrypted should have similar
print(f"Entropy (Decrypted): {entropy_decrypted:.4f}")  # encrypted should be more

# Save outputs
cv2.imwrite("decrypted_output.png", decrypted_resized)

# Show images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB))
plt.title("Encrypted Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(decrypted_resized, cv2.COLOR_BGR2RGB))
plt.title("Decrypted Image")
plt.axis("off")
plt.tight_layout()
plt.show()
