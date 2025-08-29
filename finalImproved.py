import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import hashlib
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import networkx as nx
import time


BLOCK_SIZE = 4
MASTER_SEED = "secure_key"
MSE_THRESHOLD = 0.1
SSIM_THRESHOLD = 1.1
ENTROPY_DIFF_THRESHOLD = 0.05
EXPECTED_WATERMARK = "AUTH"


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


def pad_image(img, block_size):
    h, w = img.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT), (h, w)

def depad_image(img, original_shape):
    h, w = original_shape
    return img[:h, :w]

def logistic_map_sequence(length, seed=0.5, r=3.999):
    x = seed
    seq = []
    for _ in range(length):
        x = r * x * (1 - x)
        seq.append(int(x * 256) % 256)
    return seq

def hash_seed(master_seed, i, j):
    combined = f"{master_seed}_{i}_{j}"
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    int_seed = int(hashed[:8], 16) / 0xFFFFFFFF
    return max(0.0001, min(0.9999, int_seed))

def generate_permutation(h, w, seed="scramble_key"):
    np.random.seed(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (10**8))
    indices = [(i, j) for i in range(h) for j in range(w)]
    permuted = indices.copy()
    np.random.shuffle(permuted)
    forward_map = dict(zip(indices, permuted))
    reverse_map = dict(zip(permuted, indices))
    return forward_map, reverse_map

def scramble_image(img, seed="scramble_key"):
    h, w = img.shape[:2]
    scrambled = np.zeros_like(img)
    fwd_map, rev_map = generate_permutation(h, w, seed)
    for (i, j), (x, y) in fwd_map.items():
        scrambled[x, y] = img[i, j]
    return scrambled, rev_map

def unscramble_image(img, rev_map):
    h, w = img.shape[:2]
    unscrambled = np.zeros_like(img)
    for (x, y), (i, j) in rev_map.items():
        unscrambled[i, j] = img[x, y]
    return unscrambled

def blockwise_encrypt(img, block_size=4, master_seed="secret123", r=3.999):
    padded_img, original_shape = pad_image(img, block_size)
    h, w = padded_img.shape[:2]
    encrypted = np.copy(padded_img)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            seed = hash_seed(master_seed, i, j)
            key = logistic_map_sequence(block_size * block_size * 3, seed=seed, r=r)
            key = np.array(key, dtype=np.uint8).reshape((block_size, block_size, 3))
            block = padded_img[i:i+block_size, j:j+block_size]
            encrypted[i:i+block_size, j:j+block_size] = cv2.bitwise_xor(block, key)
    return encrypted, original_shape

def blockwise_decrypt(encrypted_img, original_shape, block_size=4, master_seed="secret123", r=3.999):
    h, w = encrypted_img.shape[:2]
    decrypted = np.copy(encrypted_img)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            seed = hash_seed(master_seed, i, j)
            key = logistic_map_sequence(block_size * block_size * 3, seed=seed, r=r)
            key = np.array(key, dtype=np.uint8).reshape((block_size, block_size, 3))
            block = encrypted_img[i:i+block_size, j:j+block_size]
            decrypted[i:i+block_size, j:j+block_size] = cv2.bitwise_xor(block, key)

    return depad_image(decrypted, original_shape)

def embed_watermark(img, text=EXPECTED_WATERMARK):
    img = img.copy()
    for i, char in enumerate(text):
        img[i, i, 0] = ord(char)
    return img

def extract_watermark(img):
    try:
        chars = [chr(img[i, i, 0]) for i in range(len(EXPECTED_WATERMARK))]
        return ''.join(chars)
    except:
        return ""

def simulate_tampering(img, method="noise"):
    tampered = img.copy()
    h, w = img.shape[:2]
    if method == "noise":
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        tampered = cv2.add(tampered, noise)
    elif method == "flip":
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                tampered[i, j] = 255 - tampered[i, j]
    elif method == "block":
        tampered[h//8:h//2, w//8:w//2] = 0
    return tampered


class ImageEncryptorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Image Encryptor & Tamper Detector")
        self.img_original = None
        self.img_encrypted = None
        self.img_decrypted = None
        self.shape = None
        self.flag = False
        self.scrambled_img = None
        self.scramble_map = None

        tk.Button(root, text="Load Image", command=self.load_image).pack(pady=4)
        tk.Button(root, text="Scramble", command=self.scramble).pack(pady=4)
        tk.Button(root, text="Encrypt + Watermark", command=self.encrypt_image).pack(pady=4)
        tk.Button(root, text="Simulate Tampering", command=self.simulate_tamper).pack(pady=4)
        tk.Button(root, text="Decrypt Image", command=self.decrypt_image).pack(pady=4)
        tk.Button(root, text="Tamper Detection", command=self.detect_tampering).pack(pady=4)
        tk.Button(root, text="Vizualise Network Graph", command=self.visualize_network_graph).pack(pady=4)
        tk.Button(root, text="Simulate Network Transmission", command=self.simulate_network_transmission).pack(pady=4)
        tk.Button(root, text="Security Analysis", command=self.security_analysis).pack(pady=4)

        self.canvas = tk.Label(root)
        self.canvas.pack()

    def display_image(self, img):
        im = Image.fromarray(img).resize((300, 300))
        imgtk = ImageTk.PhotoImage(image=im)
        self.canvas.configure(image=imgtk)
        self.canvas.image = imgtk

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            bgr = cv2.imread(path)
            self.img_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.display_image(self.img_original)

    def scramble(self):
        self.flag = True
        start = time.time()
        scrambled, map = scramble_image(self.img_original, seed="scramble_key")
        scramble_time = time.time() - start
        print(f"Time for scrambling image: {scramble_time} seconds")
        self.scrambled_img = scrambled
        self.scramble_map = map
        self.display_image(self.scrambled_img)

    def encrypt_image(self):
        if self.img_original is not None:
            start_time = time.time()
            if self.flag is True:
                encrypted, shape = blockwise_encrypt(self.scrambled_img, block_size=4, master_seed="my_secure_key")
            else:
                encrypted, shape = blockwise_encrypt(self.img_original, block_size=4, master_seed="my_secure_key")
            encryption_time = time.time()-start_time
            print(f"Encryption time: {encryption_time} seconds")
            encrypted = embed_watermark(encrypted)
            self.img_encrypted = encrypted
            self.shape = shape
            cv2.imshow("Encrypted", cv2.cvtColor(self.img_encrypted, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("encrypted_output.png", self.img_encrypted)
            self.display_image(self.img_encrypted)

    def simulate_tamper(self):
        if self.img_encrypted is not None:
            method = np.random.choice(["noise", "block"])
            enc = simulate_tampering(self.img_encrypted, method)
            self.img_encrypted = enc
            cv2.imshow("Tampered Image", cv2.cvtColor(self.img_encrypted, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.display_image(self.img_encrypted)
            messagebox.showwarning("Tampering", f"Simulated Tampering: {method.upper()}")

    def decrypt_image(self):
        if self.img_encrypted is not None:
            start_time = time.time()
            self.img_decrypted = blockwise_decrypt(
                self.img_encrypted, self.shape, block_size=4, master_seed="my_secure_key"
            )
            decryption_time = time.time() - start_time
            print(f"Decryption time: {decryption_time} seconds")

            cv2.imshow("Decrypted (Pre-Unscramble)", cv2.cvtColor(self.img_decrypted, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if self.flag is True:
                print("Unscrambling...")
                self.img_decrypted = unscramble_image(self.img_decrypted, self.scramble_map)

                cv2.imshow("Decrypted + Unscrambled", cv2.cvtColor(self.img_decrypted, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cv2.imwrite("decrypted_output.png", cv2.cvtColor(self.img_decrypted, cv2.COLOR_RGB2BGR))
            self.display_image(self.img_decrypted)

    def detect_tampering(self):
        if self.img_original is None or self.img_decrypted is None:
            messagebox.showerror("Missing", "Load and decrypt an image first.")
            return

        gray_o = cv2.cvtColor(self.img_original, cv2.COLOR_RGB2GRAY)
        gray_d = cv2.cvtColor(self.img_decrypted, cv2.COLOR_RGB2GRAY)

        mse_val = mse(gray_o, gray_d)
        ssim_val = ssim(gray_o, gray_d)
        entropy_o = compute_entropy(self.img_original)
        entropy_d = compute_entropy(self.img_decrypted)

        hash_o = hashlib.sha256(self.img_original.tobytes()).hexdigest()
        hash_d = hashlib.sha256(self.img_decrypted.tobytes()).hexdigest()
        tampered_hash = hash_o != hash_d

        watermark = extract_watermark(self.img_encrypted)
        tampered_watermark = watermark != EXPECTED_WATERMARK

        tampered = (
            mse_val > MSE_THRESHOLD or
            # ssim_val < SSIM_THRESHOLD or
            abs(entropy_o - entropy_d) > ENTROPY_DIFF_THRESHOLD or
            # tampered_hash or
            tampered_watermark
        )

        if tampered:
            msg = f"[ALERT] Tampering Detected!\nMSE: {mse_val:.3f}, SSIM: {ssim_val:.3f}\nWatermark: {watermark}"
            messagebox.showerror("Tamper Detected", msg)
        else:
            messagebox.showinfo("Safe", "Image verified: No tampering detected.")

        diff = cv2.absdiff(gray_o, gray_d)
        plt.imshow(diff, cmap='hot')
        plt.title("Difference Heatmap")
        plt.axis('off')
        plt.show()

    def visualize_network_graph(self):
        G = nx.DiGraph()
        edges = [
            ('Source', 'Node1', 500),
            ('Source', 'Node2', 300),
            ('Node1', 'Destination', 400),
            ('Node2', 'Destination', 200),
            ('Node1', 'Node2', 100)
        ]
        for u, v, cap in edges:
            G.add_edge(u, v, capacity=cap)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
        edge_labels = {(u, v): f"{d['capacity']} KBps" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Simulated Transmission Network")
        plt.show()

    def simulate_network_transmission(self):
        img_size_kb = self.img_encrypted.nbytes / 1024
        G = nx.DiGraph()
        G.add_edge('Source', 'Node1', capacity=500)       # KBps
        G.add_edge('Source', 'Node2', capacity=300)
        G.add_edge('Node1', 'Destination', capacity=400)
        G.add_edge('Node2', 'Destination', capacity=200)
        G.add_edge('Node1', 'Node2', capacity=100)

        flow_value, flow_dict = nx.maximum_flow(G, 'Source', 'Destination')

        print(f"[Network Simulation]")
        print(f"Encrypted Image Size: {img_size_kb:.2f} KB")
        print(f"Max Network Capacity: {flow_value} KBps")

        if img_size_kb <= flow_value:
            print("→ Image can be securely transmitted in 1 cycle.\n")
        else:
            cycles = int(np.ceil(img_size_kb / flow_value))
            print(f"→ Image must be split or compressed.")
            print(f"Estimated transmission cycles needed: {cycles}\n")

    def security_analysis(self):
        gray_original = cv2.cvtColor(self.img_original, cv2.COLOR_RGB2GRAY)
        gray_decrypted = cv2.cvtColor(self.img_decrypted, cv2.COLOR_RGB2GRAY)
        
        mse_val = mse(self.img_original, self.img_decrypted)
        psnr_val = psnr(self.img_original, self.img_decrypted)
        ssim_val = ssim(gray_original, gray_decrypted)
        entropy_original = compute_entropy(self.img_original)
        entropy_encrypted = compute_entropy(self.img_encrypted)
        entropy_decrypted = compute_entropy(self.img_decrypted)

        print(f"\n[Image Similarity Metrics]")
        print(f"MSE: {mse_val:.4f}")   # lower is better. Zero means perfect match
        print(f"PSNR: {psnr_val:.2f} dB")   # above 30 is good quality
        print(f"SSIM: {ssim_val:.4f}")  # 0 - 1.1 means pefect structural integrity
        print(f"Entropy (Original): {entropy_original:.4f}")    # higher entropy in encrypted image = more randomness (more secure)
        print(f"Entropy (Encrypted): {entropy_encrypted:.4f}")  # original and decrypted should have similar
        print(f"Entropy (Decrypted): {entropy_decrypted:.4f}")  # encrypted should be more


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEncryptorGUI(root)
    root.mainloop()