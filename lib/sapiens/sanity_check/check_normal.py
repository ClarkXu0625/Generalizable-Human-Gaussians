import numpy as np, matplotlib.pyplot as plt

normal = np.load('/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/lib/sapiens/output/train/normal/sapiens_1b/0004_000/0.npy')           # (1024,1024,3)
print(np.linalg.norm(normal, axis=-1).mean())  # ≈1.0

# visualize per channel
plt.figure(figsize=(12,4))
for i, name in enumerate(['nx (+X→right)', 'ny (+Y→down)', 'nz (+Z→camera)']):
    plt.subplot(1,3,i+1)
    plt.imshow((normal[...,i]+1)/2)  # map [-1,1] → [0,1]
    plt.title(name)
    plt.axis('off')
plt.tight_layout(); plt.show()
