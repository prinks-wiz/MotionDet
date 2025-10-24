from datasets import load_dataset
import matplotlib.pyplot as plt

# Load ONLY the ball_roll split
dataset = load_dataset("mspitzna/physicsgen", name="ball_roll", trust_remote_code=True)

# Number of samples to visualize
N = 5

# Visualize the first N samples from the training split
for idx in range(N):
    sample = dataset["train"][idx]
    input_img = sample["osm"]
    target_img = sample["soundmap_512"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_img)
    ax1.set_title(f"Input OSM (ball_roll #{idx+1})")
    ax2.imshow(target_img)
    ax2.set_title(f"Target Soundmap (ball_roll #{idx+1})")
    plt.suptitle("ball_roll Sample Visualization")
    plt.show()
