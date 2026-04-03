import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from torchvision import datasets

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.feature import hog, graycomatrix, graycoprops
except ImportError as exc:
    raise ImportError(
        "scikit-image is required for SSIM/HOG/GLCM. Install with: pip install scikit-image"
    ) from exc


RANDOM_STATE = 42
TSNE_SAMPLES = 2000

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def to_numpy_images(tensor_images):
    images = tensor_images.numpy().astype("float32") / 255.0
    return images


def load_mnist_sets():
    digit_train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    digit_test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)

    fashion_train = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    fashion_test = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)

    digit = {
        "train_images": to_numpy_images(digit_train.data),
        "train_labels": digit_train.targets.numpy(),
        "test_images": to_numpy_images(digit_test.data),
        "test_labels": digit_test.targets.numpy(),
        "num_classes": 10,
    }

    fashion = {
        "train_images": to_numpy_images(fashion_train.data),
        "train_labels": fashion_train.targets.numpy(),
        "test_images": to_numpy_images(fashion_test.data),
        "test_labels": fashion_test.targets.numpy(),
        "num_classes": 10,
    }

    return digit, fashion


def _pick_key(npz_obj, candidates):
    for key in candidates:
        if key in npz_obj:
            return key
    raise KeyError(f"Missing keys: {candidates}")


def load_pneumonia_npz():
    npz_path = DATA_DIR / "pneumoniamnist.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Could not find {npz_path}. Please place the dataset in data/."
        )

    npz = np.load(npz_path)
    train_images_key = _pick_key(npz, ["train_images", "train_img", "train"])
    test_images_key = _pick_key(npz, ["test_images", "test_img", "test"])
    train_labels_key = _pick_key(npz, ["train_labels", "train_label", "train_targets"])
    test_labels_key = _pick_key(npz, ["test_labels", "test_label", "test_targets"])

    train_images = npz[train_images_key]
    test_images = npz[test_images_key]
    train_labels = npz[train_labels_key]
    test_labels = npz[test_labels_key]

    if train_images.ndim == 4:
        train_images = train_images.squeeze(-1)
    if test_images.ndim == 4:
        test_images = test_images.squeeze(-1)

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    pneumonia = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "num_classes": 2,
    }
    return pneumonia


def build_summary_table(datasets_map):
    rows = []
    for name, data in datasets_map.items():
        rows.append(
            {
                "Ten Dataset": name,
                "So luong Train": len(data["train_images"]),
                "So luong Test": len(data["test_images"]),
                "Kich thuoc anh": data["train_images"].shape[1:],
                "So luong Class": data["num_classes"],
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== DATASET SUMMARY ===")
    print(df.to_string(index=False))


def plot_label_distribution(datasets_map):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, data) in zip(axes, datasets_map.items()):
        labels = np.concatenate([data["train_labels"], data["test_labels"]])
        counts = pd.Series(labels).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
        ax.set_title(name)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")

        if name.lower().startswith("pneumonia"):
            normal = counts.get(0, 0)
            pneu = counts.get(1, 0)
            ratio = normal / max(pneu, 1)
            print(
                f"Pneumonia Imbalance Ratio (Normal/Pneumonia): {ratio:.2f}"
                " -> Can nhac class_weight"
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_distribution.png", dpi=300)
    plt.close(fig)


def plot_pixel_histograms(digit, fashion, pneumonia):
    digit_images = np.concatenate([digit["train_images"], digit["test_images"]])
    fashion_images = np.concatenate([fashion["train_images"], fashion["test_images"]])
    pneu_labels = np.concatenate([pneumonia["train_labels"], pneumonia["test_labels"]])
    pneu_images = np.concatenate([pneumonia["train_images"], pneumonia["test_images"]])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(digit_images.ravel(), bins=50, color="#4C78A8", alpha=0.7)
    axes[0].set_title("Digit MNIST: Pixel Intensity")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(fashion_images.ravel(), bins=50, color="#4C78A8", alpha=0.7)
    axes[1].set_title("Fashion MNIST: Pixel Intensity")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    for cls, color in zip([0, 1], ["#54A24B", "#E45756"]):
        pixels = pneu_images[pneu_labels == cls].ravel()
        axes[2].hist(pixels, bins=50, alpha=0.5, color=color, label=f"Class {cls}")
    axes[2].set_title("Pneumonia MNIST: Normal vs Pneumonia")
    axes[2].set_xlabel("Pixel Intensity")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pixel_histogram_digit_pneumonia.png", dpi=300)
    plt.close(fig)

    flat_digit = digit_images.ravel()
    zero_mask = np.isclose(flat_digit, 0.0)
    one_mask = np.isclose(flat_digit, 1.0)
    other_mask = ~(zero_mask | one_mask)
    digit_counts = np.array([zero_mask.sum(), one_mask.sum(), other_mask.sum()])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.pie(
        digit_counts,
        labels=["0", "1", "Other"],
        autopct="%1.1f%%",
        textprops={"fontsize": 9},
    )
    ax.set_title("Digit MNIST: Pixel Values (0/1/Other)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "digit_pixel_value_pie.png", dpi=300)
    plt.close(fig)


def compute_mean_images(images, labels, num_classes):
    means = []
    for cls in range(num_classes):
        cls_images = images[labels == cls]
        means.append(cls_images.mean(axis=0))
    return np.stack(means)


def plot_mean_grids(mean_images, title, grid_shape, file_name):
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(mean_images):
            ax.axis("off")
            continue
        ax.imshow(mean_images[idx], cmap="gray")
        ax.set_title(f"Class {idx}")
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / file_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pneumonia_difference(mean_images):
    diff = mean_images[1] - mean_images[0]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(diff, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Pneumonia - Normal (Mean Difference)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pneumonia_difference_heatmap.png", dpi=300)
    plt.close(fig)


def plot_canny_grid(digit, fashion, pneumonia):
    rng = np.random.default_rng(RANDOM_STATE)

    digit_images = digit["train_images"]
    fashion_images = fashion["train_images"]
    pneumonia_images = pneumonia["train_images"]

    digit_idx = rng.choice(len(digit_images), size=5, replace=False)
    fashion_idx = rng.choice(len(fashion_images), size=5, replace=False)
    pneumonia_idx = rng.choice(len(pneumonia_images), size=5, replace=False)

    selected_images = np.concatenate(
        [
            digit_images[digit_idx],
            fashion_images[fashion_idx],
            pneumonia_images[pneumonia_idx],
        ]
    )

    fig, axes = plt.subplots(2, 15, figsize=(24, 4))

    for i, img in enumerate(selected_images):
        img_uint8 = (img * 255).astype("uint8")
        edges = cv2.Canny(img_uint8, threshold1=50, threshold2=150)

        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(edges, cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Canny", rotation=90, labelpad=10)

    axes[0, 2].set_title("Digit MNIST", fontsize=10)
    axes[0, 7].set_title("Fashion MNIST", fontsize=10)
    axes[0, 12].set_title("Pneumonia MNIST", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "canny_fashion_pneumonia.png", dpi=300)
    plt.close(fig)


def plot_cv_feature_extraction(digit, fashion, pneumonia):
    rng = np.random.default_rng(RANDOM_STATE)

    hog_samples = {
        "Digit MNIST": digit["train_images"],
        "Fashion MNIST": fashion["train_images"],
        "Pneumonia MNIST": pneumonia["train_images"],
    }

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    for row, (name, images) in enumerate(hog_samples.items()):
        idx = rng.integers(0, len(images))
        img = images[idx]
        hog_features, hog_image = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
        )

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"{name} - Original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hog_image, cmap="gray")
        axes[row, 1].set_title(f"{name} - HOG")
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hog_comparison.png", dpi=300)
    plt.close(fig)

    # Use fewer features and higher thresholds to avoid over-saturating keypoints.
    orb = cv2.ORB_create(nfeatures=200, fastThreshold=210, edgeThreshold=150)
    fashion_idx = rng.integers(0, len(fashion["train_images"]))
    pneumonia_idx = rng.integers(0, len(pneumonia["train_images"]))

    fashion_img = (fashion["train_images"][fashion_idx] * 255).astype("uint8")
    pneumonia_img = (pneumonia["train_images"][pneumonia_idx] * 255).astype("uint8")

    fashion_kp = orb.detect(fashion_img, None)
    pneumonia_kp = orb.detect(pneumonia_img, None)

    fashion_draw = cv2.drawKeypoints(
        cv2.cvtColor(fashion_img, cv2.COLOR_GRAY2BGR),
        fashion_kp,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    pneumonia_draw = cv2.drawKeypoints(
        cv2.cvtColor(pneumonia_img, cv2.COLOR_GRAY2BGR),
        pneumonia_kp,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    fashion_draw = cv2.cvtColor(fashion_draw, cv2.COLOR_BGR2RGB)
    pneumonia_draw = cv2.cvtColor(pneumonia_draw, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(fashion_draw)
    axes[0].set_title("Fashion MNIST - ORB")
    axes[0].axis("off")
    axes[1].imshow(pneumonia_draw)
    axes[1].set_title("Pneumonia MNIST - ORB")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "orb_keypoints.png", dpi=300)
    plt.close(fig)

    pneu_images = pneumonia["train_images"]
    pneu_labels = pneumonia["train_labels"]

    normal_idx = np.where(pneu_labels == 0)[0]
    pneu_idx = np.where(pneu_labels == 1)[0]

    sample_count = 200
    normal_sample = rng.choice(normal_idx, size=min(sample_count, len(normal_idx)), replace=False)
    pneu_sample = rng.choice(pneu_idx, size=min(sample_count, len(pneu_idx)), replace=False)

    def glcm_features(images):
        contrast_vals = []
        homogeneity_vals = []
        for img in images:
            quantized = np.clip(np.round(img * 7), 0, 7).astype("uint8")
            glcm = graycomatrix(
                quantized,
                distances=[1],
                angles=[0],
                levels=8,
                symmetric=True,
                normed=True,
            )
            contrast_vals.append(graycoprops(glcm, "contrast")[0, 0])
            homogeneity_vals.append(graycoprops(glcm, "homogeneity")[0, 0])
        return contrast_vals, homogeneity_vals

    normal_contrast, normal_homogeneity = glcm_features(pneu_images[normal_sample])
    pneu_contrast, pneu_homogeneity = glcm_features(pneu_images[pneu_sample])

    glcm_df = pd.DataFrame(
        {
            "Class": ["Normal"] * len(normal_contrast)
            + ["Pneumonia"] * len(pneu_contrast)
            + ["Normal"] * len(normal_homogeneity)
            + ["Pneumonia"] * len(pneu_homogeneity),
            "Metric": ["Contrast"] * len(normal_contrast)
            + ["Contrast"] * len(pneu_contrast)
            + ["Homogeneity"] * len(normal_homogeneity)
            + ["Homogeneity"] * len(pneu_homogeneity),
            "Value": normal_contrast
            + pneu_contrast
            + normal_homogeneity
            + pneu_homogeneity,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(
        data=glcm_df[glcm_df["Metric"] == "Contrast"],
        x="Class",
        y="Value",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("GLCM Contrast")
    sns.boxplot(
        data=glcm_df[glcm_df["Metric"] == "Homogeneity"],
        x="Class",
        y="Value",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("GLCM Homogeneity")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "glcm_boxplot.png", dpi=300)
    plt.close(fig)


def tsne_and_silhouette(datasets_map):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    silhouette_rows = []

    for ax, (name, data) in zip(axes, datasets_map.items()):
        images = np.concatenate([data["train_images"], data["test_images"]])
        labels = np.concatenate([data["train_labels"], data["test_labels"]])

        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.choice(len(images), size=TSNE_SAMPLES, replace=False)
        sample_images = images[indices].reshape(TSNE_SAMPLES, -1)
        sample_labels = labels[indices]

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca")
        embedding = tsne.fit_transform(sample_images)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=sample_labels,
            cmap="tab10",
            s=8,
            alpha=0.8,
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

        legend_handles, legend_labels = scatter.legend_elements()
        ax.legend(
            legend_handles,
            legend_labels,
            title="Class",
            loc="upper right",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )

        score = silhouette_score(embedding, sample_labels)
        silhouette_rows.append({"Dataset": name, "Silhouette": score})

    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsne_comparison.png", dpi=300)
    plt.close(fig)

    silhouette_df = pd.DataFrame(silhouette_rows)
    print("\n=== SILHOUETTE SCORES ===")
    print(silhouette_df.to_string(index=False))


def compute_ssim_metrics(pneumonia):
    images = np.concatenate([pneumonia["train_images"], pneumonia["test_images"]])
    labels = np.concatenate([pneumonia["train_labels"], pneumonia["test_labels"]])

    normal_idx = np.where(labels == 0)[0]
    pneu_idx = np.where(labels == 1)[0]

    rng = np.random.default_rng(RANDOM_STATE)
    pair_count = 100

    def sample_pairs(idx_a, idx_b):
        a = rng.choice(idx_a, size=pair_count, replace=True)
        b = rng.choice(idx_b, size=pair_count, replace=True)
        return zip(a, b)

    def mean_ssim(pairs):
        scores = []
        for i, j in pairs:
            score = ssim(images[i], images[j], data_range=1.0)
            scores.append(score)
        return float(np.mean(scores))

    nn = mean_ssim(sample_pairs(normal_idx, normal_idx))
    pp = mean_ssim(sample_pairs(pneu_idx, pneu_idx))
    np_score = mean_ssim(sample_pairs(normal_idx, pneu_idx))

    print("\n=== SSIM AVERAGES (PNEUMONIA MNIST) ===")
    print(f"Normal-Normal: {nn:.4f}")
    print(f"Pneumonia-Pneumonia: {pp:.4f}")
    print(f"Normal-Pneumonia: {np_score:.4f}")


def outlier_detection(pneumonia):
    images = np.concatenate([pneumonia["train_images"], pneumonia["test_images"]])
    sums = images.reshape(len(images), -1).sum(axis=1)

    mean_val = sums.mean()
    std_val = sums.std()
    z_scores = (sums - mean_val) / (std_val + 1e-8)

    sorted_idx = np.argsort(sums)
    lowest = sorted_idx[:5]
    highest = sorted_idx[-5:][::-1]

    print("\n=== OUTLIER Z-SCORES (PNEUMONIA MNIST) ===")
    for idx in lowest:
        print(f"Low intensity idx={idx}, sum={sums[idx]:.2f}, z={z_scores[idx]:.2f}")
    for idx in highest:
        print(f"High intensity idx={idx}, sum={sums[idx]:.2f}, z={z_scores[idx]:.2f}")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for col, idx in enumerate(lowest):
        axes[0, col].imshow(images[idx], cmap="gray")
        axes[0, col].axis("off")
    for col, idx in enumerate(highest):
        axes[1, col].imshow(images[idx], cmap="gray")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Low", rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("High", rotation=90, labelpad=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pneumonia_outliers.png", dpi=300)
    plt.close(fig)


def main():
    digit, fashion = load_mnist_sets()
    pneumonia = load_pneumonia_npz()

    datasets_map = {
        "Digit MNIST": digit,
        "Fashion MNIST": fashion,
        "Pneumonia MNIST": pneumonia,
    }

    build_summary_table(datasets_map)
    plot_label_distribution(datasets_map)
    plot_pixel_histograms(digit, fashion, pneumonia)

    digit_means = compute_mean_images(
        digit["train_images"], digit["train_labels"], digit["num_classes"]
    )
    fashion_means = compute_mean_images(
        fashion["train_images"], fashion["train_labels"], fashion["num_classes"]
    )
    pneumonia_means = compute_mean_images(
        pneumonia["train_images"], pneumonia["train_labels"], pneumonia["num_classes"]
    )

    plot_mean_grids(digit_means, "Digit MNIST Mean Images", (2, 5), "mean_digit.png")
    plot_mean_grids(
        fashion_means, "Fashion MNIST Mean Images", (2, 5), "mean_fashion.png"
    )
    plot_mean_grids(
        pneumonia_means, "Pneumonia MNIST Mean Images", (1, 2), "mean_pneumonia.png"
    )
    plot_pneumonia_difference(pneumonia_means)

    plot_canny_grid(digit, fashion, pneumonia)
    plot_cv_feature_extraction(digit, fashion, pneumonia)

    tsne_and_silhouette(datasets_map)

    compute_ssim_metrics(pneumonia)
    outlier_detection(pneumonia)

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
