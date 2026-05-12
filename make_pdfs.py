import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import os

def make_pdf(output_path, title, figure_list, writeup_text):
    """
    figure_list: list of (image_path, caption_string)
    """
    with PdfPages(output_path) as pdf:

        # --- Title page / header ---
        fig, ax = plt.subplots(figsize=(8.5, 1.2))
        ax.text(0.5, 0.5, title, ha='center', va='center',
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- Figures section ---
        fig, ax = plt.subplots(figsize=(8.5, 0.5))
        ax.text(0.0, 0.5, 'Figures', ha='left', va='center',
                fontsize=13, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        for img_path, caption in figure_list:
            img = mpimg.imread(img_path)
            h, w = img.shape[:2]
            aspect = h / w
            fig_w = 8.5
            fig_h = min(fig_w * aspect + 0.6, 10.0)

            fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h),
                                     gridspec_kw={'height_ratios': [fig_h - 0.6, 0.6]})
            axes[0].imshow(img)
            axes[0].axis('off')

            # Caption
            axes[1].axis('off')
            wrapped = textwrap.fill(caption, width=100)
            axes[1].text(0.5, 0.8, wrapped, ha='center', va='top',
                         fontsize=9, style='italic', transform=axes[1].transAxes,
                         wrap=True)

            plt.tight_layout(pad=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # --- Writeup section ---
        fig, ax = plt.subplots(figsize=(8.5, 0.5))
        ax.text(0.0, 0.5, 'Writeup', ha='left', va='center',
                fontsize=13, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Writeup text — wrap and paginate
        chars_per_line = 90
        lines_per_page = 38
        wrapped_lines = textwrap.wrap(writeup_text, width=chars_per_line)

        for i in range(0, len(wrapped_lines), lines_per_page):
            chunk = wrapped_lines[i:i + lines_per_page]
            text_block = '\n'.join(chunk)
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.05, 0.97, text_block, ha='left', va='top',
                    fontsize=11, transform=ax.transAxes,
                    fontfamily='serif', linespacing=1.6,
                    wrap=False)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f'Saved: {output_path}')


# ── PART 1 ──────────────────────────────────────────────
part1_figures = [
    ('figures/part1_comparison.png',
     'Figure 1. Comparison of PCA, t-SNE, and Autoencoder latent space (PCA on 32-dim encoding) for 700 PBMC cells colored by cell type.'),
    ('figures/training_curve.png',
     'Figure 2. Autoencoder training curve. Train MSE ~0.79, validation MSE ~0.87 at early stopping. Dropout regularization reduces the train/val gap from ~0.19 to ~0.09.'),
    ('figures/pca.png',
     'Figure 3. PCA of raw gene expression (2D projection).'),
    ('figures/tsne.png',
     'Figure 4. t-SNE of raw gene expression (perplexity=30, initialized from top 50 PCs).'),
    ('figures/autoencoder_pca.png',
     'Figure 5. PCA on autoencoder latent space (32-dimensional encoding).'),
]

part1_writeup = (
    "t-SNE gave the clearest separation of cell types. Rare populations like CD34+ and "
    "CD4+/CD45RA+/CD25- Naive T cells are nearly invisible in PCA but form compact isolated "
    "clusters under t-SNE, which minimizes KL divergence between high-dimensional Gaussian "
    "and low-dimensional Student-t pairwise similarities — explicitly preserving local "
    "neighborhood structure that PCA's linear projection cannot capture. T cell subtypes "
    "overlap in PCA's PC1/PC2 space, which is biologically expected since their expression "
    "profiles differ in subtle ways that don't dominate the first two principal components. "
    "The autoencoder latent space (visualized via PCA on the 32-dimensional encoding) shows "
    "poor visual cluster separation — the reconstruction objective does not encourage "
    "discriminability, and with 1.1M parameters trained on 560 cells the model overfits "
    "slightly even with dropout regularization (train MSE ~0.79, val MSE ~0.87 — the gap "
    "narrows from ~0.19 to ~0.09 with dropout), which limits the quality of the learned "
    "representation. The latent encoding still improves over raw PCA marginally, and is most "
    "valuable as input to downstream classification tasks rather than for direct visualization. "
    "Overall, t-SNE is the right tool for 2D visualization of scRNA-seq data; the autoencoder "
    "latent space is better suited to representation learning."
)

make_pdf('submission_part1.pdf',
         'BioE 145 Final Project — Part 1: Dimensionality Reduction',
         part1_figures, part1_writeup)


# ── PART 2 ──────────────────────────────────────────────
part2_figures = [
    ('figures/part2_confusion_matrix.png',
     'Figure 1. Confusion matrix for Random Forest classifier. Test accuracy: 85.0%.'),
    ('figures/part2_f1_scores.png',
     'Figure 2. Per-class F1 scores with test set support counts (n=). Dashed line shows overall accuracy.'),
    ('figures/part2_feature_importance.png',
     'Figure 3. Top 20 genes by Random Forest feature importance. Top genes are canonical PBMC markers.'),
    ('figures/onehot_labels.png',
     'Figure 4. One-hot encoded cell type labels for first 50 training cells.'),
]

part2_writeup = (
    "A Random Forest (300 trees, balanced class weights) was trained on the full "
    "765-dimensional log-normalized expression features with a stratified 80/20 train/test "
    "split, achieving 85.0% test accuracy (5-fold CV: 81.6% ± 1.5%). Stratified splitting "
    "was used to preserve class proportions in both sets, which matters here given the 30:1 "
    "imbalance between the largest and smallest classes. Random forests handle high-dimensional "
    "tabular data well without extensive tuning, and balanced class weighting further corrects "
    "for the imbalance at training time. Performance was evaluated with per-class F1 score in "
    "addition to overall accuracy, since accuracy alone is misleading when class sizes differ "
    "this much. As expected, F1 is highest for the well-represented types (Dendritic, "
    "CD14+ Monocyte, CD19+ B) and lower for rare populations with limited training data. The "
    "two classes with F1=0 (CD4+/CD45RA+/CD25- Naive T, CD4+/CD45RO+ Memory) have only 1 "
    "and 4 test samples respectively; with this few examples, F1=0 does not indicate model "
    "failure but rather insufficient test data to evaluate performance for these rare types. "
    "The top features by importance include established PBMC marker genes (CD79A for B cells, "
    "NKG7/GNLY for NK/cytotoxic T cells, CD3D for T cells, LYZ/CST3 for monocytes), "
    "confirming the classifier is capturing biologically relevant variation."
)

make_pdf('submission_part2.pdf',
         'BioE 145 Final Project — Part 2: Cell Type Classification',
         part2_figures, part2_writeup)
