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
    "t-SNE gave the clearest separation of the three methods. Rare populations like CD34+ and "
    "CD4+/CD45RA+/CD25- Naive T cells are nearly invisible in PCA but show up as distinct "
    "clusters in t-SNE. PCA is a linear projection, so T cell subtypes — which differ in "
    "subtle, high-dimensional ways — tend to collapse onto each other in the first two PCs. "
    "The autoencoder latent space is harder to read visually. Even after projecting the "
    "32-dimensional encoding down with PCA, there is substantial cluster overlap. The model "
    "was trained for reconstruction, not class separation, and with 1.1M parameters on 560 "
    "training samples it still overfits somewhat despite dropout — train MSE ~0.79, val MSE "
    "~0.87. Dropout brought the gap down from ~0.19 to ~0.09, which helped. For visualization "
    "t-SNE is clearly the better choice; the autoencoder encoding is more useful as a feature "
    "representation for a downstream classifier."
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
     'Figure 3. Top 20 genes by Random Forest feature importance. Top-ranked genes are known PBMC cell type markers.'),
    ('figures/onehot_labels.png',
     'Figure 4. One-hot encoded cell type labels for first 50 training cells.'),
]

part2_writeup = (
    "A Random Forest with 300 trees and balanced class weights was trained on the 765-gene "
    "expression features using a stratified 80/20 split, achieving 85.0% test accuracy "
    "(5-fold CV: 81.6% +/- 1.5%). Stratified splitting was important here — with a 30:1 "
    "class imbalance, a random split can easily leave the rarest classes barely represented "
    "in the test set. "
    "F1 scores give a more honest picture than accuracy alone. The model does well on the "
    "larger classes: CD19+ B and Dendritic cells both have F1 above 0.90, and CD14+ Monocyte "
    "is close at 0.89. Smaller classes are harder — CD8+ Cytotoxic T drops to 0.62. Two "
    "classes, CD4+/CD45RA+/CD25- Naive T and CD4+/CD45RO+ Memory, have F1=0, but those have "
    "only 1 and 4 test samples respectively. There is not enough data there to say much. "
    "The top genes by feature importance include CD79A, GNLY, NKG7, and CD3D — known markers "
    "for B cells, NK/cytotoxic T cells, and T cells — alongside monocyte markers like LYZ "
    "and CST3. The model is picking up on real biological signal rather than noise."
)

make_pdf('submission_part2.pdf',
         'BioE 145 Final Project — Part 2: Cell Type Classification',
         part2_figures, part2_writeup)
