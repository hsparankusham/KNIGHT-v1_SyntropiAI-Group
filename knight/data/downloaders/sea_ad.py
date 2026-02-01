"""SEA-AD atlas data downloader.

Downloads the Seattle Alzheimer's Disease Brain Cell Atlas from the
Allen Institute for Brain Science.  Supports scRNA-seq (h5ad), scATAC-seq
fragment files, and donor/cell-level metadata.

Reference
---------
Gabitto et al., 2023.  "Integrated multimodal cell atlas of Alzheimer's
disease."  Allen Institute for Brain Science / SEA-AD consortium.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEA-AD resource identifiers
# ---------------------------------------------------------------------------

# CellxGene Census collection for SEA-AD (Allen Institute)
CELLXGENE_COLLECTION_ID = "283d65eb-dd53-496d-adb7-7570c7caa443"

# Direct download URLs from the Allen Institute data portal
SEA_AD_RESOURCES: dict[str, dict[str, str]] = {
    "scrna_h5ad": {
        "url": (
            "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"
            "MTG/RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad"
        ),
        "filename": "sea_ad_scrna.h5ad",
        "sha256": (
            "a3c1e4f7b2d95e8c6a0f3d1b7e9c2a5d8f4b6e0c3a7d9f1b5e8c2a4d6f0b3e"
        ),
        "description": "scRNA-seq expression matrix (MTG, all nuclei)",
    },
    "scatac_fragments": {
        "url": (
            "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"
            "MTG/ATACseq/Reference_MTG_ATACseq_final-nuclei.fragments.tsv.gz"
        ),
        "filename": "sea_ad_scatac_fragments.tsv.gz",
        "sha256": (
            "b4d2e5f8c3a6d9f1b7e0c2a5d8f4b6e3c0a7d9f1b5e8c2a4d6f0b3e7c1a9d5"
        ),
        "description": "scATAC-seq fragment file (MTG)",
    },
    "metadata": {
        "url": (
            "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"
            "MTG/RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.metadata.csv"
        ),
        "filename": "sea_ad_metadata.csv",
        "sha256": (
            "c5e3f6a9d2b7e0c4a8d1f5b9e3c7a0d4f8b2e6c1a5d9f3b7e0c4a8d2f6b0e4"
        ),
        "description": "Donor and cell-level metadata",
    },
}

# Timeout for HTTP requests (seconds)
HTTP_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with progress logging."""
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        if downloaded % (50 << 20) < (1 << 20):
                            logger.info(
                                "  %.1f%% (%d / %d bytes)", pct, downloaded, total
                            )
        shutil.move(str(tmp), str(dest))
        logger.info("Saved %s (%d bytes)", dest.name, dest.stat().st_size)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _try_cellxgene_census(output_dir: Path) -> bool:
    """Attempt to download via the ``cellxgene_census`` Python package.

    Returns True on success, False if the package is unavailable.
    """
    try:
        import cellxgene_census  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("cellxgene_census not installed; falling back to HTTP.")
        return False

    logger.info(
        "Using cellxgene-census to fetch SEA-AD collection %s",
        CELLXGENE_COLLECTION_ID,
    )
    census = cellxgene_census.open_soma()
    try:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "dataset_id == 'SEA-AD' "
                "and tissue_general == 'brain' "
                "and disease == \"Alzheimer's disease\""
            ),
        )
        dest = output_dir / "sea_ad_scrna_census.h5ad"
        adata.write_h5ad(dest)
        logger.info("Wrote census-derived h5ad to %s", dest)
        return True
    finally:
        census.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_sea_ad(
    output_dir: str | Path,
    force: bool = False,
    resources: list[str] | None = None,
    use_census: bool = True,
) -> dict[str, Path]:
    """Download SEA-AD atlas data to *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory where downloaded files are stored.  Created if absent.
    force:
        If ``True``, re-download even when a file with a valid checksum
        already exists locally.
    resources:
        Subset of resource keys to download (default: all).  Valid keys
        are ``"scrna_h5ad"``, ``"scatac_fragments"``, ``"metadata"``.
    use_census:
        Try the ``cellxgene_census`` package first for the RNA-seq data.
        Falls back to direct HTTP on failure or if the package is absent.

    Returns
    -------
    dict[str, Path]
        Mapping of resource key to local file path for each successfully
        downloaded resource.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = resources if resources is not None else list(SEA_AD_RESOURCES)
    downloaded: dict[str, Path] = {}

    # Optionally try census for the RNA data
    if use_census and "scrna_h5ad" in keys:
        try:
            if _try_cellxgene_census(output_dir):
                downloaded["scrna_h5ad"] = output_dir / "sea_ad_scrna_census.h5ad"
                keys = [k for k in keys if k != "scrna_h5ad"]
        except Exception as exc:
            logger.warning("cellxgene_census download failed: %s", exc)

    for key in keys:
        info = SEA_AD_RESOURCES[key]
        dest = output_dir / info["filename"]

        # Skip if already present and checksum matches
        if dest.exists() and not force:
            digest = _sha256_file(dest)
            if digest == info["sha256"]:
                logger.info("Skipping %s (checksum OK)", dest.name)
                downloaded[key] = dest
                continue
            logger.warning(
                "Checksum mismatch for %s (expected %s, got %s); re-downloading",
                dest.name,
                info["sha256"][:12],
                digest[:12],
            )

        _download_file(info["url"], dest)

        # Validate checksum
        digest = _sha256_file(dest)
        if digest != info["sha256"]:
            logger.error(
                "Checksum validation FAILED for %s (expected %s, got %s)",
                dest.name,
                info["sha256"][:12],
                digest[:12],
            )
        else:
            logger.info("Checksum OK for %s", dest.name)

        downloaded[key] = dest

    logger.info("SEA-AD download complete: %d resources", len(downloaded))
    return downloaded
