"""GEO dataset downloader for neuroimmunology single-cell studies.

Fetches supplementary h5ad / count-matrix files from NCBI GEO for key
Alzheimer's disease and neuroimmunology single-cell RNA-seq datasets used
by the KNIGHT v1 foundation model.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

GEO_DATASETS: dict[str, dict[str, Any]] = {
    "mathys_2023": {
        "accession": "GSE188236",
        "description": (
            "Mathys et al. 2023 - Single-nucleus transcriptomics of the "
            "prefrontal cortex in Alzheimer's disease (MIT/Broad)"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "microglia", "astrocytes", "oligodendrocytes", "OPCs",
            "excitatory neurons", "inhibitory neurons",
        ],
        "n_cells_approx": 430_000,
    },
    "zhou_2020": {
        "accession": "GSE174367",
        "description": (
            "Zhou et al. 2020 - Human and mouse single-nucleus "
            "transcriptomics of Alzheimer's disease"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "microglia", "astrocytes", "oligodendrocytes", "OPCs",
            "excitatory neurons", "inhibitory neurons",
        ],
        "n_cells_approx": 150_000,
    },
    "leng_2021": {
        "accession": "GSE160936",
        "description": (
            "Leng et al. 2021 - Molecular characterization of the "
            "entorhinal cortex in Alzheimer's disease"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "excitatory neurons", "inhibitory neurons", "astrocytes",
            "microglia", "oligodendrocytes",
        ],
        "n_cells_approx": 42_000,
    },
    "grubman_2019": {
        "accession": "GSE148822",
        "description": (
            "Grubman et al. 2019 - A single-cell atlas of entorhinal "
            "cortex from individuals with Alzheimer's disease"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "astrocytes", "microglia", "oligodendrocytes",
            "excitatory neurons", "inhibitory neurons", "endothelial",
        ],
        "n_cells_approx": 13_000,
    },
    "lau_2020": {
        "accession": "GSE157827",
        "description": (
            "Lau et al. 2020 - Single-nucleus transcriptomic analysis "
            "of human dorsolateral prefrontal cortex in AD"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "excitatory neurons", "inhibitory neurons", "astrocytes",
            "microglia", "oligodendrocytes", "OPCs",
        ],
        "n_cells_approx": 170_000,
    },
    "sun_2023": {
        "accession": "GSE180759",
        "description": (
            "Sun et al. 2023 - Human microglial state dynamics in "
            "Alzheimer's disease progression"
        ),
        "organism": "Homo sapiens",
        "cell_types": ["microglia"],
        "n_cells_approx": 30_000,
    },
    "gabitto_2023": {
        "accession": "GSE202210",
        "description": (
            "Gabitto et al. 2023 - SEA-AD companion dataset; integrated "
            "multimodal cell atlas of Alzheimer's disease"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "microglia", "astrocytes", "oligodendrocytes", "OPCs",
            "excitatory neurons", "inhibitory neurons",
        ],
        "n_cells_approx": 340_000,
    },
    "green_2023": {
        "accession": "GSE229481",
        "description": (
            "Green et al. 2023 - ROSMAP single-cell multiomics atlas "
            "of the aging and Alzheimer's brain"
        ),
        "organism": "Homo sapiens",
        "cell_types": [
            "microglia", "astrocytes", "oligodendrocytes", "OPCs",
            "excitatory neurons", "inhibitory neurons", "pericytes",
        ],
        "n_cells_approx": 2_400_000,
    },
    # --- CRISPRi / Perturb-seq datasets (for perturbation prediction) ---
    "drager_2022_microglia_crispri": {
        "accession": "GSE178317",
        "description": (
            "Drager et al. 2022 - CRISPRi CROP-seq screen in iPSC-derived "
            "microglia targeting 81 genes of the druggable genome (Kampmann Lab)"
        ),
        "organism": "Homo sapiens",
        "cell_types": ["microglia"],
        "n_cells_approx": 29_000,
        "perturbation_type": "CRISPRi",
        "reference": "https://doi.org/10.1038/s41593-022-01131-4",
    },
    "leng_2022_astrocyte_crispri": {
        "accession": "GSE182308",
        "description": (
            "Leng et al. 2022 - CRISPRi CROP-seq screen in iPSC-derived "
            "astrocytes identifying regulators of inflammatory reactive states "
            "(Kampmann Lab)"
        ),
        "organism": "Homo sapiens",
        "cell_types": ["astrocytes"],
        "n_cells_approx": 20_000,
        "perturbation_type": "CRISPRi",
        "reference": "https://doi.org/10.1038/s41593-022-01180-9",
    },
}

# NCBI GEO FTP base
_GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# Timeout for HTTP/FTP requests (seconds)
_HTTP_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _geo_ftp_url(accession: str) -> str:
    """Build the GEO series FTP directory URL for *accession*."""
    prefix = accession[:5] + "nnn"
    return f"{_GEO_FTP_BASE}/{prefix}/{accession}"


def _suppl_url(accession: str) -> str:
    """URL for the supplementary files tarball."""
    return f"{_geo_ftp_url(accession)}/suppl/{accession}_RAW.tar"


def _matrix_url(accession: str) -> str:
    """URL for the series matrix (soft) file."""
    return (
        f"{_geo_ftp_url(accession)}/matrix/"
        f"{accession}_series_matrix.txt.gz"
    )


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, streaming with progress logging."""
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if total and written % (50 << 20) < (1 << 20):
                        logger.info(
                            "  %.1f%% (%d / %d bytes)",
                            written / total * 100,
                            written,
                            total,
                        )
        shutil.move(str(tmp), str(dest))
        logger.info("Saved %s (%d bytes)", dest.name, dest.stat().st_size)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def _try_geoparse(accession: str, output_dir: Path) -> Path | None:
    """Try to download using GEOparse; return path or None."""
    try:
        import GEOparse  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("GEOparse not installed; using direct FTP download.")
        return None

    logger.info("Fetching %s via GEOparse", accession)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(output_dir), silent=True)
    # GEOparse stores the soft file under destdir
    soft_path = output_dir / f"{accession}_family.soft.gz"
    if soft_path.exists():
        return soft_path
    return None


def _extract_tar(tar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a tar archive and return paths of extracted files."""
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
        for member in tf.getmembers():
            extracted.append(dest_dir / member.name)
    logger.info("Extracted %d files from %s", len(extracted), tar_path.name)
    return extracted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_geo_dataset(
    accession: str,
    output_dir: str | Path,
    use_geoparse: bool = True,
) -> Path:
    """Download a single GEO dataset by accession.

    Parameters
    ----------
    accession:
        GEO series accession (e.g. ``"GSE188236"``).
    output_dir:
        Root directory for downloads.  A subdirectory named after the
        accession will be created.
    use_geoparse:
        Attempt to use the ``GEOparse`` package before falling back to
        direct FTP download.

    Returns
    -------
    Path
        Directory containing the downloaded files for this accession.
    """
    output_dir = Path(output_dir) / accession
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GEO dataset %s to %s", accession, output_dir)

    # Try GEOparse first
    if use_geoparse:
        result = _try_geoparse(accession, output_dir)
        if result is not None:
            logger.info("GEOparse download succeeded for %s", accession)

    # Always try to grab the supplementary tarball (count matrices etc.)
    tar_dest = output_dir / f"{accession}_RAW.tar"
    if not tar_dest.exists():
        try:
            _download(_suppl_url(accession), tar_dest)
            _extract_tar(tar_dest, output_dir)
        except Exception as exc:
            logger.warning(
                "Could not download supplementary tarball for %s: %s",
                accession,
                exc,
            )

    # Grab the series matrix as a lightweight metadata fallback
    matrix_dest = output_dir / f"{accession}_series_matrix.txt.gz"
    if not matrix_dest.exists():
        try:
            _download(_matrix_url(accession), matrix_dest)
        except Exception as exc:
            logger.warning(
                "Could not download series matrix for %s: %s", accession, exc
            )

    logger.info("Finished downloading %s", accession)
    return output_dir


def download_all_geo(
    output_dir: str | Path,
    datasets: Sequence[str] | None = None,
    use_geoparse: bool = True,
) -> dict[str, Path]:
    """Download multiple GEO datasets.

    Parameters
    ----------
    output_dir:
        Root download directory.
    datasets:
        Dataset keys from :data:`GEO_DATASETS` to download.
        If ``None``, all registered datasets are downloaded.
    use_geoparse:
        Forward to :func:`download_geo_dataset`.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset key to the local directory for each dataset.
    """
    output_dir = Path(output_dir)
    keys = list(datasets) if datasets is not None else list(GEO_DATASETS)

    results: dict[str, Path] = {}
    for key in keys:
        if key not in GEO_DATASETS:
            logger.warning("Unknown dataset key %r; skipping.", key)
            continue
        info = GEO_DATASETS[key]
        accession = info["accession"]
        logger.info(
            "=== %s (%s): %s ===",
            key,
            accession,
            info["description"][:80],
        )
        try:
            results[key] = download_geo_dataset(
                accession, output_dir, use_geoparse=use_geoparse
            )
        except Exception:
            logger.exception("Failed to download %s (%s)", key, accession)

    logger.info(
        "GEO download complete: %d / %d datasets succeeded",
        len(results),
        len(keys),
    )
    return results
