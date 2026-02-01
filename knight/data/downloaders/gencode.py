"""GENCODE gene annotation downloader.

Downloads and parses the GENCODE GTF annotation file for GRCh38/hg38,
which is required for scATAC-seq peak annotation (classifying peaks as
promoter, gene body, or intergenic).
"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# GENCODE v47 (GRCh38.p14) — matches 10x Cell Ranger 2024-A reference
GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_47/gencode.v47.primary_assembly.annotation.gtf.gz"
)
GENCODE_VERSION = "v47"


def download_gencode_gtf(
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    """Download the GENCODE GTF annotation file.

    Parameters
    ----------
    output_dir:
        Directory to save the GTF file.
    force:
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded (compressed) GTF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"gencode.{GENCODE_VERSION}.primary_assembly.annotation.gtf.gz"

    if dest.exists() and not force:
        logger.info("GENCODE GTF already exists: %s", dest)
        return dest

    logger.info("Downloading GENCODE %s GTF from %s", GENCODE_VERSION, GENCODE_GTF_URL)
    tmp = dest.with_suffix(".gz.part")
    try:
        urllib.request.urlretrieve(GENCODE_GTF_URL, str(tmp))
        tmp.rename(dest)
        logger.info("Saved GENCODE GTF: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


def parse_gencode_genes(
    gtf_path: str | Path,
    feature_type: str = "gene",
    gene_types: list[str] | None = None,
) -> pd.DataFrame:
    """Parse a GENCODE GTF file into a gene annotation DataFrame.

    Parameters
    ----------
    gtf_path:
        Path to the compressed (.gz) or uncompressed GTF file.
    feature_type:
        GTF feature type to extract (default: ``"gene"``).
    gene_types:
        Filter to specific gene biotypes (e.g. ``["protein_coding"]``).
        Default: all types.

    Returns
    -------
    pd.DataFrame
        Columns: ``chrom``, ``start``, ``end``, ``strand``, ``gene_name``,
        ``gene_id``, ``gene_type``.
    """
    gtf_path = Path(gtf_path)
    logger.info("Parsing GENCODE GTF: %s", gtf_path)

    opener = gzip.open if gtf_path.suffix == ".gz" else open
    records = []

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != feature_type:
                continue

            attrs = _parse_gtf_attributes(fields[8])
            gene_type = attrs.get("gene_type", "")
            if gene_types and gene_type not in gene_types:
                continue

            records.append({
                "chrom": fields[0],
                "start": int(fields[3]),
                "end": int(fields[4]),
                "strand": fields[6],
                "gene_name": attrs.get("gene_name", ""),
                "gene_id": attrs.get("gene_id", ""),
                "gene_type": gene_type,
            })

    df = pd.DataFrame(records)
    logger.info(
        "Parsed %d genes (%d protein-coding)",
        len(df),
        (df["gene_type"] == "protein_coding").sum(),
    )
    return df


def _parse_gtf_attributes(attr_string: str) -> dict[str, str]:
    """Parse GTF attribute string into a dict."""
    attrs = {}
    for item in attr_string.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" ", 1)
        if len(parts) == 2:
            key = parts[0]
            value = parts[1].strip('"')
            attrs[key] = value
    return attrs


def get_gene_annotation(
    cache_dir: str | Path = "data/external",
    force: bool = False,
) -> pd.DataFrame:
    """Download (if needed) and return the gene annotation DataFrame.

    This is the main entry point — handles downloading and caching.

    Parameters
    ----------
    cache_dir:
        Directory for caching the GTF and parsed parquet.
    force:
        Force re-download and re-parse.

    Returns
    -------
    pd.DataFrame
        Gene annotation with columns: chrom, start, end, strand,
        gene_name, gene_id, gene_type.
    """
    cache_dir = Path(cache_dir)
    parquet_cache = cache_dir / f"gencode_{GENCODE_VERSION}_genes.parquet"

    if parquet_cache.exists() and not force:
        logger.info("Loading cached gene annotation: %s", parquet_cache)
        return pd.read_parquet(parquet_cache)

    gtf_path = download_gencode_gtf(cache_dir, force=force)
    df = parse_gencode_genes(gtf_path)

    # Cache as parquet for fast reloading
    df.to_parquet(parquet_cache, index=False)
    logger.info("Cached gene annotation to %s", parquet_cache)

    return df
