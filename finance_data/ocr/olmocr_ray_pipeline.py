"""olmOCR pipeline using Ray Data + vLLM offline inference.

This is an alternative to olmocr_pipeline.py that replaces the async HTTP
client approach with Ray Data's map_batches and vLLM's offline LLM engine.

Architecture:
  1. Build a Ray Dataset where each row is one PDF page record.
  2. Render each page to a base64 PNG image (CPU actors).
  3. Run vLLM offline inference in batches on GPU actors.
  4. Aggregate page results back into per-PDF Dolma documents.
"""

from __future__ import annotations

import base64
import datetime
import glob
import hashlib
import logging
import os
import pathlib
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from loguru import logger as _loguru_logger
from PIL import Image
from pypdf import PdfReader

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter.filter import Language, PdfFilter
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.train.dataloader import FrontMatterParser
from olmocr.version import VERSION

from finance_data.settings import sec_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}"
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, format=_LOG_FMT, level="INFO")
logger = _loguru_logger.bind(name=__name__)

logging.getLogger("pypdf").setLevel(logging.ERROR)

DEFAULT_MODEL = sec_settings.olmocr_model
DEFAULT_WORKSPACE = sec_settings.olmocr_workspace

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RayOcrConfig:
    """Configuration for the Ray Data + vLLM OCR pipeline."""

    workspace: str
    model: str = DEFAULT_MODEL
    max_page_retries: int = 3
    max_page_error_rate: float = 0.004
    target_longest_image_dim: int = 1288
    apply_filter: bool = False
    markdown: bool = True
    # vLLM engine kwargs forwarded to vllm.LLM
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    max_tokens: int = 8000
    temperature: float = 0.1
    # Ray Data concurrency – number of GPU replicas for the predictor
    num_gpu_actors: int = 1
    # CPU concurrency for page rendering
    num_cpu_actors: int = 4
    # batch size fed into vLLM per map_batches call
    batch_size: int = 32


# ---------------------------------------------------------------------------
# Helper: PDF page rendering (CPU)
# ---------------------------------------------------------------------------

def _is_tarball(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz")


def _get_pdf_filter():
    return PdfFilter(
        languages_to_keep={Language.ENGLISH, None},
        apply_download_spam_check=True,
        apply_form_check=True,
    )


def _expand_pdf_to_pages(
    row: dict,
    target_longest_image_dim: int,
    apply_filter: bool,
) -> list[dict]:
    """
    Given a row with keys 'pdf_path' (original path) and 'local_path'
    (local temp copy), render every page and return one dict per page.

    Each returned dict has:
        pdf_path, local_path, page_num, num_pages, image_base64
    """
    pdf_path: str = row["pdf_path"]
    local_path: str = row["local_path"]

    if apply_filter:
        try:
            if _get_pdf_filter().filter_out_pdf(local_path):
                logger.info(f"Filtering out {pdf_path}")
                return []
        except Exception:
            pass

    try:
        reader = PdfReader(local_path)
        num_pages = reader.get_num_pages()
    except Exception as exc:
        logger.warning(f"Could not read {pdf_path}: {exc}")
        return []

    pages = []
    for page_num in range(1, num_pages + 1):
        try:
            image_base64 = render_pdf_to_base64png(
                local_path,
                page_num,
                target_longest_image_dim=target_longest_image_dim,
            )
        except Exception as exc:
            logger.warning(f"Render failed {pdf_path} p{page_num}: {exc}")
            image_base64 = None

        pages.append(
            {
                "pdf_path": pdf_path,
                "local_path": local_path,
                "page_num": page_num,
                "num_pages": num_pages,
                "image_base64": image_base64,
            }
        )
    return pages


# ---------------------------------------------------------------------------
# vLLM predictor actor (GPU)
# ---------------------------------------------------------------------------

class OlmOCRPredictor:
    """Ray Data actor that runs vLLM offline inference for olmOCR."""

    def __init__(self, config: RayOcrConfig):
        # Import inside actor so the driver process doesn't need a GPU.
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self._prompt_text = build_no_anchoring_v4_yaml_prompt()
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._parser = FrontMatterParser(front_matter_class=PageResponse)

        llm_kwargs: dict[str, Any] = {
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"video": 0},
        }
        self._llm = LLM(**llm_kwargs)
        logger.info(f"OlmOCRPredictor: loaded model {config.model}")

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Process a batch of page rows.

        Input batch keys: pdf_path, local_path, page_num, num_pages, image_base64
        Output batch adds:  natural_text, primary_language, is_rotation_valid,
                            rotation_correction, is_table, is_diagram,
                            input_tokens, output_tokens, is_fallback, is_valid
        """
        from vllm import SamplingParams

        pdf_paths = batch["pdf_path"]
        local_paths = batch["local_path"]
        page_nums = batch["page_num"]
        num_pages_list = batch["num_pages"]
        images_b64 = batch["image_base64"]

        sampling_params = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Build vLLM chat messages for valid pages
        messages_list: list[list[dict]] = []
        valid_indices: list[int] = []
        for i, img_b64 in enumerate(images_b64):
            if img_b64 is None:
                continue
            valid_indices.append(i)
            messages_list.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ]
            )

        # Run inference (chat mode)
        outputs = self._llm.chat(
            messages_list,  # type: ignore[arg-type]
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Build result arrays aligned to input batch
        natural_texts: list[str] = [""] * len(pdf_paths)
        primary_languages: list[str | None] = [None] * len(pdf_paths)
        is_rotation_valids = [True] * len(pdf_paths)
        rotation_corrections = [0] * len(pdf_paths)
        is_tables = [False] * len(pdf_paths)
        is_diagrams = [False] * len(pdf_paths)
        input_tokens_list = [0] * len(pdf_paths)
        output_tokens_list = [0] * len(pdf_paths)
        is_fallbacks = [False] * len(pdf_paths)
        is_valids = [False] * len(pdf_paths)

        for vi, out in zip(valid_indices, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            in_tok = len(out.prompt_token_ids) if out.prompt_token_ids else 0
            out_tok = len(out.outputs[0].token_ids) if out.outputs else 0
            finish_reason = out.outputs[0].finish_reason if out.outputs else None

            is_valid = finish_reason == "stop"

            try:
                front_matter, text = self._parser._extract_front_matter_and_text(raw_text)
                page_response: PageResponse = self._parser._parse_front_matter(
                    front_matter, text
                )
                natural_texts[vi] = page_response.natural_text or ""
                primary_languages[vi] = page_response.primary_language
                is_rotation_valids[vi] = page_response.is_rotation_valid
                rotation_corrections[vi] = page_response.rotation_correction
                is_tables[vi] = page_response.is_table
                is_diagrams[vi] = page_response.is_diagram
            except Exception as exc:
                logger.warning(
                    f"Parse failed {pdf_paths[vi]} p{page_nums[vi]}: {exc}"
                )
                is_valid = False
                natural_texts[vi] = _fallback_text(local_paths[vi], page_nums[vi])
                is_fallbacks[vi] = True

            input_tokens_list[vi] = in_tok
            output_tokens_list[vi] = out_tok
            is_valids[vi] = is_valid

        # Fallback for pages where rendering failed
        for i in range(len(pdf_paths)):
            if i not in valid_indices:
                natural_texts[i] = _fallback_text(local_paths[i], page_nums[i])
                is_fallbacks[i] = True
                is_valids[i] = True  # fallback counts as valid output

        return {
            "pdf_path": pdf_paths,
            "local_path": local_paths,
            "page_num": page_nums,
            "num_pages": num_pages_list,
            "natural_text": natural_texts,
            "primary_language": primary_languages,
            "is_rotation_valid": is_rotation_valids,
            "rotation_correction": rotation_corrections,
            "is_table": is_tables,
            "is_diagram": is_diagrams,
            "input_tokens": input_tokens_list,
            "output_tokens": output_tokens_list,
            "is_fallback": is_fallbacks,
            "is_valid": is_valids,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_to_pil(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def _fallback_text(local_path: str, page_num: int) -> str:
    try:
        return get_anchor_text(local_path, page_num, pdf_engine="pdftotext") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def _build_dolma_document(pdf_path: str, page_rows: list[dict]) -> dict | None:
    """Assemble a Dolma document from sorted page result rows."""
    page_rows = sorted(page_rows, key=lambda r: r["page_num"])

    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, row in enumerate(page_rows):
        content = (row["natural_text"] or "") + (
            "\n" if index < len(page_rows) - 1 else ""
        )
        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, row["page_num"]])

    if not document_text.strip():
        logger.info(f"No document text for {pdf_path}")
        return None

    metadata = {
        "Source-File": pdf_path,
        "olmocr-version": VERSION,
        "pdf-total-pages": len(page_rows),
        "total-input-tokens": sum(r["input_tokens"] for r in page_rows),
        "total-output-tokens": sum(r["output_tokens"] for r in page_rows),
        "total-fallback-pages": sum(r["is_fallback"] for r in page_rows),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    return {
        "id": id_,
        "text": document_text,
        "source": "olmocr",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans,
            "primary_language": [r["primary_language"] for r in page_rows],
            "is_rotation_valid": [r["is_rotation_valid"] for r in page_rows],
            "rotation_correction": [r["rotation_correction"] for r in page_rows],
            "is_table": [r["is_table"] for r in page_rows],
            "is_diagram": [r["is_diagram"] for r in page_rows],
        },
    }


def _build_markdown_with_page_tags(document_text: str, pdf_page_spans: list) -> str:
    result = ""
    for start_char, end_char, page_num in pdf_page_spans:
        page_text = document_text[start_char:end_char]
        result += f"<PAGE-NUM-{page_num}>\n{page_text}</PAGE-NUM-{page_num}>\n"
    return result


def _get_markdown_path(workspace: str, source_file: str) -> str:
    if "::" in source_file:
        tarball_path, internal_path = source_file.split("::", 1)
        tarball_basename = os.path.splitext(os.path.basename(tarball_path))[0]
        if tarball_basename.endswith(".tar"):
            tarball_basename = tarball_basename[:-4]
        relative_path = os.path.join(tarball_basename, internal_path)
    else:
        relative_path = source_file.lstrip("/")

    parts = relative_path.split("/")
    safe_parts = [p for p in parts if p and p != ".."]
    relative_path = "/".join(safe_parts)

    md_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".md"
    dir_path = os.path.dirname(relative_path)
    markdown_dir = os.path.join(workspace, "markdown", dir_path)
    return os.path.join(markdown_dir, md_filename)


# ---------------------------------------------------------------------------
# Input preparation: collect (original_path, local_tmp_path) pairs
# ---------------------------------------------------------------------------

def _collect_pdf_rows(pdf_paths: list[str]) -> list[dict]:
    """
    Return a list of dicts suitable for a Ray Dataset.
    Each dict: {pdf_path: str, local_path: str}

    Tarballs are expanded; their extracted temp files are returned as rows with
    the tarball::internal notation in pdf_path.
    """
    rows = []
    for path in pdf_paths:
        if not os.path.exists(path):
            logger.warning(f"Path not found, skipping: {path}")
            continue

        if _is_tarball(path):
            rows.extend(_expand_tarball(path))
        elif path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            local_path = _copy_to_tmp(path)
            if local_path:
                rows.append({"pdf_path": path, "local_path": local_path})
        else:
            logger.warning(f"Unsupported file type, skipping: {path}")
    return rows


def _copy_to_tmp(path: str) -> str | None:
    """Copy a PDF/image to a temp file and return the temp path."""
    try:
        suffix = pathlib.Path(path).suffix or ".pdf"
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tf:
            with open(path, "rb") as f:
                data = f.read()
            tf.write(data)
            tmp_path = tf.name

        # Convert images to PDF if needed
        if is_png(tmp_path) or is_jpeg(tmp_path):
            with open(tmp_path, "wb") as f:
                f.write(convert_image_to_pdf_bytes(tmp_path))
        return tmp_path
    except Exception as exc:
        logger.warning(f"Could not copy {path}: {exc}")
        return None


def _expand_tarball(tarball_path: str) -> list[dict]:
    """Extract PDFs from a tarball and return rows."""
    rows = []
    tmp_dir = tempfile.mkdtemp()
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".pdf"):
                    local_path = os.path.join(tmp_dir, os.path.basename(member.name))
                    extracted = tar.extractfile(member)
                    if extracted:
                        with open(local_path, "wb") as f:
                            f.write(extracted.read())
                        rows.append(
                            {
                                "pdf_path": f"{tarball_path}::{member.name}",
                                "local_path": local_path,
                            }
                        )
    except Exception as exc:
        logger.warning(f"Could not expand tarball {tarball_path}: {exc}")
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_olmo_ocr_ray(
    pdf_dir: str,
    workspace: str = DEFAULT_WORKSPACE,
    model: str = DEFAULT_MODEL,
    max_page_retries: int = 3,
    max_page_error_rate: float = 0.004,
    apply_filter: bool = False,
    markdown: bool = True,
    target_longest_image_dim: int = 1288,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 16384,
    max_tokens: int = 8000,
    temperature: float = 0.1,
    tensor_parallel_size: int = 1,
    num_gpu_actors: int = 1,
    num_cpu_actors: int = 4,
    batch_size: int = 32,
) -> list[dict]:
    """Run olmOCR using Ray Data + vLLM offline inference.

    Parameters
    ----------
    pdf_dir:
        Directory containing PDF files to process.
    workspace:
        Output directory for results and markdown files.
    model:
        HuggingFace model id or local path for olmOCR.
    num_gpu_actors:
        Number of parallel GPU replicas for vLLM inference.
    num_cpu_actors:
        Number of CPU replicas for PDF page rendering.
    batch_size:
        Pages per vLLM inference batch.

    Returns
    -------
    List of Dolma document dicts.
    """
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = RayOcrConfig(
        workspace=workspace,
        model=model,
        max_page_retries=max_page_retries,
        max_page_error_rate=max_page_error_rate,
        target_longest_image_dim=target_longest_image_dim,
        apply_filter=apply_filter,
        markdown=markdown,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        temperature=temperature,
        num_gpu_actors=num_gpu_actors,
        num_cpu_actors=num_cpu_actors,
        batch_size=batch_size,
    )

    os.makedirs(workspace, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect input PDF rows
    # ------------------------------------------------------------------
    pdf_paths = glob.glob(str(pathlib.Path(pdf_dir) / "*.pdf"))
    if not pdf_paths:
        logger.info(f"No PDFs found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")
    pdf_rows = _collect_pdf_rows(pdf_paths)
    if not pdf_rows:
        logger.info("No valid PDFs to process")
        return []

    logger.info(f"Prepared {len(pdf_rows)} PDF rows (including tarball contents)")

    # ------------------------------------------------------------------
    # 2. Create Ray Dataset and expand to pages (CPU)
    # ------------------------------------------------------------------
    ds = ray.data.from_items(pdf_rows)

    # Flat-map each PDF row into per-page rows
    page_ds = ds.flat_map(
        lambda row: _expand_pdf_to_pages(
            row,
            target_longest_image_dim=config.target_longest_image_dim,
            apply_filter=config.apply_filter,
        ),
        num_cpus=1,
        concurrency=config.num_cpu_actors,
    )

    # ------------------------------------------------------------------
    # 3. vLLM inference (GPU actors)
    # ------------------------------------------------------------------
    result_ds = page_ds.map_batches(
        OlmOCRPredictor,
        fn_constructor_kwargs={"config": config},
        batch_size=config.batch_size,
        num_gpus=config.tensor_parallel_size,
        concurrency=config.num_gpu_actors,
    )

    # ------------------------------------------------------------------
    # 4. Collect results and group by PDF
    # ------------------------------------------------------------------
    all_rows = result_ds.take_all()

    # Group by pdf_path
    from collections import defaultdict
    pages_by_pdf: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        pages_by_pdf[row["pdf_path"]].append(row)

    # ------------------------------------------------------------------
    # 5. Build Dolma documents and write markdown
    # ------------------------------------------------------------------
    dolma_docs = []
    for pdf_path, page_rows in pages_by_pdf.items():
        num_pages = page_rows[0]["num_pages"] if page_rows else 0
        num_fallback = sum(r["is_fallback"] for r in page_rows)

        if num_pages > 0 and num_fallback / num_pages > config.max_page_error_rate:
            logger.error(
                f"{pdf_path}: {num_fallback}/{num_pages} fallback pages exceed "
                f"max_page_error_rate={config.max_page_error_rate}, discarding"
            )
            continue

        doc = _build_dolma_document(pdf_path, page_rows)
        if doc is None:
            continue

        dolma_docs.append(doc)

        if config.markdown:
            natural_text = _build_markdown_with_page_tags(
                doc["text"], doc["attributes"]["pdf_page_numbers"]
            )
            md_path = _get_markdown_path(config.workspace, pdf_path)
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w") as f:
                f.write(natural_text)

    logger.info(f"Produced {len(dolma_docs)} Dolma documents from {len(pdf_rows)} PDFs")
    return dolma_docs
