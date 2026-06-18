"""Build Docling converters for PDF ingestion."""

from __future__ import annotations

try:  # pragma: no cover - import-time dependency guard
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise RuntimeError(
        "Docling PDF support requires the docling package to be installed. "
        "Install it with `python -m pip install -r requirements.txt`."
    ) from exc


def build_pdf_pipeline_options(
    *,
    image_resolution_scale: float,
    enable_picture_description: bool,
    vlm_url: str | None = None,
    vlm_model_name: str | None = None,
    vlm_timeout: int | None = None,
    vlm_prompt: str | None = None,
) -> PdfPipelineOptions:
    """Create PDF pipeline options for Docling conversion."""
    pdf_options = PdfPipelineOptions(
        images_scale=image_resolution_scale,
        do_picture_description=enable_picture_description,
    )

    if enable_picture_description:
        resolved_vlm_url = (vlm_url or "").rstrip("/")
        if not resolved_vlm_url:
            raise ValueError(
                "document_vlm_url is required when document_enable_picture_description is enabled."
            )

        picture_description_options = PictureDescriptionApiOptions(
            url=f"{resolved_vlm_url}/v1/chat/completions",
            prompt=vlm_prompt or "Describe this image in sentences in a single paragraph.",
            params={"model": vlm_model_name or "llava"},
            headers={"Authorization": "Bearer not-needed"},
            timeout=vlm_timeout or 60,
        )

        pdf_options.generate_picture_images = True
        pdf_options.picture_description_options = picture_description_options
        pdf_options.enable_remote_services = True

    return pdf_options


def build_document_converter(
    *,
    image_resolution_scale: float,
    enable_picture_description: bool,
    vlm_url: str | None = None,
    vlm_model_name: str | None = None,
    vlm_timeout: int | None = None,
    vlm_prompt: str | None = None,
) -> DocumentConverter:
    """Create a PDF-only Docling document converter."""
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=build_pdf_pipeline_options(
                    image_resolution_scale=image_resolution_scale,
                    enable_picture_description=enable_picture_description,
                    vlm_url=vlm_url,
                    vlm_model_name=vlm_model_name,
                    vlm_timeout=vlm_timeout,
                    vlm_prompt=vlm_prompt,
                )
            )
        },
    )
