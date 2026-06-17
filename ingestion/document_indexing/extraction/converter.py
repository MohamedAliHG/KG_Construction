"""Build Docling converters for PDF ingestion."""

from __future__ import annotations

from importlib import import_module


def build_pdf_pipeline_options(
    *,
    image_resolution_scale: float,
    enable_picture_description: bool,
):
    try:
        docling_pipeline_options = import_module("docling.datamodel.pipeline_options")
        PdfPipelineOptions = getattr(docling_pipeline_options, "PdfPipelineOptions")
    except Exception as exc:  # pragma: no cover - import-time dependency guard
        raise RuntimeError(
            "Docling PDF support requires the docling package to be installed. "
            "Install it with `python -m pip install -r requirements.txt`."
        ) from exc

    pdf_options = PdfPipelineOptions()
    pdf_options.images_scale = image_resolution_scale
    pdf_options.do_picture_description = enable_picture_description

    return pdf_options


def build_document_converter(
    *,
    image_resolution_scale: float,
    enable_picture_description: bool,
):
    try:
        docling_document_converter = import_module("docling.document_converter")
        DocumentConverter = getattr(docling_document_converter, "DocumentConverter")
        InputFormat = getattr(docling_document_converter, "InputFormat")
        PdfFormatOption = getattr(docling_document_converter, "PdfFormatOption")
    except Exception as exc:  # pragma: no cover - import-time dependency guard
        raise RuntimeError(
            "Docling PDF support requires the docling package to be installed. "
            "Install it with `python -m pip install -r requirements.txt`."
        ) from exc

    try:
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=build_pdf_pipeline_options(
                        image_resolution_scale=image_resolution_scale,
                        enable_picture_description=enable_picture_description,
                    )
                )
            },
        )
    except TypeError:
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
        )
    return converter
