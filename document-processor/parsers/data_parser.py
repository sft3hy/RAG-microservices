# parsers/data_parser.py

import logging
from typing import Optional
from ocr_engine import OCRProcessor
import io

logger = logging.getLogger(__name__)


def extract_text_from_image(
    self,
    file_content: bytes,
    filename: str,
) -> Optional[str]:
    """Extract text from CSV files."""
    try:
        import pandas as pd

        # Detect encoding
        encoding = self.detect_encoding(file_content)

        # Try different separators
        separators = [",", ";", "\t", "|"]

        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_content), encoding=encoding, sep=sep, nrows=5
                )
                if len(df.columns) > 1:
                    # Read the full file
                    df = pd.read_csv(
                        io.BytesIO(file_content), encoding=encoding, sep=sep
                    )

                    # Convert to text format
                    text_content = f"CSV File: {filename}\n\n"
                    text_content += f"Columns: {' | '.join(df.columns.tolist())}\n\n"

                    # Add sample of data
                    for idx, row in df.iterrows():
                        row_text = " | ".join(
                            [str(val) if pd.notna(val) else "" for val in row.values]
                        )
                        text_content += row_text + "\n"

                        # Limit to prevent huge files
                        if idx >= 1000:
                            text_content += f"\n... (truncated, showing first 1000 rows of {len(df)} total)\n"
                            break

                    return text_content

            except Exception:
                continue

        # If all separators fail, treat as plain text
        return file_content.decode(encoding, errors="ignore")

    except Exception as e:
        logger.error(f"Failed to extract text from CSV {filename}: {e}")
        return None
