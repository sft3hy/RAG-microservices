import re
from typing import List, Dict, Tuple
from langchain.text_splitter import MarkdownTextSplitter
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(
        self,
        child_chunk_size: int = 250,
        parent_chunk_size: int = 2500,
        contextual_header_size: int = 100,
        chunk_overlap: int = 50,
    ):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.contextual_header_size = contextual_header_size
        self.chunk_overlap = chunk_overlap

        # Initialize markdown splitters
        self.parent_splitter = MarkdownTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
        )

        self.child_splitter = MarkdownTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
        )

    def convert_to_markdown(self, text: str, document_name: str) -> str:
        """Convert plain text to markdown format."""
        # Add document title as main header
        markdown_text = f"# {document_name}\n\n"

        # Split into paragraphs and add appropriate formatting
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if paragraph looks like a heading
            if len(para) < 100 and not para.endswith("."):
                markdown_text += f"## {para}\n\n"
            else:
                markdown_text += f"{para}\n\n"

        return markdown_text

    def generate_contextual_header(
        self, document_name: str, section_header: str = ""
    ) -> str:
        """Generate contextual header for chunks."""
        if section_header:
            header = f"{document_name} > {section_header}"
        else:
            header = document_name

        # Truncate to max size
        if len(header) > self.contextual_header_size:
            header = header[: self.contextual_header_size - 3] + "..."

        return header

    def extract_section_header(self, chunk: str) -> str:
        """Extract section header from chunk if available."""
        lines = chunk.split("\n")
        for line in lines:
            line = line.strip()
            # Look for markdown headers
            if line.startswith("#"):
                return line.lstrip("#").strip()
            # Look for lines that might be headers (short, no period)
            elif len(line) < 80 and line and not line.endswith("."):
                return line

        return ""

    def create_structured_chunks(
        self,
        text: str,
        document_name: str,
        document_id: int,
        user_id: str,
        file_type: str,
    ) -> List[Dict]:
        """
        Creates a nested structure of parent chunks, each containing its child chunks.
        This makes it easier to insert into a database and establish parent-child relationships.

        Returns:
            A list of parent chunk dictionaries. Each dictionary contains a 'child_chunks' key
            with a list of its corresponding child chunk dictionaries.
        """
        markdown_text = self.convert_to_markdown(text, document_name)
        parent_texts = self.parent_splitter.split_text(markdown_text)

        structured_chunks = []
        child_index_counter = 0

        for i, parent_text in enumerate(parent_texts):
            parent_section_header = self.extract_section_header(parent_text)
            parent_contextual_header = self.generate_contextual_header(
                document_name, parent_section_header
            )

            parent_chunk_data = {
                "document_id": document_id,
                "user_id": user_id,
                "document_name": document_name,
                "file_type": file_type,
                "chunk_text": parent_text,
                "contextual_header": parent_contextual_header,
                "chunk_index": i,
                "child_chunks": [],  # This will hold the children
            }

            # Create child chunks from this parent's text
            child_texts = self.child_splitter.split_text(parent_text)

            for child_text in child_texts:
                # A child's header is either its own or inherited from its parent
                child_section_header = (
                    self.extract_section_header(child_text) or parent_section_header
                )

                child_contextual_header = self.generate_contextual_header(
                    document_name, child_section_header
                )

                child_chunk_data = {
                    "document_id": document_id,
                    "user_id": user_id,
                    "document_name": document_name,
                    "file_type": file_type,
                    "chunk_text": child_text,
                    "contextual_header": child_contextual_header,
                    "chunk_index": child_index_counter,
                    # Note: parent_chunk_id is intentionally omitted.
                    # It must be added after the parent is inserted into Weaviate.
                }
                parent_chunk_data["child_chunks"].append(child_chunk_data)
                child_index_counter += 1

            structured_chunks.append(parent_chunk_data)

        total_children = sum(len(p["child_chunks"]) for p in structured_chunks)
        logger.info(
            f"Created {len(structured_chunks)} parent chunks and {total_children} child chunks"
        )
        return structured_chunks
