# utils/file_handler.py

import logging
import traceback
import streamlit as st
import requests
import os
import json

from database.weav_operations import WeaviateManager

weav_manager = WeaviateManager()

# --- Configuration ---
# Set the API endpoint URL from an environment variable for flexibility,
# with a local default for development.
PROCESSING_URL = "http://localhost:8002"
PROCESS_DOCUMENT_ENDPOINT = f"{PROCESSING_URL}/process-document/"

# --- Logger Setup ---
logger = logging.getLogger(__name__)


def process_document_upload(uploaded_file, components, settings, user_id):
    """Process a single uploaded document and store it in the database."""
    try:
        # Process file content

        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        # You can pass additional metadata to the API if needed.
        payload = {"user_id": user_id}

        # Send the POST request to the processing microservice with a timeout.
        response = requests.post(
            PROCESS_DOCUMENT_ENDPOINT, files=files, data=payload, timeout=300
        )  # 5-minute timeout

        # Raise an exception for bad status codes (4xx or 5xx).
        response.raise_for_status()

        # Process the successful JSON response from the server.
        try:
            result = response.json()

            # Store document metadata in database
            document_id = components["doc_ops"].insert_document(
                document_name=result["filename"],
                user_id=user_id,
                document_text=str(result["extracted_text"])[:1000],
                file_size=result["file_size_bytes"],
                file_type=result["file_type"],
            )

            # Create chunks
            structured_chunks = components["document_chunker"].create_structured_chunks(
                text=result["extracted_text"],
                document_name=result["filename"],
                document_id=document_id,
                user_id=user_id,
                file_type=result["file_type"],
            )

            all_child_chunks_for_batch = []

            try:
                for parent_with_children in structured_chunks:
                    # Separate the children from the parent data
                    child_chunks_data = parent_with_children["child_chunks"]
                    print("total children:", len(child_chunks_data))
                    parent_uuid = weav_manager.insert_parent_chunk(
                        document_id=parent_with_children["document_id"],
                        user_id=parent_with_children["user_id"],
                        chunk_text=parent_with_children["chunk_text"],
                        contextual_header=parent_with_children["contextual_header"],
                        chunk_index=parent_with_children["chunk_index"],
                        document_name=parent_with_children["document_name"],
                        file_type=parent_with_children["file_type"],
                    )
                    for child_chunk in child_chunks_data:
                        child_chunk["parent_chunk_id"] = parent_uuid

                    # Add the processed children to a list for batch insertion
                    all_child_chunks_for_batch.extend(child_chunks_data)

                print("total child chunks to insert:", len(all_child_chunks_for_batch))

                # Batch insert all the child chunks for the document at once
                if all_child_chunks_for_batch:
                    weav_manager.batch_insert_chunks(
                        all_child_chunks_for_batch, chunk_type="child"
                    )

                logging.info(
                    f"Successfully processed and uploaded document ID: {document_id}"
                )

            except Exception as e:
                logging.error(f"Failed to process document ID {document_id}: {e}")
            # finally:
            # 7. Close the Weaviate client connection
            # weav_manager.close()

            # Mark document as fully processed
            components["doc_ops"].mark_document_processed(document_id)

            st.success(f"✅ Successfully processed {uploaded_file.name}")

            return True
        except json.JSONDecodeError:
            st.error("Received a non-JSON response from the server.")
            logger.error(f"Could not decode JSON from response: {response.text}")
            return False

    except Exception as e:
        error_msg = f"Error processing document {uploaded_file.name}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")
        return False
