from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document,
)
from retrieval import (
    build_vector_database,
    create_embedding_matrix,
    load_vector_database,
    save_vector_database,
    search_documents,
)


VECTOR_DATABASE_PATH = Path(
    "vector_database/documents.json"
)

# Current replacement recommended by Groq for the older
# llama-3.1-8b-instant model.
GENERATION_MODEL = "openai/gpt-oss-20b"


def create_groq_client() -> Groq:
    """Create the Groq API client."""

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        st.error(
            "GROQ_API_KEY was not found. "
            "Add it to the .env file."
        )
        st.stop()

    return Groq(
        api_key=api_key
    )


def format_reference(
    record: dict[str, Any],
) -> str:
    """Create a readable document reference."""

    locator_type = record[
        "locator_type"
    ].capitalize()

    locator_number = record[
        "locator_number"
    ]

    return (
        f"{record['document']}, "
        f"{locator_type} {locator_number}"
    )


def create_context(
    retrieved_records: list[dict[str, Any]],
) -> str:
    """Format retrieved chunks for the LLM."""

    context_sections = []

    for source_number, record in enumerate(
        retrieved_records,
        start=1,
    ):
        reference = format_reference(
            record
        )

        section = f"""
SOURCE {source_number}
Reference: {reference}
Content:
{record["text"]}
"""

        context_sections.append(
            section.strip()
        )

    return "\n\n".join(
        context_sections
    )


def generate_answer(
    client: Groq,
    question: str,
    retrieved_records: list[dict[str, Any]],
) -> str:
    """Generate a grounded answer using Groq."""

    if not retrieved_records:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    context = create_context(
        retrieved_records
    )

    system_message = """
You are an AI document assistant.

Answer questions using only the document extracts supplied
by the application.

Rules:

1. Do not use outside knowledge.
2. Do not invent names, facts, policies, dates or requirements.
3. Every important factual claim must include a reference.
4. Copy references exactly as they appear in the source extracts.
5. Use the citation format:
   [Document name, Page 4]
   [Document name, Slide 7]
   [Document name, Paragraph 12]
6. Do not cite a document location that was not supplied.
7. If the extracts do not contain the answer, state:
   "I could not find enough information in the uploaded documents."
8. If sources conflict, explain the conflict and cite both.
9. Keep the response concise and easy to understand.
10. Do not exceed 250 words unless the user requests detail.
"""

    user_message = f"""
DOCUMENT EXTRACTS

{context}

USER QUESTION

{question}

Answer using only the extracts above.
Include document references for the answer.
"""

    completion = (
        client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=0.1,
            max_completion_tokens=500,
        )
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    if not answer:
        return "No answer was generated."

    return answer


def initialise_session_state() -> None:
    """Initialise application variables."""

    if "vector_database" not in st.session_state:
        st.session_state.vector_database = (
            load_vector_database(
                VECTOR_DATABASE_PATH
            )
        )

    if "embedding_matrix" not in st.session_state:
        st.session_state.embedding_matrix = (
            create_embedding_matrix(
                st.session_state.vector_database
            )
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "document_names" not in st.session_state:
        names = {
            record["document"]
            for record in st.session_state.vector_database
        }

        st.session_state.document_names = sorted(
            names
        )


def process_uploaded_documents(
    uploaded_files,
) -> None:
    """Read uploaded documents and build the knowledge base."""

    all_records = []
    failed_files = []

    extraction_progress = st.progress(0)
    extraction_text = st.empty()

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        try:
            extraction_text.write(
                f"Reading {uploaded_file.name}..."
            )

            records = process_document(
                file_bytes=uploaded_file.getvalue(),
                document_name=uploaded_file.name,
            )

            all_records.extend(
                records
            )

        except Exception as error:
            failed_files.append(
                f"{uploaded_file.name}: {error}"
            )

        extraction_progress.progress(
            index / len(uploaded_files)
        )

    if not all_records:
        st.error(
            "No searchable text was found in "
            "the uploaded documents."
        )

        if failed_files:
            st.write(failed_files)

        return

    extraction_text.write(
        f"Created {len(all_records)} text chunks."
    )

    embedding_progress = st.progress(0)
    embedding_text = st.empty()

    def update_progress(
        completed: int,
        total: int,
    ) -> None:
        percentage = completed / total

        embedding_progress.progress(
            percentage
        )

        embedding_text.write(
            f"Creating local embeddings: "
            f"{completed}/{total} "
            f"({percentage * 100:.1f}%)"
        )

    vector_database = build_vector_database(
        records=all_records,
        progress_callback=update_progress,
    )

    save_vector_database(
        vector_database=vector_database,
        file_path=VECTOR_DATABASE_PATH,
    )

    st.session_state.vector_database = (
        vector_database
    )

    st.session_state.embedding_matrix = (
        create_embedding_matrix(
            vector_database
        )
    )

    st.session_state.document_names = sorted(
        {
            record["document"]
            for record in vector_database
        }
    )

    st.session_state.chat_history = []

    embedding_progress.progress(1.0)

    embedding_text.success(
        "Document processing completed."
    )

    st.success(
        f"Processed {len(uploaded_files)} file(s) "
        f"and created {len(vector_database)} "
        f"searchable chunks."
    )

    if failed_files:
        with st.expander(
            "Files that could not be processed"
        ):
            for failure in failed_files:
                st.warning(failure)


def clear_knowledge_base() -> None:
    """Delete the current document collection."""

    st.session_state.vector_database = []
    st.session_state.embedding_matrix = None
    st.session_state.chat_history = []
    st.session_state.document_names = []

    if VECTOR_DATABASE_PATH.exists():
        VECTOR_DATABASE_PATH.unlink()


def display_sources(
    records: list[dict[str, Any]],
) -> None:
    """Show evidence used for an answer."""

    with st.expander(
        "View retrieved evidence"
    ):
        for number, record in enumerate(
            records,
            start=1,
        ):
            reference = format_reference(
                record
            )

            st.markdown(
                f"#### Source {number}: {reference}"
            )

            st.caption(
                f"Similarity score: "
                f"{record['similarity']:.3f}"
            )

            st.write(
                record["text"]
            )

            st.divider()


def display_chat_history() -> None:
    """Display previous conversation messages."""

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                display_sources(
                    message["sources"]
                )


def main() -> None:
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📚",
        layout="wide",
    )

    initialise_session_state()
    client = create_groq_client()

    st.title("📚 AI Document Assistant")

    st.write(
        "Upload PDF, Word, PowerPoint, text or "
        "Markdown documents and ask questions "
        "about their contents."
    )

    with st.sidebar:
        st.header("Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[
                extension.replace(".", "")
                for extension in SUPPORTED_EXTENSIONS
            ],
            accept_multiple_files=True,
        )

        process_button = st.button(
            "Process Uploaded Documents",
            type="primary",
            use_container_width=True,
        )

        if process_button:
            if not uploaded_files:
                st.warning(
                    "Please upload at least one document."
                )
            else:
                process_uploaded_documents(
                    uploaded_files
                )

        clear_button = st.button(
            "Clear Knowledge Base",
            use_container_width=True,
        )

        if clear_button:
            clear_knowledge_base()

            st.success(
                "The knowledge base was cleared."
            )

            st.rerun()

        st.divider()

        st.metric(
            "Searchable chunks",
            len(
                st.session_state.vector_database
            ),
        )

        if st.session_state.document_names:
            st.subheader(
                "Processed documents"
            )

            for document_name in (
                st.session_state.document_names
            ):
                st.write(
                    f"✓ {document_name}"
                )
        else:
            st.info(
                "No documents have been processed."
            )

    display_chat_history()

    question = st.chat_input(
        "Ask a question about your documents"
    )

    if not question:
        return

    if not st.session_state.vector_database:
        st.warning(
            "Upload and process documents before "
            "asking a question."
        )
        return

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        retrieval_start = time.perf_counter()

        retrieved_records = search_documents(
            question=question,
            vector_database=(
                st.session_state.vector_database
            ),
            embedding_matrix=(
                st.session_state.embedding_matrix
            ),
            top_k=4,
            minimum_score=0.15,
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        generation_start = time.perf_counter()

        answer = generate_answer(
            client=client,
            question=question,
            retrieved_records=retrieved_records,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_records,
        }

        st.session_state.chat_history.append(
            assistant_message
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

            st.caption(
                f"Document search: "
                f"{retrieval_time:.2f} seconds | "
                f"Answer generation: "
                f"{generation_time:.2f} seconds"
            )

            display_sources(
                retrieved_records
            )

    except Exception as error:
        error_text = str(error)

        if "401" in error_text:
            st.error(
                "The Groq API key is invalid. "
                "Check GROQ_API_KEY in the .env file."
            )

        elif "429" in error_text:
            st.error(
                "The Groq rate limit has been reached. "
                "Wait briefly and try again."
            )

        elif "413" in error_text:
            st.error(
                "Too much text was sent to Groq. "
                "Reduce top_k or the chunk size."
            )

        else:
            st.error(
                f"An error occurred: {error}"
            )


if __name__ == "__main__":
    main()