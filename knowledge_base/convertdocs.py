import os
import pypandoc
import pdfplumber

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "raw_docs")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "knowledge_base")

def convert_docx_to_md(input_path, output_path):
    try:
        import pypandoc

        try:
            pypandoc.get_pandoc_path()
        except OSError:
            print("Downloading pandoc...")
            pypandoc.download_pandoc()

        pypandoc.convert_file(
            input_path,
            "md",
            outputfile=output_path,
            extra_args=["--standalone"]
        )

        print(f"[DOCX] Converted: {input_path} -> {output_path}")

    except Exception as e:
        print(f"[ERROR DOCX] {input_path}: {e}")


def ensure_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def clean_text(text: str) -> str:
    """
    Cleaning sederhana supaya lebih rapi untuk RAG
    """
    if not text:
        return ""

    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    text = text.replace("•", "-")

    return text


def convert_docx_to_md(input_path, output_path):
    try:
        pypandoc.convert_file(
            input_path,
            "md",
            outputfile=output_path,
            extra_args=["--standalone"]
        )
        print(f"[DOCX] Converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"[ERROR DOCX] {input_path}: {e}")


def convert_pdf_to_md(input_path, output_path):
    try:
        text = []

        with pdfplumber.open(input_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text.append(content)

        full_text = "\n".join(text)
        full_text = clean_text(full_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[PDF] Converted: {input_path} -> {output_path}")

    except Exception as e:
        print(f"[ERROR PDF] {input_path}: {e}")


def process_files():
    ensure_output_folder()

    for file in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, file)

        if file.lower().endswith(".docx"):
            output_name = file.replace(".docx", ".md")
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            convert_docx_to_md(input_path, output_path)

        elif file.lower().endswith(".pdf"):
            output_name = file.replace(".pdf", ".md")
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            convert_pdf_to_md(input_path, output_path)

        else:
            print(f"[SKIP] Unsupported file: {file}")


if __name__ == "__main__":
    process_files()
    print("\n✅ All documents converted!")