import fitz  # PyMuPDF
import os
import csv

# Define the folder containing PDFs
pdf_folder = "Add/Path/To/PDFs/Folder"  # Change this to your PDF folder path
max_size_mb = 10  # Maximum file size in MB
log_file = os.path.join(pdf_folder, "compression_log.csv")  # CSV Log file

def compress_pdf(input_path, output_path):
    """Compress a PDF file by reducing image quality and size."""
    try:
        doc = fitz.open(input_path)
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
        return True
    except Exception as e:
        return f"ERROR: {e}"

def process_pdfs(pdf_folder, max_size_mb):
    """Compress PDFs in a folder if they exceed max_size_mb and log results to CSV."""
    log_data = [["Filename", "Original Size (MB)", "New Size (MB)", "Status"]]
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(pdf_folder, filename)
            temp_output_path = os.path.join(pdf_folder, f"temp_{filename}")

            file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                print(f"Working on it: Compressing {filename} ({file_size_mb:.2f} MB)...")
                
                # Compress PDF
                result = compress_pdf(input_path, temp_output_path)

                # Ensure temp file was created before checking its size
                if os.path.exists(temp_output_path):
                    new_size_mb = os.path.getsize(temp_output_path) / (1024 * 1024)

                    if new_size_mb <= max_size_mb:
                        os.replace(temp_output_path, input_path)  # Overwrite original file
                        print(f"DONE: Reduced {filename} to {new_size_mb:.2f} MB")
                        log_data.append([filename, f"{file_size_mb:.2f}", f"{new_size_mb:.2f}", "Compressed"])
                    else:
                        print(f"ERROR: Compression couldn't reduce {filename} below {max_size_mb}MB")
                        os.remove(temp_output_path)  # Remove temporary file
                        log_data.append([filename, f"{file_size_mb:.2f}", "-", "ERROR: Still too large"])
                else:
                    print(f"ERROR: Compression failed, temp file was not created.")
                    log_data.append([filename, f"{file_size_mb:.2f}", "-", "ERROR: Compression Failed"])
            else:
                print(f"OK: Skipping {filename}, already below {max_size_mb}MB.")
                log_data.append([filename, f"{file_size_mb:.2f}", "-", "OK: Skipped"])

    # Save results to CSV
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(log_data)
    
    print(f"\n FINISHED RUNNING THE SCRIPT: Compression log saved to: {log_file}")

# Run the compression
process_pdfs(pdf_folder, max_size_mb)