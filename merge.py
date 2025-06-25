import PyPDF2

def merge_pdfs(pdf1_path, pdf2_path, output_path):
    merger = PyPDF2.PdfMerger()

    # Append the two PDFs
    merger.append(pdf1_path)
    merger.append(pdf2_path)

    # Write out the merged PDF
    with open(output_path, 'wb') as output_file:
        merger.write(output_file)

    print(f"Merged PDF saved as: {output_path}")

# Example usage
merge_pdfs('1.pdf', '2.pdf', 'Ovidiu_Burcea-Raport_preliminar_short-semnat.pdf')
