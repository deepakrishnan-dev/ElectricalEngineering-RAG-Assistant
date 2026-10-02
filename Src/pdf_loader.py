# extraction of text from pdf file


from pypdf import PdfReader

# function for extracting text from pdf file
def Loadpdf(file_path):
    reader = PdfReader(file_path)

    pages = []  #store pages of pdf file

    for page in reader.pages:
        text = page.extract_text()         # extract text from each page        
        pages.append(text)

    return pages

