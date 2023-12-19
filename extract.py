import io
import docx
import os


corpus = io.open("corpus_info.txt", "w", encoding="utf-8")


for file in os.listdir('data'):
    if file.split('.')[-1] == "docx":
        document = docx.Document("data/"+file)
        print("\nDOCX")
        text = ""

        for i, page in enumerate(document.paragraphs):
            print(f"PAGE {i+1}")
            text += page.text + "\n\n"

        text += "\n\n"

        corpus.write(text)

    if file.split('.')[-1] == "txt":
        f = open(f"data/{file}", "r")
        print("\nTXT")
        corpus.write(f.read() + "\n\n")
