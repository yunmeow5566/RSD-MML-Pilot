from pathlib import Path

import easyocr


def get_reader(languages:list[str] = ['en']) -> easyocr.Reader:
    # Create and return an EasyOCR reader for the specified languages.
    # For the first time, this will download the necessary model files.
    return easyocr.Reader(languages)

def get_text_from_image(image_path: str, reader: easyocr.Reader) -> list[str]:
    # Use the provided EasyOCR reader to extract text from the specified image.
    # The function returns a list of strings.
    # Note that if you remove 'detail=0', this function can return more details.
    return reader.readtext(image_path, detail=0)

reader = get_reader()

script_path = Path(__file__).resolve()
    
result = get_text_from_image(str(script_path.parent / 'input' / 'example01.jpg'), reader)
print(result)