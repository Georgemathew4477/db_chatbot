import shutil, sys, streamlit as st

def which(binname):
    import shutil
    p = shutil.which(binname)
    return p or "NOT FOUND"

st.title("Environment Debug")

st.write({
    "python": sys.version,
    "ffmpeg": which("ffmpeg"),
    "tesseract": which("tesseract"),
    "pdftotext": which("pdftotext"),
    "pdftoppm": which("pdftoppm"),
})

try:
    import pytesseract
    st.write({"pytesseract_version": getattr(pytesseract, "__version__", "n/a")})
except Exception as e:
    st.write({"pytesseract_import_error": str(e)})
