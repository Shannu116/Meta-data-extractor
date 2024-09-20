import os
import hashlib
import mimetypes
import exif
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
from PyPDF2 import PdfReader
from docx import Document
import datetime
import platform
import gradio as gr

def get_file_dates(file_path):
    """Get creation and modification dates, accounting for different OS behaviors."""
    stat = os.stat(file_path)
    
    if platform.system() == 'Windows':
        ctime = datetime.datetime.fromtimestamp(stat.st_ctime)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    else:  # Unix-based systems
        try:
            ctime = datetime.datetime.fromtimestamp(stat.st_birthtime)  # macOS and some BSD
        except AttributeError:
            ctime = datetime.datetime.fromtimestamp(stat.st_ctime)  # Linux
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    
    return ctime, mtime

def calculate_file_hash(file_path, hash_algorithm="sha256"):
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            hash_obj = hashlib.new(hash_algorithm, file_data)
            file_hash = hash_obj.hexdigest()
            return file_hash
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."

def get_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    else:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return 'image/' + ext[1:]
        elif ext == '.pdf':
            return 'application/pdf'
        elif ext == '.docx':
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            return 'application/octet-stream'

def get_file_metadata(file_path):
    metadata = {}
    
    # Basic file information
    metadata['File Name'] = os.path.basename(file_path)
    metadata['File Size'] = f"{os.path.getsize(file_path)} bytes"
    metadata['File Type'] = get_file_type(file_path)
    
    # Get accurate creation and modification times
    ctime, mtime = get_file_dates(file_path)
    metadata['Creation Time'] = ctime.strftime('%Y-%m-%d %H:%M:%S')
    metadata['Last Modified'] = mtime.strftime('%Y-%m-%d %H:%M:%S')
    
    metadata['SHA256 Hash'] = calculate_file_hash(file_path)
    
    # Specific file type metadata
    if metadata['File Type'].startswith('image'):
        metadata.update(get_image_metadata(file_path))
    elif metadata['File Type'] == 'application/pdf':
        metadata.update(get_pdf_metadata(file_path))
    elif metadata['File Type'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        metadata.update(get_docx_metadata(file_path))
    
    return metadata

def get_image_metadata(file_path):
    image_metadata = {}
    try:
        # EXIF data using exif library
        with open(file_path, 'rb') as img_file:
            img = exif.Image(img_file)
            if img.has_exif:
                image_metadata['Camera Make'] = img.get('make', 'N/A')
                image_metadata['Camera Model'] = img.get('model', 'N/A')
                image_metadata['Date Taken'] = img.get('datetime', 'N/A')
                image_metadata['GPS Info'] = f"{img.get('gps_latitude', 'N/A')}, {img.get('gps_longitude', 'N/A')}"
        
        # Additional metadata using PIL
        with Image.open(file_path) as img:
            image_metadata['Resolution'] = f"{img.width}x{img.height}"
            image_metadata['Color Mode'] = img.mode
            pil_info = {TAGS.get(tag, tag): str(value) for tag, value in img.getexif().items()}
            image_metadata.update(pil_info)
        
        # More detailed EXIF data using exifread
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
            exif_data = {tag: str(tags[tag]) for tag in tags if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote')}
            image_metadata.update(exif_data)
        
        # Analyze for potential Photoshop editing
        software_used = image_metadata.get('Software', 'Unknown')
        if 'Adobe Photoshop' in software_used:
            image_metadata['Forensic Note'] = "WARNING: Image may have been edited with Adobe Photoshop"
    except Exception as e:
        image_metadata['Error'] = str(e)
    return image_metadata

def get_pdf_metadata(file_path):
    pdf_metadata = {}
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf = PdfReader(pdf_file)
            info = pdf.metadata
            pdf_metadata['Author'] = info.author if info.author else 'N/A'
            pdf_metadata['Creator'] = info.creator if info.creator else 'N/A'
            pdf_metadata['Producer'] = info.producer if info.producer else 'N/A'
            pdf_metadata['Subject'] = info.subject if info.subject else 'N/A'
            pdf_metadata['Title'] = info.title if info.title else 'N/A'
            pdf_metadata['Creation Date'] = info.creation_date if info.creation_date else 'N/A'
            pdf_metadata['Modification Date'] = info.modification_date if info.modification_date else 'N/A'
            pdf_metadata['Number of Pages'] = len(pdf.pages)
    except Exception as e:
        pdf_metadata['Error'] = str(e)
    return pdf_metadata

def get_docx_metadata(file_path):
    docx_metadata = {}
    try:
        doc = Document(file_path)
        core_properties = doc.core_properties
        docx_metadata['Author'] = core_properties.author if core_properties.author else 'N/A'
        docx_metadata['Created'] = core_properties.created.strftime('%Y-%m-%d %H:%M:%S') if core_properties.created else 'N/A'
        docx_metadata['Last Modified By'] = core_properties.last_modified_by if core_properties.last_modified_by else 'N/A'
        docx_metadata['Last Printed'] = core_properties.last_printed.strftime('%Y-%m-%d %H:%M:%S') if core_properties.last_printed else 'N/A'
        docx_metadata['Title'] = core_properties.title if core_properties.title else 'N/A'
        docx_metadata['Subject'] = core_properties.subject if core_properties.subject else 'N/A'
        docx_metadata['Keywords'] = core_properties.keywords if core_properties.keywords else 'N/A'
        docx_metadata['Comments'] = core_properties.comments if core_properties.comments else 'N/A'
        docx_metadata['Category'] = core_properties.category if core_properties.category else 'N/A'
        docx_metadata['Number of Pages'] = doc.element.body.get_page_count()
    except Exception as e:
        docx_metadata['Error'] = str(e)
    return docx_metadata

def extract_metadata(file):
    if file is None:
        return "No file uploaded. Please upload a file to extract metadata."
    
    file_path = file.name
    metadata = get_file_metadata(file_path)
    
    result = "\nForensic Metadata Analysis Results:\n"
    result += "====================================\n"
    for key, value in metadata.items():
        if isinstance(value, dict):
            result += f"{key}:\n"
            for sub_key, sub_value in value.items():
                result += f"  {sub_key}: {sub_value}\n"
        else:
            result += f"{key}: {value}\n"
    
    return result

# Gradio interface
iface = gr.Interface(
    fn=extract_metadata,
    inputs=gr.File(label="Upload File for Forensic Analysis"),
    outputs=gr.Textbox(label="Forensic Metadata Analysis Results", lines=20),
    title="Comprehensive Forensic Metadata Analyzer",
    description="Upload a file (images, PDFs, DOCX) for detailed forensic metadata analysis. This tool extracts extensive metadata, calculates file hashes, and identifies potential signs of manipulation."
)

if __name__ == "__main__":
    iface.launch()