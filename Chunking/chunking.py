import os
import re
import json

# Configuration
# Use relative paths for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of Legal-RAG-Document
SOURCE_DIR = os.path.join(BASE_DIR, "Held Section") 
OUTPUT_FILE = os.path.join(BASE_DIR, "Chunking", "chunks.json")

def clean_text(text):
    """
    Removes newlines and extra spaces to form a single coherent string.
    """
    return " ".join(text.split())

def split_into_sentences(text):
    """
    Basic sentence splitter.
    Splits by period followed by space, or other end marks.
    This is a simple implementation; for production, use nltk or spacy.
    """
    # Split by common sentence terminators (. ? !) followed by a space or end of string.
    # We keep the delimiter in the capture group to re-attach it.
    parts = re.split(r'([.?!])\s+', text)
    sentences = []
    
    # Reassemble
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i] + parts[i+1] # Attach punctuation
        sentences.append(sent)
    
    # Handle last part if any
    if len(parts) % 2 != 0 and parts[-1]:
         sentences.append(parts[-1])
         
    return sentences

def create_sub_chunks(text, max_chars=1000):
    """
    Splits a large text block into smaller chunks based on sentences,
    respecting a character limit.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        # If adding this sentence exceeds max_chars and we strictly want to break...
        # But usually we prefer to just fill up the bucket.
        if current_len + len(sentence) > max_chars and current_chunk:
            # Commit current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def parse_text_file(filepath):
    """
    Generator that parses a single text file and yields chunk objects.
    """
    filename = os.path.basename(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Strategy Selection
    has_bullets = '■' in content
    
    raw_blocks = []
    
    if has_bullets:
        # Strategy A: Split by Bullet '■'
        # Treat each bulleted section as a logical block.
        segments = content.split('■')
        for segment in segments:
            if not segment.strip():
                continue
            
            # Find Para ID
            matches = re.findall(r'(\[Para[^\]]+\])', segment, re.IGNORECASE | re.DOTALL)
            para_id = matches[-1] if matches else "N/A"
            
            # Remove the tag from content to clean it up
            content_text = segment
            if para_id != "N/A":
                content_text = content_text.replace(para_id, "")
                para_id = " ".join(para_id.split())
            
            raw_blocks.append((content_text, para_id))
            
    else:
        # Strategy B: Split by [Para ...] tags
        parts = re.split(r'(\[Para[^\]]+\])', content, flags=re.IGNORECASE | re.DOTALL)
        
        current_text = parts[0]
        
        for i in range(1, len(parts), 2):
            tag = parts[i]
            para_id = " ".join(tag.split())
            raw_blocks.append((current_text, para_id))
            
            if i + 1 < len(parts):
                current_text = parts[i+1]
            else:
                current_text = ""
                
        if current_text.strip():
             raw_blocks.append((current_text, "N/A"))

    # Process Blocks into Chunks
    for text_content, para_id in raw_blocks:
        clean_content = clean_text(text_content)
        if not clean_content:
            continue
            
        # Micro-Segmentation
        sub_chunk_texts = create_sub_chunks(clean_content, max_chars=1200)
        
        for sub_text in sub_chunk_texts:
            chunk = {
                "content": sub_text,
                "metadata": {
                    "source": filename,
                    "para_id": para_id
                }
            }
            yield chunk

def main():
    print(f"Scanning directory: {SOURCE_DIR}")
    
    # Check if directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory not found at {SOURCE_DIR}")
        return

    # Counter
    count = 0
    
    # Open output file for streaming write
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            # We will use JSONL format (one valid JSON object per line)
            # This is much more memory efficient for large datasets than a single JSON array.
            
            # Walk through all files
            for root, dirs, files in os.walk(SOURCE_DIR):
                for file in files:
                    if file.lower().endswith('.txt'):
                        fullpath = os.path.join(root, file)
                        print(f"Processing: {file}")
                        
                        # Use generator to get chunks one by one
                        for chunk in parse_text_file(fullpath):
                            # Dump single chunk as a line
                            json.dump(chunk, f_out, ensure_ascii=False)
                            f_out.write("\n")
                            count += 1
            
        print(f"Generated {count} chunks from all files.")
        print(f"Chunks saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Fatal Error: {e}")

if __name__ == "__main__":
    main()
