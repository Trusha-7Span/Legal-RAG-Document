import os
import re
import json

# Configuration
SOURCE_DIR = r"c:\Users\Trusha Khachariya\OneDrive\Desktop\Held Section"
OUTPUT_FILE = r"c:\Users\Trusha Khachariya\OneDrive\Desktop\Held Section\chunks.json"

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
    current_sent = ""
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
    filename = os.path.basename(filepath)
    chunks = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

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
            # We look for [Para ...] sequence. 
            # Use findall to get all, pick the last one as the block's label if multiple exist.
            # Regex matches [Para followed by anything not ] until ]
            matches = re.findall(r'(\[Para[^\]]+\])', segment, re.IGNORECASE | re.DOTALL)
            
            para_id = matches[-1] if matches else "N/A"
            
            # Remove the tag from content to clean it up
            content_text = segment
            if para_id != "N/A":
                content_text = content_text.replace(para_id, "")
                # para_id might have newlines in it (e.g. [Para 133 to\n135]), clean it for metadata value
                para_id = " ".join(para_id.split())
            
            raw_blocks.append((content_text, para_id))
            
    else:
        # Strategy B: Split by [Para ...] tags
        # Assumes [Para ...] marks the end of a block.
        # We split keeping the delimiter.
        parts = re.split(r'(\[Para[^\]]+\])', content, flags=re.IGNORECASE | re.DOTALL)
        
        # parts[0] is text before first tag
        # parts[1] is first tag
        # parts[2] is text after first tag (before second tag)
        # parts[3] is second tag
        # ...
        
        current_text = parts[0]
        
        # We pair text with the FOLLOWING tag.
        # Loop starts from 1
        for i in range(1, len(parts), 2):
            tag = parts[i]
            
            # The current_text belongs to this tag
            para_id = " ".join(tag.split())
            raw_blocks.append((current_text, para_id))
            
            # Next text chunk
            if i + 1 < len(parts):
                current_text = parts[i+1]
            else:
                current_text = ""
                
        # Handle trailing text if any (after last tag)
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
            chunks.append(chunk)

    return chunks

def main():
    all_chunks = []
    
    print(f"Scanning directory: {SOURCE_DIR}")
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith('.txt'):
                fullpath = os.path.join(root, file)
                print(f"Processing: {file}")
                file_chunks = parse_text_file(fullpath)
                all_chunks.extend(file_chunks)
                
    print(f"Generated {len(all_chunks)} chunks from all files.")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"Chunks saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
