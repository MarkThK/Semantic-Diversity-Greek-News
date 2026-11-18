import os
import re
import json
import csv

def extract_news_data(base_dir):
    extracted_data = []
    
    stats = {
        "total_files": 0,
        "files_with_explicit_numbering": 0,
        "files_with_newline_link": 0
    }
    
    # Iterate over subdirectories (news websites)
    for website_name in os.listdir(base_dir):
        website_path = os.path.join(base_dir, website_name)
        
        if not os.path.isdir(website_path):
            continue
            
        # Iterate over text files (timestamps)
        for file_name in os.listdir(website_path):
            if not file_name.endswith('.txt'):
                continue
                
            stats["total_files"] += 1
            timestamp = os.path.splitext(file_name)[0]
            file_path = os.path.join(website_path, file_name)
            
            try:
                # Read-only mode prevents any changes to the source files
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Skipping {file_path} due to read error: {e}")
                continue
                
            i = 0
            article_counter = 1
            
            file_has_explicit_numbering = False
            file_has_newline_link = False
            
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                    
                raw_title = None
                link = None
                
                # Condition A: Article \nLink: <URL>
                if i + 1 < len(lines) and lines[i+1].strip().lower().startswith("link:"):
                    file_has_newline_link = True
                    raw_title = line
                    link = re.sub(r'(?i)^link:\s*', '', lines[i+1].strip())
                    i += 2
                    
                # Condition B & C: Article [space(s)] <URL> OR Article \n<URL>
                else:
                    match = re.search(r'\s+(https?://\S+)', line)
                    if match:
                        link = match.group(1)
                        raw_title = line[:match.start()].strip()
                        i += 1
                    elif i + 1 < len(lines) and re.match(r'^https?://\S+', lines[i+1].strip()):
                        raw_title = line
                        link = lines[i+1].strip()
                        i += 2
                    else:
                        i += 1
                        continue
                        
                # Handle Numbering
                if raw_title and link:
                    # THE FIX: Changed \s* to \s+ so it requires a space after the dot!
                    # This prevents 750.000 from being chopped into "750" and "000".
                    num_match = re.match(r'^(\d+)\.\s+(.*)', raw_title)
                    if num_match:
                        file_has_explicit_numbering = True
                        numbering = num_match.group(1)
                        headline = num_match.group(2)
                    else:
                        # Auto-assign the counter to the extracted data ONLY
                        numbering = str(article_counter)
                        headline = raw_title
                        
                    extracted_data.append({
                        'website': website_name,
                        'timestamp': timestamp,
                        'numbering': numbering,
                        'headline': headline,
                        'link': link
                    })
                    
                    article_counter += 1
                    
            # Tally up the file flags
            if file_has_explicit_numbering:
                stats["files_with_explicit_numbering"] += 1
            if file_has_newline_link:
                stats["files_with_newline_link"] += 1
                
    return extracted_data, stats

def save_data(data, output_json, output_csv):
    # 'w' mode ensures the previous files are completely overwritten
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if data:
        keys = data[0].keys()
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    GLOBAL_DIR_PATH = "" 
    JSON_OUTPUT = "extracted_data.json"
    CSV_OUTPUT = "extracted_data.csv"
    
    print("Extracting data...")
    parsed_data, stats = extract_news_data(GLOBAL_DIR_PATH)
    
    if parsed_data:
        save_data(parsed_data, JSON_OUTPUT, CSV_OUTPUT)
        
        print("\n" + "="*40)
        print("EXTRACTION COMPLETE (OVERWROTE PREVIOUS FILES)")
        print("="*40)
        print(f"Total articles extracted: {len(parsed_data)}")
        print(f"Data saved to: {JSON_OUTPUT} and {CSV_OUTPUT}")
        print("-" * 40)
        print("FORMAT REPORT:")
        print(f"- Total .txt files scanned: {stats['total_files']}")
        print(f"- Files containing explicit numbering: {stats['files_with_explicit_numbering']}")
        print(f"- Files containing '\\nLink:' format: {stats['files_with_newline_link']}")
        print("="*40)
    else:
        print("No data was extracted. Check your directory path.")

