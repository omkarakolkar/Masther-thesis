
import csv
from collections import Counter
from pathlib import Path

# Config
INPUT_CSV = Path("/home/omkar/MasterThesis/datasets/FSD50K/dev_with_duration.csv")
#OUTPUT_CSV = Path("/home/omkar/datasets/FSD50K/single_label_files.csv")  # Extracted CSV

class_counts = Counter()
header = []

print(f"Reading from: {INPUT_CSV}")

with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
    
    reader = csv.DictReader(infile)
    # Get fieldnames from reader, ensure we keep them
    fieldnames = reader.fieldnames
    
    #writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    #writer.writeheader()
    
    label_classes = set()

    rows_written = 0
    total_rows = 0

    for row in reader:
        total_rows += 1
        labels = row["labels"].split(",")
        
        # Check if single label
        if len(labels) == 1:
            label = labels[0].strip()
            class_counts[label] += 1
            #writer.writerow(row)
            rows_written += 1

print(f"Total rows processed: {total_rows}")
print(f"Single label rows extracted: {rows_written}")
#print(f"Output written to: {OUTPUT_CSV}")

print("\n--- File Counts per Class (Single Label) ---")
# Sort by count descending
for label, count in class_counts.most_common():
    print(f"{label}: {count}")
