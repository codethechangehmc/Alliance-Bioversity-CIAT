import json
import re
import string
from pathlib import Path

CLEANED = Path('pdf_processing/finished_data/cleaned_bo1005-leketa-2019.json')

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace('\n', ' ')
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def find_section_for_note(note, cleaned_pdf_dict):
    if not isinstance(note, str) or note.strip() == '' or note.strip().upper() == 'NA':
        return ('NA', 'NA')

    norm_note = normalize_text(note)
    note_words = [w for w in norm_note.split() if len(w) > 2]
    if not note_words:
        return ('not_found', '')

    # Try exact substring match first
    for section in cleaned_pdf_dict.get('sections', []):
        heading = section.get('heading') or ''
        content_lines = section.get('content', [])
        section_text = heading + ' ' + ' '.join(content_lines)
        if normalize_text(norm_note) in normalize_text(section_text):
            for sent in re.split(r'[\.\?!]\s+', ' '.join(content_lines)):
                if normalize_text(norm_note) in normalize_text(sent):
                    return (heading or 'unknown', sent.strip())
            return (heading or 'unknown', section_text[:300])

    # Fallback: sentence overlap heuristic
    for section in cleaned_pdf_dict.get('sections', []):
        heading = section.get('heading') or ''
        content_lines = section.get('content', [])
        section_text = ' '.join(content_lines)
        sentences = re.split(r'[\.\?!]\s+', section_text)
        for sent in sentences:
            norm_sent = normalize_text(sent)
            sent_words = set([w for w in norm_sent.split() if len(w) > 2])
            if not sent_words:
                continue
            overlap = sum(1 for w in note_words if w in sent_words)
            if overlap / max(1, len(note_words)) >= 0.4:
                return (heading or 'unknown', sent.strip())

    return ('not_found', '')


if not CLEANED.exists():
    print('cleaned json not found at', CLEANED)
    raise SystemExit(1)

with open(CLEANED, 'r', encoding='utf-8') as fh:
    cleaned = json.load(fh)

# pick a candidate sentence from the PDF to use as a 'note' to search for
# we'll use the first non-empty content line from the first section
sample_note = ''
for section in cleaned.get('sections', []):
    for line in section.get('content', []):
        if line and len(line.strip()) > 20:
            sample_note = line.strip()
            break
    if sample_note:
        break

print('Sample note chosen (truncated):', sample_note[:200])
sec, exc = find_section_for_note(sample_note, cleaned)
print('Matched section heading:', sec)
print('Matched excerpt:', exc[:300])

# Also test with a short phrase that might appear in notes
short_note = sample_note.split('.')[0][:120]
print('\nShort note test:', short_note)
sec2, exc2 = find_section_for_note(short_note, cleaned)
print('Matched section heading:', sec2)
print('Matched excerpt:', exc2[:300])
