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

samples = []
for section in cleaned.get('sections', []):
    heading = section.get('heading') or 'unknown'
    content = ' '.join(section.get('content', []))
    # split into sentences
    sentences = re.split(r'(?<=[\.\?!])\s+', content)
    for sent in sentences:
        s = sent.strip()
        if len(s) >= 60:
            samples.append((heading, s))

# limit samples to 60 to keep runtime small
samples = samples[:60]

results = []
for heading, sentence in samples:
    sec, exc = find_section_for_note(sentence, cleaned)
    correct = False
    exp_heading = heading if heading else 'unknown'
    # consider match correct if headings match exactly or both unknown
    if (not exp_heading or exp_heading.lower()== 'unknown') and (not sec or sec.lower() in ('unknown','not_found','na')):
        correct = True
    elif exp_heading and sec and normalize_text(exp_heading) == normalize_text(sec):
        correct = True
    results.append({'expected': exp_heading, 'found': sec, 'excerpt': exc, 'correct': correct, 'sentence': sentence[:200]})

# summary
correct_count = sum(1 for r in results if r['correct'])
print(f'Tested {len(results)} samples. Correct matches: {correct_count} ({correct_count/len(results)*100:.1f}%)')

# print up to 10 mismatches
mismatches = [r for r in results if not r['correct']]
print(f'Number of mismatches: {len(mismatches)}')
for m in mismatches[:10]:
    print('\n---')
    print('Expected heading:', m['expected'])
    print('Found heading   :', m['found'])
    print('Excerpt         :', m['excerpt'][:300])
    print('Sentence sample :', m['sentence'])
