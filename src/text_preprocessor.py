import re

# Daftar singkatan umum dalam Bahasa Indonesia & Inggris
PROTECTED_ABBREVIATIONS = [
    # Titel
    'dr', 'drg', 'prof', 'ir', 'hj', 'h', 's', 'm', 'a', 'b', 'c',
    # Singkatan umum
    'mr', 'mrs', 'ms', 'sr', 'jr',
    # Bahasa Indonesia
    'spt', 'dsb', 'dll', 'dst', 'ttd', 'yth', 'an', 'a.n', 'a/n',
    # Instansi
    'rs', 'polri', 'tni', 'pt', 'cv', 'tbk',
    # Negara
    'a.m', 'p.m', 'e.g', 'i.e', 'etc', 'vol', 'ed', 'no',
    # Angka romawi (opsional)
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
]

def split_into_sentences(text):
    """
    Split teks menjadi kalimat dengan proteksi singkatan & angka desimal.
    """
    text = str(text).strip()
    
    # Hapus reference Wikipedia
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Normalisasi whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # STEP 1: PROTEKSI SINGKATAN (ganti titik jadi placeholder)
    protected_text = text
    
    # Buat pattern: dr. → dr<DOT>, drg. → drg<DOT>, dst.
    for abbr in sorted(PROTECTED_ABBREVIATIONS, key=len, reverse=True):
        # Case insensitive: Dr., DR., dr.
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\.', re.IGNORECASE)
        replacement = abbr.upper() + '<DOT>'  # konsisten uppercase
        protected_text = pattern.sub(replacement, protected_text)
    
    # Proteksi angka desimal: 138.793 → 138<DOT>793
    protected_text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected_text)
    
    # STEP 2: SPLIT KALIMAT
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
    
    # STEP 3: KEMBALIKAN TITIK ASLI
    clean_sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        
        # Kembalikan <DOT> jadi titik
        sent = sent.replace('<DOT>', '.')
        
        # Hapus titik di AWAL (jika ada)
        sent = re.sub(r'^\.+', '', sent).strip()
        
        # Pastikan ada titik di AKHIR
        if sent and not sent[-1] in '.!?':
            sent += '.'
        
        # Filter panjang
        if len(sent) >= 20:
            clean_sentences.append(sent)
    
    return clean_sentences