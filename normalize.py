import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Basic Unicode normalization.

    For now, this only:
    - handles missing values
    - converts Unicode to NFKC
    - converts to lowercase
    - removes extra whitespace
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = remove_latin_accents(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_latin_accents(text):
    """
    Remove accents from Latin characters while preserving
    non-Latin scripts.
    """
    result = []

    for char in text:
        if "LATIN" in unicodedata.name(char, ""):
            decomposed = unicodedata.normalize("NFKD", char)
            result.append(
                "".join(
                    c for c in decomposed
                    if not unicodedata.combining(c)
                )
            )
        else:
            result.append(char)

    return "".join(result)


def extract_postal_code(address, country):
    """
    Extract common postal/PIN codes.
    """

    country = country.lower().strip()

    if country == "us":
        match = re.search(r"\b\d{5}(?:-\d{4})?\b(?=\s*$)", address)
        if match:
            return match.group(0)

    elif country == "india":
        match = re.search(r"\b[1-9]\d{5}\b", address)
        if match:
            return match.group(0)

    elif country in {"france", "franc"}:
        match = re.search(r"\b\d{5}\b", address)
        if match:
            return match.group(0)

    return ""

def clean_address(address, country):
    """
    Remove common address filler/relational phrases.
    """

    patterns = [
        r"\bdoor\s*no\.?\b",
        r"\bnear\b",
        r"\bopp\.?\b",
        r"\bopposite\b",
        r"\bc/o\b",
        r"\bs/o\b",
    ]

    for pattern in patterns:
        address = re.sub(pattern, " ", address)

    address = re.sub(r"\s*,\s*", ", ", address)
    address = re.sub(r"\s*\.\s*", " ", address)
    address = re.sub(r"\s+", " ", address).strip()
    

    return address

def detect_script(text):
    """
    Detect the main writing script used in a text.
    """

    scripts = set()

    for char in text:
        code = ord(char)

        if (
            0x0041 <= code <= 0x005A
            or 0x0061 <= code <= 0x007A
        ):
            scripts.add("latin")

        elif 0x0900 <= code <= 0x097F:
            scripts.add("devanagari")

        elif 0x0B80 <= code <= 0x0BFF:
            scripts.add("tamil")

        elif char.isalpha():
            scripts.add("other")

    if not scripts:
        return "unknown"

    if len(scripts) == 1:
        return next(iter(scripts))

    return "mixed"
def normalize_row(name, address, country):
    """
    Normalize one business record.
    """

    name = normalize_text(name)
    address = normalize_text(address)
    address = clean_address(address, country)
    country = normalize_text(country)
    postal_code = extract_postal_code(address, country)
    name_script = detect_script(name)

    # Remove common legal suffixes from the business name
    legal_suffixes = [
        "incorporated",
        "inc.",
        "inc",
        "llc",
        "l.l.c.",
        "l.l.c",
        "private limited",
        "pvt.",
        "pvt",
        "limited",
        "ltd.",
        "ltd",
        "corporation",
        "corp.",
        "corp",
        "company",
        "co.",
        "co",
        "pc",
    ]
    dba_name = ""

    dba_pattern = r"\s+(?:dba|d/b/a|doing business as)\s+"

    dba_match = re.search(dba_pattern, name)

    if dba_match:
        legal_name = name[:dba_match.start()].strip()
        dba_name = name[dba_match.end():].strip()
        name = legal_name

    legal_suffix = ""

    for suffix in legal_suffixes:
       if name.endswith(" " + suffix):
            legal_suffix = suffix
            name = name[:-(len(suffix) + 1)].strip()
            break

       if dba_name.endswith(" " + suffix):
           legal_suffix = suffix
           dba_name = dba_name[:-(len(suffix) + 1)].strip()
           break

    return {
        "core_name": name,
        "dba_name": dba_name,
        "legal_suffix": legal_suffix,
        "core_address": address,
        "postal_code": postal_code,
        "landmark": "",
        "country_code": country,
        "name_script": name_script,
    }
def normalize_dataframe(df):
    """
    Normalize a DataFrame using vectorized pandas operations
    where possible.
    """

    result = pd.DataFrame(index=df.index)

    result["entity_id"] = df["entity_id"]

    # Normalize country
    country = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.lower()
        .str.strip()
    )

    # Normalize names
    names = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Normalize addresses
    addresses = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove common address phrases
    addresses = (
        addresses
        .str.replace(r"\bdoor\s*no\.?\b", " ", regex=True)
        .str.replace(r"\bnear\b", " ", regex=True)
        .str.replace(r"\bopp\.?\b", " ", regex=True)
        .str.replace(r"\bopposite\b", " ", regex=True)
        .str.replace(r"\bc/o\b", " ", regex=True)
        .str.replace(r"\bs/o\b", " ", regex=True)
        .str.replace(r"\s*,\s*", ", ", regex=True)
        .str.replace(r"\s*\.\s*", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove common legal suffixes from names
    legal_pattern = (
        r"\s+(?:incorporated|inc\.?|llc|l\.l\.c\.?|"
        r"private limited|pvt\.?|limited|ltd\.?|"
        r"corporation|corp\.?|company|co\.?|pc)$"
    )

    names = names.str.replace(
        legal_pattern,
        "",
        regex=True
    ).str.strip()

    # Extract postal/PIN codes
    postal = pd.Series("", index=df.index, dtype=str)

    us_mask = country.eq("us")
    india_mask = country.eq("india")
    france_mask = country.isin(["france", "franc"])

    postal.loc[us_mask] = (
        addresses.loc[us_mask]
        .str.extract(r"(\d{5}(?:-\d{4})?)\s*$", expand=False)
        .fillna("")
    )

    postal.loc[india_mask] = (
        addresses.loc[india_mask]
        .str.extract(r"\b([1-9]\d{5})\b", expand=False)
        .fillna("")
    )

    postal.loc[france_mask] = (
        addresses.loc[france_mask]
        .str.extract(r"\b(\d{5})\b", expand=False)
        .fillna("")
    )

    # Simple script detection
    def detect_script_fast(text):
        if not text:
            return "unknown"

        has_latin = bool(re.search(r"[a-z]", text))
        has_devanagari = bool(re.search(r"[\u0900-\u097f]", text))
        has_tamil = bool(re.search(r"[\u0b80-\u0bff]", text))

        scripts = sum([
            has_latin,
            has_devanagari,
            has_tamil
        ])

        if scripts == 0:
            return "unknown"
        if scripts > 1:
            return "mixed"
        if has_latin:
            return "latin"
        if has_devanagari:
            return "devanagari"
        return "tamil"

    result["core_name"] = names
    result["dba_name"] = ""
    result["legal_suffix"] = ""
    result["core_address"] = addresses
    result["postal_code"] = postal
    result["landmark"] = ""
    result["country_code"] = country
    result["name_script"] = names.map(detect_script_fast)

    return result