"""Freeze official documents, then extract text without using a language model."""

from datetime import datetime, timezone
import email
import html
from html.parser import HTMLParser
import re
import urllib.parse
import urllib.request

from .common import digest, read_json, require, write_json

SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
DIGIT_TO_SUPER = str.maketrans("0123456789", SUPERSCRIPTS)
SUPER_TO_DIGIT = str.maketrans(SUPERSCRIPTS, "0123456789")


def check_url(url, hosts=None):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme in ("http", "https") and parsed.hostname, "Expected an HTTP(S) URL")
    require(not parsed.username and not parsed.password and not parsed.fragment, "Credentials/fragments are not allowed in URLs")
    if hosts is not None:
        require(any(parsed.hostname == host or parsed.hostname.endswith("." + host) for host in hosts), "Document host is outside allowed_source_hosts")
    return url


class Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTMLText(HTMLParser):
    def __init__(self, superscript_classes=()):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0
        self.superscript_classes = set(superscript_classes)
        self.superscript_stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("span", "sup"):
            superscript = tag == "sup" or bool(set((attrs.get("class") or "").split()) & self.superscript_classes)
            superscript = superscript or bool(re.search(r"vertical-align\s*:\s*super", attrs.get("style") or "", re.I))
            self.superscript_stack.append(superscript)
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "br", "div", "tr", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("span", "sup") and self.superscript_stack:
            self.superscript_stack.pop()
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if tag in ("p", "div", "tr", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data.translate(DIGIT_TO_SUPER) if any(self.superscript_stack) else data)


def decode_document(raw, content_type=""):
    if b"MIME-Version:" in raw[:500] or b"multipart/related" in raw[:500]:
        parts = [part for part in email.message_from_bytes(raw).walk() if part.get_content_type() == "text/html"]
        require(len(parts) == 1, "Expected one HTML body in the web archive")
        content_type = parts[0].get("Content-Type", "")
        raw = parts[0].get_payload(decode=True)
    charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    if charset is None:
        charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", raw[:4000].decode("ascii", errors="ignore"), re.I)
    if charset:
        text = raw.decode(charset[1])
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1251")
    return text


def reference_links(raw, content_type, base_url):
    """Preserve official amendment links separately from normative plain text."""
    if raw.startswith(b"%PDF"):
        return []
    text = decode_document(raw, content_type)
    links = {}
    for href, label in re.findall(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', text, re.I | re.S):
        title = " ".join(html.unescape(re.sub(r"<[^>]+>", "", label)).split())
        if not re.fullmatch(r"от \d{2}\.\d{2}\.\d{4} № ?\d+-ФЗ", title):
            continue
        url = urllib.parse.urljoin(base_url, html.unescape(href))
        parsed = urllib.parse.urlsplit(url)
        ids = urllib.parse.parse_qs(parsed.query).get("nd", [])
        if parsed.hostname == "pravo.gov.ru" and len(ids) == 1 and ids[0].isdigit():
            # The official portal's MIME export for the document linked in HTML.
            export_url = "http://pravo.gov.ru/proxy/ips/?savertf=&nd=" + ids[0] + "&rdk=0"
            metadata_url = "http://pravo.gov.ru/proxy/ips/?doc_itself=&vkart=card&nd=" + ids[0] + "&rdk=0"
            links[export_url] = dict(title=title, url=export_url, link_url=url, metadata_url=metadata_url)
    return list(links.values())


def extract(raw, content_type=""):
    if raw.startswith(b"%PDF"):
        try:
            import fitz
        except ImportError:
            raise ValueError("PDF extraction requires PyMuPDF. Supply HTML/MIME or install it separately.") from None
        with fitz.open(stream=raw, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        require(len(text.strip()) >= 100, "PDF has no usable text layer. OCR is not implemented")
        return text
    text = decode_document(raw, content_type)
    if re.search(r"<html\b|<!doctype\s+html", text, re.I):
        require("</html>" in text.lower(), "Truncated HTML document")
        superscript_classes = re.findall(r"\.([\w-]+)\s*\{[^}]*vertical-align\s*:\s*super\b[^}]*\}", text, re.I)
        parser = HTMLText(superscript_classes)
        parser.feed(text)
        text = "".join(parser.parts)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip()) + "\n"


def articles(text):
    # Mixed heading styles occur in introductory acts. Amending acts also
    # contain nested dotted headings inside their quoted replacements.
    digits = "0-9" + SUPERSCRIPTS
    number_pattern = r"([" + digits + r"]+(?:[.\-][" + digits + r"]+)*)"
    bare = list(re.finditer(r"(?m)^Статья[ \t]+" + number_pattern + r"[ \t]*$", text))
    dotted = list(re.finditer(r"(?m)^Статья[ \t]+" + number_pattern + r"\.[ \t]+", text))
    matches = sorted(bare + dotted, key=lambda match: match.start())
    if bare and dotted:
        def sequential(items):
            return [match[1] for match in items] == [str(n) for n in range(1, len(items) + 1)]
        if not sequential(matches):
            require(sequential(bare), "Ambiguous mixture of article heading styles")
            matches = bare
    result = {}
    for index, match in enumerate(matches):
        raw_number = match[1]
        number = re.sub("[" + SUPERSCRIPTS + "]+",
                        lambda m: ("." if m.start() and raw_number[m.start() - 1].isdigit() else "") + m[0].translate(SUPER_TO_DIGIT),
                        raw_number)
        require(number not in result, f"Ambiguous article heading: {number}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[number] = text[match.start():end].strip()
    return result


class Sources:
    def __init__(self, root, config_dir, hosts, offline=False):
        self.root = root / "sources"
        self.config_dir = config_dir
        self.hosts = hosts
        self.offline = offline
        self.documents = {}
        self.frozen_documents = {}

    def fetch(self, spec):
        url = check_url(spec["url"], self.hosts)
        identity = digest(url)
        if identity in self.frozen_documents:
            document = self.frozen_documents[identity]
            if spec.get("sha256"):
                require(document["raw_sha256"] == spec["sha256"], "Supplied source hash mismatch")
            return document
        directory = self.root / identity
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            raw = (directory / "source.bin").read_bytes()
            text = (directory / "text.txt").read_text(encoding="utf-8")
            require(digest(raw) == metadata["raw_sha256"] and digest(text.encode()) == metadata["text_sha256"], "Frozen source changed")
            require(extract(raw, metadata["content_type"]) == text, "Source extraction changed")
        else:
            content_type = spec.get("content_type", "")
            if "path" in spec:
                require(bool(spec.get("sha256")), "Local snapshots require an expected sha256")
                raw = (self.config_dir / spec["path"]).read_bytes()
                provenance = "user_supplied_snapshot"
            else:
                require(not self.offline, "Offline replay is missing a frozen source")
                request = urllib.request.Request(url, headers={"User-Agent": "RuLawProofBench-builder/0.1"})
                opener = urllib.request.build_opener(Redirects(self.hosts))
                try:
                    with opener.open(request, timeout=60) as response:
                        check_url(response.url, self.hosts)
                        content_type = response.headers.get("Content-Type", "")
                        raw = response.read(16 * 1024 * 1024 + 1)
                        final_url = response.url
                except OSError:
                    raise ValueError("Source download failed. Supply a verified local snapshot or retry") from None
                provenance = "download"
            require(len(raw) <= 16 * 1024 * 1024, "Source exceeds 16 MiB")
            text = extract(raw, content_type)
            require(len(text.strip()) >= 100, "Source contains too little text")
            if spec.get("end_marker"):
                require(spec["end_marker"] in text[-4000:], "Document completeness marker missing")
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "source.bin").write_bytes(raw)
            (directory / "text.txt").write_text(text, encoding="utf-8")
            metadata = dict(url=url, final_url=final_url if provenance == "download" else url,
                            raw_sha256=digest(raw), text_sha256=digest(text.encode()),
                            content_type=content_type, acquisition=provenance,
                            archived_at=datetime.now(timezone.utc).isoformat())
            write_json(metadata_path, metadata)
        if spec.get("sha256"):
            require(digest(raw) == spec["sha256"], "Supplied source hash mismatch")
        document = dict(id=identity, url=url, text=text, reference_links=reference_links(raw, metadata["content_type"], url),
                        **{k: v for k, v in metadata.items() if k not in ("url",)})
        self.documents[identity] = document
        return document

    def fork(self, root):
        """Share only frozen documents; new downloads belong to this article."""
        child = Sources(root, self.config_dir, self.hosts, self.offline)
        child.documents = dict(self.documents)
        child.frozen_documents = dict(self.documents)
        return child

    def fragment(self, spec, number):
        document = self.fetch(spec)
        available = articles(document["text"])
        require(str(number) in available, f"Article {number} was not uniquely extracted")
        return dict(source_id=document["id"], url=document["url"], article=str(number), text=available[str(number)])
