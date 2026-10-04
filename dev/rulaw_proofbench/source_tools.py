"""Reproducible, dependency-free extraction of archived pravo.gov.ru documents."""
import email
from html.parser import HTMLParser
import re


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'br', 'div', 'tr'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract(raw):
    if raw.startswith(b'MIME-Version:'):
        message = email.message_from_bytes(raw)
        parts = [part for part in message.walk() if part.get_content_type() == 'text/html']
        if len(parts) != 1:
            raise ValueError('Expected exactly one HTML document in official web archive')
        raw = parts[0].get_payload(decode=True)
    html = raw.decode('cp1251')
    if '</html>' not in html.lower():
        raise ValueError('Truncated HTML')
    parser = TextExtractor()
    parser.feed(html)
    return '\n'.join(' '.join(line.split()) for line in ''.join(parser.parts).splitlines()
                     if line.strip()) + '\n'


def article(text, number):
    matches = list(re.finditer(r'^Статья ' + re.escape(str(number)) + r'\.', text, re.M))
    if len(matches) != 1:
        raise ValueError(f'Article {number}: expected unique heading, found {len(matches)}')
    start = matches[0].start()
    following = re.search(r'^Статья \d', text[matches[0].end():], re.M)
    end = matches[0].end() + following.start() if following else len(text)
    return text[start:end].strip()
