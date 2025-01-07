import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract


def ensure_http(url: str) -> str:
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        return f"http://{url}"
    return url


class URLAnalyzer:
    def __init__(self):
        self.url = None

    def analyse_url(self, *urls, domain_age_days=0):
        results = []

        for url in urls:
            self.url = url

            length = len(self.url)
            starts_with_ip = self.__starts_with_ip()
            url_entropy = self.__url_entropy(length)
            has_punycode = self.__url_has_punycode()
            digit_letter_ratio = self.__digit_letter_ratio()
            dot_count = self.__char_count('.')
            at_count = self.__char_count('@')
            dash_count = self.__char_count('-')
            tld_count = self.__tld_count()
            domain_has_digits = self.__domain_has_digits()
            subdomain_count = self.__subdomain_count()
            nan_char_entropy = self.__nan_char_entropy()
            has_internal_links = tld_count > 0
            domain_age_days = domain_age_days

            results.append(
                [
                    length,
                    starts_with_ip,
                    url_entropy,
                    has_punycode,
                    digit_letter_ratio,
                    dot_count,
                    at_count,
                    dash_count,
                    tld_count,
                    domain_has_digits,
                    subdomain_count,
                    nan_char_entropy,
                    has_internal_links,
                    domain_age_days
                ]
            )

        return pd.DataFrame(
            results, columns=[
                'url_length',
                'starts_with_ip',
                'url_entropy',
                'has_punycode',
                'digit_letter_ratio',
                'dot_count',
                'at_count',
                'dash_count',
                'tld_count',
                'domain_has_digits',
                'subdomain_count',
                'nan_char_entropy',
                'has_internal_links',
                'domain_age_days'
            ])

    def __domain_has_digits(self) -> bool:
        return len(re.findall(r'\d',
                              tldextract.extract(
                                  self.url).domain)) > 0

    def __starts_with_ip(self) -> bool:
        # regex for IP pattern
        ip_pattern = r'(^http[s]?://)?(\d{1,3}\.){3}\d{1,3}'

        # Check url for IP
        return re.match(ip_pattern, self.url) is not None

    def __url_entropy(self, url_len: int) -> float:
        char_count = Counter(self.url)

        return -sum((count / url_len) * math.log2(count / url_len) for count in char_count.values())

    def __url_has_punycode(self) -> bool:
        punycode_pattern = r'xn--'

        return re.search(punycode_pattern, self.url) is not None

    def __digit_letter_ratio(self) -> float:
        self.__digits = re.findall(r'\d', self.url)
        letters = re.findall(r'[a-zA-Z]', self.url)

        if len(letters) == 0:
            return 0.0

        return len(self.__digits) / len(letters)

    def __char_count(self, char: str) -> int:
        return self.url.count(char)

    def __subdomain_count(self) -> int:
        parsed_url = urlparse(ensure_http(self.url))
        hostname = parsed_url.hostname
        if hostname:
            segments = hostname.split('.')
            return max(0, len(segments) - 2)
        return 0

    def __nan_char_entropy(self) -> float:
        non_alnum_chars = re.findall(r'[^a-zA-Z0-9]', self.url)
        char_count = Counter(non_alnum_chars)
        total_chars = len(self.url)

        if total_chars == 0:
            return 0
        return -sum((count / total_chars) * math.log2(count / total_chars) for count in char_count.values())

    def __tld_count(self) -> int:
        parts = ensure_http(self.url).split('/')
        if len(parts) == 3:
            return 0
        for i in range(3):
            parts.pop(0)
        suffixes = [tldextract.extract(part).suffix for part in parts if '.' in part]
        suffixes = [suffix for suffix in suffixes if suffix != '']

        return len(suffixes)
