import asyncio

import httpx
import pytest

import idea_council.verify as verify_mod
from idea_council.confidence import Evidence
from idea_council.verify import check_source, doi_of, page_text, quote_in_page

PAGE = "<html><body><script>var x=1</script><p>LLM judges show   position bias in pairwise comparisons.</p></body></html>"


HOSTS = {"example.com": "93.184.215.14", "doi.org": "104.18.1.1", "api.openalex.org": "104.18.2.2",
         "evil.example": "127.0.0.1", "redirector.example": "93.184.215.15"}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    async def resolve(host, port):
        if host in HOSTS:
            return [HOSTS[host]]
        try:  # numeric forms resolve the way the OS would
            import ipaddress
            return [str(ipaddress.ip_address(int(host, 0) if host[:1].isdigit() and "." not in host else host))]
        except ValueError:
            raise OSError("no such host") from None
    monkeypatch.setattr(verify_mod, "_resolve", resolve)


def ev(url, quote="position bias in pairwise comparisons"):
    return Evidence(url=url, grade="B", operator="Op", origin="Or", quote=quote)


def run(evidence, handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_source(client, evidence)
    return asyncio.run(go())


def html_handler(request):
    if request.url.host == "api.openalex.org":
        if "10.9999" in str(request.url):
            return httpx.Response(404)
        if "10.1111" in str(request.url):
            return httpx.Response(200, json={"is_retracted": True})
        return httpx.Response(200, json={"is_retracted": False, "title": "Real paper",
                                         "abstract_inverted_index": {"judges": [1], "Panel": [0], "agree": [2]}})
    if request.url.host == "redirector.example":
        return httpx.Response(302, headers={"location": "http://evil.example/admin"})
    if request.url.path == "/missing":
        return httpx.Response(404)
    if request.url.path == "/blocked":
        return httpx.Response(403)
    if request.url.path == "/paper.pdf":
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
    return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)


def test_quote_matching_ignores_markup_and_spacing():
    page = page_text(PAGE, "text/html")
    assert quote_in_page("Position bias in  pairwise comparisons.", page)
    assert quote_in_page("LLM judges show position ... in pairwise comparisons", page)
    assert not quote_in_page("in pairwise comparisons ... LLM judges show position", page)  # order matters
    assert not quote_in_page("judges never show position bias", page)
    assert not quote_in_page("bias", page)  # too short to prove anything
    assert "varx1" not in page  # scripts are dropped


def test_source_check_keeps_found_quote_and_drops_the_rest():
    kept, _ = run(ev("https://example.com/a"), html_handler)
    assert kept is not None and kept.checked == "quote found"
    gone, reason = run(ev("https://example.com/a", quote="a sentence that is not on the page"), html_handler)
    assert gone is None and "quote not found" in reason
    gone, reason = run(ev("https://example.com/a", quote=""), html_handler)
    assert gone is None and "no quote" in reason
    gone, reason = run(ev("https://example.com/missing"), html_handler)
    assert gone is None and "404" in reason


def test_refused_or_binary_pages_are_kept_but_marked_unchecked():
    kept, _ = run(ev("https://example.com/blocked"), html_handler)
    assert kept.checked.startswith("unchecked")
    kept, _ = run(ev("https://example.com/paper.pdf"), html_handler)
    assert kept.checked.startswith("unchecked")


def test_unknown_or_retracted_doi_and_private_urls_are_dropped():
    gone, reason = run(ev("https://doi.org/10.9999/fake.123"), html_handler)
    assert gone is None and "not found in OpenAlex" in reason
    gone, reason = run(ev("https://doi.org/10.1111/retracted.1"), html_handler)
    assert gone is None and "retracted" in reason
    gone, reason = run(ev("http://127.0.0.1/admin"), html_handler)
    assert gone is None and "not a public URL" in reason
    assert doi_of("https://doi.org/10.52202/079017-2197") == "10.52202/079017-2197"
    assert doi_of("https://example.com/a") is None


def test_doi_quote_can_match_openalex_title_and_abstract():
    kept, _ = run(ev("https://doi.org/10.5555/real.1", quote="Real paper Panel judges agree"), html_handler)
    assert kept is not None and kept.checked == "quote found (OpenAlex record)"


def test_redirects_and_numeric_hosts_to_private_addresses_are_blocked():
    gone, reason = run(ev("https://redirector.example/a"), html_handler)
    assert gone is None and "not a public URL" in reason and "evil.example" in reason
    for url in ("http://2130706433/", "http://0x7f000001/", "http://localhost/x"):
        gone, reason = run(ev(url), html_handler)
        assert gone is None, url


def test_malformed_urls_are_dropped_not_raised():
    for url in ("https://[::1/a", "https://example.com:abc/", "https://example.com/\x00"):
        gone, reason = run(ev(url), html_handler)
        assert gone is None and reason


def test_doi_suffixes_are_trimmed():
    assert doi_of("https://www.biorxiv.org/content/10.1101/2020.03.22.002386v3.full") == "10.1101/2020.03.22.002386"
    assert doi_of("https://onlinelibrary.wiley.com/doi/10.1002/anie.201915678/abstract") == "10.1002/anie.201915678"
    assert doi_of("https://doi.org/10.1037/h0048850") == "10.1037/h0048850"


def test_stance_and_flags_are_parsed_strictly():
    from idea_council.verify import _to_evidence
    base = {"url": "https://a.com", "grade": "A", "quote": "q"}
    assert _to_evidence({**base, "stance": "refute"}).stance == "contradict"
    assert _to_evidence({**base, "stance": "반대"}).stance == "contradict"
    assert _to_evidence({**base, "stance": "support"}).stance == "support"
    assert _to_evidence({**base, "stance": "neutral"}) is None
    assert _to_evidence({**base}) is None  # no stance, no vote
    e = _to_evidence({**base, "stance": "support", "primary": "false", "conflict_of_interest": "true"})
    assert e.primary is False and e.conflict_of_interest is True
