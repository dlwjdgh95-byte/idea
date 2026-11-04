import asyncio

import httpx

from idea_council.confidence import Evidence
from idea_council.verify import check_source, doi_of, page_text, quote_in_page

PAGE = "<html><body><script>var x=1</script><p>LLM judges show   position bias in pairwise comparisons.</p></body></html>"


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
    assert quote_in_page("LLM judges show ... pairwise comparisons", page)
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
