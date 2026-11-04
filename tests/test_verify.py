import asyncio

import httpx
import pytest

import idea_council.verify as verify_mod
from idea_council.confidence import Evidence
from idea_council.verify import check_source, doi_of, page_text, quote_in_page

PAGE = "<html><body><script>var x=1</script><p>LLM judges show   position bias in pairwise comparisons.</p></body></html>"


HOSTS = {"pubmed.ncbi.nlm.nih.gov": "130.14.29.110", "made-up-mirror.example": "93.184.215.16", "example.com": "93.184.215.14", "doi.org": "104.18.1.1", "api.openalex.org": "104.18.2.2",
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
        return httpx.Response(200, json={"id": "https://openalex.org/W1", "is_retracted": False, "title": "Real paper",
                                         "abstract_inverted_index": {"judges": [1], "Panel": [0], "agree": [2],
                                                                     "strongly": [3], "together": [4]}})
    if request.url.host == "made-up-mirror.example":
        raise httpx.ConnectError("no route")
    if request.url.host == "redirector.example":
        return httpx.Response(302, headers={"location": "http://evil.example/admin"})
    if request.url.host == "pubmed.ncbi.nlm.nih.gov":
        return httpx.Response(203, headers={"content-type": "text/html"},
                              text="<html>Cookies must be enabled to use this site.</html>")
    if request.url.path == "/law":
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text='<html><body><iframe src="/LSW/lsInfoP.do?x=1"></iframe></body></html>')
    if request.url.host == "doi.org" and "10.9999" in str(request.url):
        return httpx.Response(404)
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
    assert gone is None and "404" in reason  # not in OpenAlex, and doi.org itself does not resolve it
    gone, reason = run(ev("https://doi.org/10.1111/retracted.1"), html_handler)
    assert gone is None and "retracted" in reason
    gone, reason = run(ev("http://127.0.0.1/admin"), html_handler)
    assert gone is None and "not a public URL" in reason
    assert doi_of("https://doi.org/10.52202/079017-2197") == "10.52202/079017-2197"
    assert doi_of("https://example.com/a") is None


def test_doi_quote_can_match_openalex_abstract_but_not_title_alone():
    kept, _ = run(ev("https://doi.org/10.5555/real.1", quote="Panel judges agree strongly together"), html_handler)
    assert kept is not None and kept.checked == "quote found (OpenAlex abstract)"
    assert kept.work == "https://openalex.org/W1"
    gone, _ = run(ev("https://made-up-mirror.example/10.5555/real.1", quote="Real paper ... Real paper"),
                       html_handler)
    assert gone is None


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


def test_openalex_record_only_stands_in_for_doi_and_publisher_hosts():
    gone, reason = run(ev("https://made-up-mirror.example/10.5555/real.1", quote="Real paper Panel judges agree"),
                       html_handler)
    assert gone is None and "unreachable" in reason


def test_counter_search_must_be_text():
    import asyncio as _a
    import json as _j

    from idea_council.llm import Completion
    from idea_council.verify import Verifier

    class LLM:
        async def complete(self, model, messages, *, web_search=False):
            evidence = [{"url": f"https://s{i}.example/a", "grade": "B", "stance": "support",
                         "operator": f"O{i}", "origin": f"G{i}", "quote": "q" * 30} for i in range(3)]
            return Completion(model, "```json\n" + _j.dumps({"evidence": evidence, "counter_search": None}) + "\n```")

    check = _a.run(Verifier(LLM(), "m", check_urls=False).verify("claim"))
    assert check.verdict.independent_support == 3 and not check.verdict.passed


def test_meta_charset_is_used():
    from idea_council.verify import _charset
    raw = '<html><head><meta charset="euc-kr"></head><body>한국어 원문 구절입니다 정말로</body></html>'.encode("euc-kr")
    resp = httpx.Response(200, headers={"content-type": "text/html"}, content=raw)
    assert _charset(resp, raw) == "euc-kr"


def test_bot_wall_is_unchecked_not_dropped_and_pmid_is_looked_up():
    from idea_council.verify import work_lookup
    assert work_lookup("https://pubmed.ncbi.nlm.nih.gov/32353859/") == ("pmid", "32353859")
    assert work_lookup("https://pmc.ncbi.nlm.nih.gov/articles/PMC7239082/") == ("pmcid", "PMC7239082")
    kept, _ = run(ev("https://pubmed.ncbi.nlm.nih.gov/32353859/", quote="a sentence only on the real page"),
                  html_handler)
    assert kept is not None and kept.checked.startswith("unchecked")
    kept, _ = run(ev("https://pubmed.ncbi.nlm.nih.gov/32353859/", quote="Panel judges agree strongly together"),
                  html_handler)
    assert kept.checked == "quote found (OpenAlex abstract)"


def test_counter_search_phrases():
    from idea_council.verify import counter_searched
    assert counter_searched("'X 반론' 검색: 주장을 반박하는 근거는 존재하지 않음")
    assert counter_searched("searched 'X criticism': no rebuttal; n/a results were irrelevant")
    assert not counter_searched("반대 근거 검색 안 함")
    assert not counter_searched("not performed")
    assert not counter_searched("N/A")


def test_arxiv_urls_resolve_through_openalex_doi_and_amp_params_are_cut():
    from idea_council.verify import work_lookup
    assert work_lookup("https://arxiv.org/abs/2006.10256v2") == ("doi", "10.48550/arxiv.2006.10256")
    assert doi_of("https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0230405&type=printable") \
        == "10.1371/journal.pone.0230405"


def test_iframe_shell_is_unchecked_not_dropped():
    kept, _ = run(ev("https://example.com/law", quote="개인정보처리자는 정보주체의 동의를 받아"), html_handler)
    assert kept is not None and kept.checked.startswith("unchecked")


def test_origin_paper_joins_its_preprint_and_published_twin(monkeypatch):
    from idea_council.confidence import assess as _assess
    from idea_council.verify import _with_origin_work

    async def fake_check(client, kind, value):
        title = "title:bertpretrainingofdeepbidirectionaltransformersforlanguageunderstanding"
        if value == "10.48550/arxiv.1810.04805":
            return None, "", f"https://openalex.org/W2896457183 10.48550/arxiv.1810.04805 {title}", True
        return None, "", f"https://openalex.org/W2963341956 10.18653/v1/n19-1423 {title}", True
    monkeypatch.setattr(verify_mod, "_openalex_check", fake_check)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(html_handler)) as client:
            preprint = Evidence(url="https://arxiv.org/abs/1810.04805", grade="B", operator="Google AI",
                                origin="arXiv 1810.04805", quote="q" * 20)
            news = Evidence(url="https://news.example/bert", grade="B", operator="News",
                            origin="10.18653/v1/N19-1423", quote="q" * 20)
            other = Evidence(url="https://blog.example/x", grade="B", operator="Blog", origin="own test", quote="q" * 20)
            return [await _with_origin_work(client, e) for e in (preprint, news, other)]
    items = asyncio.run(go())
    items[0] = Evidence(**{**items[0].__dict__, "work": items[0].work + " title:bertpretrainingofdeepbidirectionaltransformersforlanguageunderstanding"})
    v = _assess(items)
    assert v.independent_support == 2  # preprint + story about the published paper = one source


def test_ranking_as_string():
    from idea_council.parsing import parse_review
    review = parse_review('```json\n{"ranking": "B > A > C", "reviews": []}\n```', ["A", "B", "C"])
    assert review.ranking == ["B", "A", "C"]


def test_footnotes_scope_notes_and_nature_dois():
    from idea_council.verify import counter_searched
    page = page_text('<p>Large language models are trained.<sup class="reference">[1]</sup> They predict text.</p>',
                     "text/html")
    assert quote_in_page("Large language models are trained. They predict text.", page)
    assert quote_in_page("Large language models are trained.[1] They predict text.", page)
    assert counter_searched("'X 반박' 검색: 반박 자료 없음. 유료 논문 원문은 확인하지 않았음")
    assert counter_searched("searched 'X fails': no rebuttal found; long-term trials were not done yet")
    assert not counter_searched("not performed")
    assert doi_of("https://www.nature.com/articles/s41586-020-2649-2") == "10.1038/s41586-020-2649-2"


def test_server_errors_are_unchecked():
    def h(request):
        return httpx.Response(502)
    kept, _ = run(ev("https://example.com/slow"), h)
    assert kept is not None and kept.checked.startswith("unchecked")


def test_nature_pdf_link_doi():
    assert doi_of("https://www.nature.com/articles/s41586-020-2649-2.pdf") == "10.1038/s41586-020-2649-2"


def test_rebuttal_flag_needs_a_credible_listed_rebuttal():
    import asyncio as _a
    import json as _j

    from idea_council.llm import Completion
    from idea_council.verify import Verifier

    class LLM:
        async def complete(self, model, messages, *, web_search=False):
            evidence = [{"url": f"https://s{i}.example/a", "grade": "B", "stance": "support", "operator": f"O{i}",
                         "origin": f"G{i}", "quote": "q" * 30} for i in range(3)]
            body = {"evidence": evidence, "counter_search": "searched 'X wrong': nothing relevant",
                    "rebuttal_found": True}
            return Completion(model, "```json\n" + _j.dumps(body) + "\n```")

    check = _a.run(Verifier(LLM(), "m", check_urls=False).verify("claim"))
    assert check.verdict.passed  # a bare flag with no A/B rebuttal listed does not block


def test_more_paper_url_forms_and_versioned_dois():
    from idea_council.verify import work_lookup
    assert work_lookup("https://arxiv.org/html/2306.05685v2") == ("doi", "10.48550/arxiv.2306.05685")
    assert work_lookup("https://www.ncbi.nlm.nih.gov/pubmed/12345678") == ("pmid", "12345678")
    assert doi_of("https://www.medrxiv.org/content/10.1101/2021.01.01.21249094v1") == "10.1101/2021.01.01.21249094"
    assert doi_of("https://doi.org/10.5555/journal.v12") == "10.5555/journal.v12"  # a real DOI ending in v12


def test_long_article_mentioning_captcha_is_not_a_bot_wall():
    from idea_council.verify import _looks_like_challenge
    assert _looks_like_challenge("<html><body>Just a moment... enable JavaScript and cookies</body></html>")
    article = "<p>" + " ".join(["research"] * 300) + " CAPTCHA systems are studied here.</p>"
    assert not _looks_like_challenge(article)


def test_script_rendered_statistics_page_is_kept_unchecked():
    from idea_council.verify import _script_rendered
    page = "<html><script>" + "x" * 100000 + "</script><body>World Bank menu Data Catalog</body></html>"
    assert _script_rendered(page)
    article = "<html><body><p>" + "word " * 5000 + "</p></body></html>"
    assert not _script_rendered(article)


def test_structured_counter_search_counts_as_text():
    from idea_council.verify import _text, counter_searched
    assert counter_searched(_text({"query": "X criticism", "result": "no credible rebuttal among 8 results"}))


def test_doi_meta_suffix_is_trimmed():
    assert doi_of("https://iopscience.iop.org/article/10.1088/1748-9326/ab1234/meta") == "10.1088/1748-9326/ab1234"
