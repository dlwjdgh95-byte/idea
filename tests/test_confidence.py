from idea_council.confidence import Evidence, assess, independent_groups


def ev(url, grade="B", stance="support", operator="", origin="", **kw):
    return Evidence(url=url, grade=grade, stance=stance, operator=operator, origin=origin, **kw)


def test_two_independent_a_sources_pass_path1():
    v = assess([
        ev("https://doi.org/10.1/a", "A", operator="Journal X", origin="study-a"),
        ev("https://doi.org/10.1/b", "A", operator="Journal Y", origin="study-b"),
    ])
    assert (v.path, v.band, v.passed) == ("path1", (90, 99), True)


def test_single_primary_source_passes_path1():
    v = assess([ev("https://stats.gov/table", "A", operator="통계청", origin="통계청", primary=True)])
    assert v.path == "path1" and v.passed


def test_three_independent_web_sources_pass_path2():
    v = assess([
        ev("https://news-a.com/1", operator="A일보", origin="자체 취재 A"),
        ev("https://news-b.com/2", operator="B방송", origin="자체 취재 B"),
        ev("https://blog-c.com/3", operator="C연구소", origin="자체 측정 C"),
    ])
    assert (v.path, v.band, v.label) == ("path2", (80, 89), "사실(웹 교차 확인)")
    assert v.passed


def test_same_press_release_counts_once():
    v = assess([
        ev("https://news-a.com/1", operator="A일보", origin="회사 보도자료 2026-10-01"),
        ev("https://news-b.com/2", operator="B방송", origin="회사 보도자료 2026-10-01"),
        ev("https://news-c.com/3", operator="C매체", origin="회사 보도자료 2026-10-01"),
    ])
    assert v.independent_support == 1
    assert not v.passed


def test_same_domain_counts_once_but_hosting_platforms_do_not():
    same_site = independent_groups([
        ev("https://news-a.com/1", operator="기자1", origin="취재1"),
        ev("https://www.news-a.com/2", operator="기자2", origin="취재2"),
    ])
    assert len(same_site) == 1
    hosted = independent_groups([
        ev("https://arxiv.org/abs/1", "B", operator="Lab 1", origin="paper 1"),
        ev("https://arxiv.org/abs/2", "B", operator="Lab 2", origin="paper 2"),
    ])
    assert len(hosted) == 2


def test_unknown_independence_collapses_and_grade_c_is_ignored():
    v = assess([
        ev("https://x.com/1"),
        ev("https://y.com/2"),
        ev("https://z.com/3"),
        ev("https://anon.example/4", "C", operator="익명", origin="?"),
    ])
    assert v.independent_support == 1
    assert v.band == (40, 59)


def test_conflict_of_interest_is_not_counted():
    v = assess([
        ev("https://vendor.com/blog", operator="Vendor", origin="Vendor", conflict_of_interest=True),
        ev("https://news-a.com/1", operator="A일보", origin="취재 A"),
    ])
    assert v.independent_support == 1


def test_contradiction_blocks_pass():
    support = [ev(f"https://s{i}.com", operator=f"S{i}", origin=f"o{i}") for i in range(3)]
    mixed = assess(support + [ev("https://c.com", "A", "contradict", operator="C", origin="c")])
    assert mixed.band == (40, 59) and not mixed.passed
    tie = assess(support[:1] + [ev("https://c.com", "A", "contradict", operator="C", origin="c")])
    assert tie.band == (40, 59)  # 동수는 혼재, 반대가 우세할 때만 반박됨
    refuted = assess(support[:1] + [
        ev("https://c.com", "A", "contradict", operator="C", origin="c"),
        ev("https://d.com", "B", "contradict", operator="D", origin="d"),
    ])
    assert refuted.label == "반박됨" and refuted.band == (0, 39)


def test_missing_origin_or_operator_cannot_be_independent():
    v = assess([
        ev("https://news-a.com/1", operator="A일보"),
        ev("https://news-b.com/2", operator="B방송"),
        ev("https://news-c.com/3", origin="자체 취재 C"),
    ])
    assert v.independent_support == 1 and not v.passed


def test_no_evidence_is_lack_of_evidence():
    v = assess([])
    assert v.label == "근거 부족" and v.confidence < 40


def test_path2_needs_a_counter_search():
    three = [ev(f"https://n{i}.com/1", operator=f"N{i}", origin=f"o{i}") for i in range(3)]
    assert assess(three, counter_searched=True).passed
    v = assess(three, counter_searched=False)
    assert not v.passed and v.band == (60, 79)


def test_unconfirmed_quotes_cannot_pass_alone():
    pdf = ev("https://stats.gov/t.pdf", "A", operator="통계청", origin="통계청", primary=True,
             checked="unchecked: not a text page")
    assert assess([pdf]).band == (60, 79)
    ok = ev("https://stats.gov/t.html", "A", operator="통계청", origin="통계청", primary=True, checked="quote found")
    assert assess([ok]).passed
    three = [ev(f"https://n{i}.com/1", operator=f"N{i}", origin=f"o{i}", checked="unchecked: access refused")
             for i in range(3)]
    assert not assess(three).passed


def test_one_paper_through_several_mirrors_is_one_source():
    v = assess([
        ev("https://doi.org/10.48550/arXiv.2306.05685", "A", operator="UC Berkeley", origin="own"),
        ev("https://www.semanticscholar.org/paper/10.48550/arXiv.2306.05685", "A", operator="LMSYS", origin="paper"),
        ev("https://arxiv.org/abs/2306.05685", "A", operator="Zheng et al.", origin="arXiv 2306.05685"),
    ])
    assert v.independent_support == 1 and not v.passed


def test_platform_operator_neither_merges_nor_shows_independence():
    platform = [
        ev("https://pubmed.ncbi.nlm.nih.gov/1/", "A", operator="PubMed (NLM)", origin="Kim 2020 trial"),
        ev("https://pubmed.ncbi.nlm.nih.gov/2/", "A", operator="PubMed (NLM)", origin="Lee 2022 cohort"),
    ]
    assert assess(platform).band == (40, 59)  # operator not really known
    named = [
        ev("https://pubmed.ncbi.nlm.nih.gov/1/", "A", operator="Seoul Natl Univ", origin="Kim 2020 trial"),
        ev("https://pubmed.ncbi.nlm.nih.gov/2/", "A", operator="Yonsei Univ", origin="Lee 2022 cohort"),
    ]
    assert assess(named).path == "path1"


def test_news_about_a_paper_shares_its_origin():
    v = assess([
        ev("https://doi.org/10.1038/s41586-020-2649-2", "A", operator="Harris lab", origin="https://doi.org/10.1038/s41586-020-2649-2"),
        ev("https://sciencedaily.com/x", operator="ScienceDaily", origin="doi:10.1038/s41586-020-2649-2"),
        ev("https://phys.org/y", operator="Phys.org", origin="10.1038/s41586-020-2649-2"),
    ])
    assert v.independent_support == 1


def test_same_url_twice_counts_once():
    v = assess([ev("https://medium.com/@a/post", operator="A", origin="a"),
                ev("https://medium.com/@a/post/", operator="A author", origin="a post")])
    assert v.independent_support == 1


def test_unknown_sources_do_not_bridge_independent_groups():
    v = assess([
        ev("https://x.org/1", "A", operator="X", origin="x1"),
        ev("https://y.org/1", "A", operator="Y", origin="y1"),
        ev("https://x.org/2", operator="X"),
        ev("https://y.org/2", operator="Y"),
    ])
    assert v.path == "path1"


def test_unconfirmed_groups_count_once():
    srcs = [ev(f"https://n{i}.com/1", operator=f"N{i}", origin=f"o{i}",
               checked="quote found" if i == 0 else "unchecked: access refused") for i in range(4)]
    v = assess(srcs)
    assert v.independent_support == 2 and not v.passed


def test_unknown_sources_never_add_to_a_count():
    reuters = ev("https://reuters.com/a", operator="Reuters", origin="Reuters 취재")
    bbc = ev("https://bbc.com/a", operator="BBC", origin="BBC 취재")
    reuters2 = ev("https://reuters.com/b", operator="Reuters")
    assert assess([reuters, bbc, reuters2]).band == (60, 79)
    a1 = ev("https://x.org/1", "A", operator="X", origin="x1")
    a_unknown = ev("https://y.org/1", "A", operator="unknown", origin="?")
    assert assess([a1, a_unknown]).band == (60, 79)
    only_unknown = [ev(f"https://u{i}.org", "A") for i in range(3)]
    assert assess(only_unknown).band == (40, 59)


def test_rebuttal_found_blocks_path2():
    three = [ev(f"https://n{i}.com/1", operator=f"N{i}", origin=f"o{i}") for i in range(3)]
    assert assess(three, rebuttal_found=True).band == (40, 59)


def test_unconfirmed_contra_counts_once():
    sup = [ev(f"https://s{i}.com", operator=f"S{i}", origin=f"s{i}", checked="quote found") for i in range(2)]
    con = [ev(f"https://c{i}.com", stance="contradict", operator=f"C{i}", origin=f"c{i}",
              checked="unchecked: access refused") for i in range(3)]
    assert assess(sup + con).label != "반박됨"


def test_papers_in_the_same_journal_are_not_merged():
    v = assess([
        ev("https://www.nejm.org/doi/full/10.1056/NEJMoa2001017", "A", operator="Wuhan team", origin="trial 1"),
        ev("https://www.nejm.org/doi/full/10.1056/NEJMoa2007764", "A", operator="NIAID", origin="trial 2"),
    ])
    assert v.path == "path1"


def test_bare_arxiv_id_origin_and_korean_particle_after_doi():
    from idea_council.confidence import doi_of, work_id
    assert work_id("2306.05685") == work_id("arXiv 2306.05685") == work_id("arXiv:2306.05685")
    assert doi_of("10.1038/abc의 결과") == "10.1038/abc"


def test_own_primary_document_proves_only_that_it_was_published():
    filing = ev("https://dart.fss.or.kr/x", "A", operator="삼성전자", origin="공시", primary=True,
                conflict_of_interest=True, checked="quote found")
    v = assess([filing])
    assert v.passed and "당사자" in v.label
