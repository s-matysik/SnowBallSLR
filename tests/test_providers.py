"""Provider parsers and HTTP plumbing, tested against recorded payloads.

No live calls: transport is mocked with respx, so the suite honours INV-4 while
still covering retry, 404 and pagination behaviour.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from snowballslr.determinism.cache import Cache
from snowballslr.errors import ProviderError, RateLimitError
from snowballslr.providers.base import BaseProvider, RateLimiter, RetryPolicy
from snowballslr.providers.crossref import CrossrefProvider, parse_item, parse_reference
from snowballslr.providers.grobid import parse_tei_references
from snowballslr.providers.openalex import (
    OpenAlexProvider,
    invert_abstract,
)
from snowballslr.providers.openalex import (
    parse_work as parse_openalex,
)
from snowballslr.providers.semanticscholar import (
    SemanticScholarProvider,
)
from snowballslr.providers.semanticscholar import (
    parse_work as parse_s2,
)
from snowballslr.types import Direction

OPENALEX_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1145/2601248.2601268",
    "title": "Guidelines for snowballing in systematic literature studies",
    "publication_year": 2014,
    "type": "proceedings-article",
    "language": "en",
    "is_retracted": False,
    "cited_by_count": 1500,
    "referenced_works_count": 21,
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
    "authorships": [{"author": {"display_name": "Claes Wohlin", "orcid": None}}],
    "primary_location": {"source": {"display_name": "EASE '14"}},
    "abstract_inverted_index": {"Background:": [0], "snowballing": [1], "works": [2]},
}

CROSSREF_ITEM = {
    "DOI": "10.1002/jrsm.1563",
    "title": ["Citationchaser: a tool for citation chasing"],
    "type": "journal-article",
    "container-title": ["Research Synthesis Methods"],
    "issued": {"date-parts": [[2022, 4, 1]]},
    "author": [{"family": "Haddaway", "given": "Neal R."}],
    "reference-count": 30,
    "is-referenced-by-count": 200,
    "reference": [
        {"DOI": "10.1145/2601248.2601268", "year": "2014", "author": "Wohlin"},
        {"article-title": "An unstructured reference", "year": "2011", "author": "Smith"},
        {"unstructured": "Somebody, somewhere, sometime."},
        {},
    ],
}

S2_PAPER = {
    "paperId": "abc123",
    "corpusId": 987654,
    "externalIds": {"DOI": "10.1016/j.infsof.2022.106908", "PubMed": "12345"},
    "title": "Successful combination of database search and snowballing",
    "abstract": "Hybrid search strategies...",
    "year": 2022,
    "venue": "Information and Software Technology",
    "publicationTypes": ["JournalArticle"],
    "referenceCount": 40,
    "citationCount": 90,
    "authors": [{"name": "Claes Wohlin"}],
}


# -- parsers -------------------------------------------------------------


def test_invert_abstract_reconstructs_word_order():
    assert invert_abstract({"world": [1], "hello": [0]}) == "hello world"
    assert invert_abstract(None) is None
    assert invert_abstract({}) is None


def test_openalex_parser_extracts_all_fields():
    w = parse_openalex(OPENALEX_WORK)
    assert w.key == "doi:10.1145/2601248.2601268"
    assert w.year == 2014 and w.type == "proceedings-article"
    assert w.source_ids["openalex"] == "W2741809807"
    assert w.first_author_surname == "Wohlin"
    assert w.venue == "EASE '14"
    assert w.abstract.startswith("Background:")


def test_openalex_parser_tolerates_missing_fields():
    assert parse_openalex(None) is None
    minimal = parse_openalex({"id": "https://openalex.org/W9", "display_name": "T"})
    assert minimal.key == "oa:W9" and minimal.doi is None


def test_crossref_item_parser():
    w = parse_item(CROSSREF_ITEM)
    assert w.key == "doi:10.1002/jrsm.1563"
    assert w.year == 2022 and w.venue == "Research Synthesis Methods"
    assert w.first_author_surname == "Haddaway"


def test_crossref_reference_parser_flags_unresolved():
    with_doi = parse_reference(CROSSREF_ITEM["reference"][0])
    without_doi = parse_reference(CROSSREF_ITEM["reference"][1])
    unstructured = parse_reference(CROSSREF_ITEM["reference"][2])
    assert with_doi.doi == "10.1145/2601248.2601268" and not with_doi.unresolved
    assert without_doi.unresolved and without_doi.title
    assert unstructured is not None and unstructured.unresolved
    assert parse_reference(CROSSREF_ITEM["reference"][3]) is None


def test_s2_parser():
    w = parse_s2(S2_PAPER)
    assert w.key == "doi:10.1016/j.infsof.2022.106908"
    assert w.source_ids["s2"] == "987654" and w.source_ids["pmid"] == "12345"
    assert w.type == "journalarticle"
    assert parse_s2(None) is None


def test_grobid_tei_reference_extraction():
    tei = """<?xml version="1.0"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><div><listBibl>
      <biblStruct>
        <analytic>
          <title level="a">Guidelines for snowballing</title>
          <author><persName><surname>Wohlin</surname></persName></author>
          <idno type="DOI">10.1145/2601248.2601268</idno>
        </analytic>
        <monogr><title level="j">EASE</title>
          <imprint><date type="published" when="2014"/></imprint></monogr>
      </biblStruct>
      <biblStruct><analytic><title level="a">No date study</title>
        <author><persName><surname>Smith</surname></persName></author>
      </analytic></biblStruct>
    </listBibl></div></back></text></TEI>"""
    refs = parse_tei_references(tei)
    assert len(refs) == 2
    assert refs[0]["doi"] == "10.1145/2601248.2601268"
    assert refs[0]["year"] == 2014 and refs[0]["authors"] == ["Wohlin"]
    assert refs[1]["year"] is None


def test_grobid_tei_handles_malformed_xml():
    assert parse_tei_references("<not xml") == []


# -- HTTP plumbing -------------------------------------------------------


def test_rate_limiter_is_a_noop_at_zero_rps():
    RateLimiter(0).wait()  # must not raise or block


def test_retry_backoff_is_jitter_free():
    policy = RetryPolicy(base=1.0, factor=2.0)
    assert [policy.delay(i) for i in range(3)] == [1.0, 2.0, 4.0]


@respx.mock
def test_openalex_resolve_and_cache(tmp_path, monkeypatch):
    monkeypatch.undo()  # this test supplies its own mocked transport
    route = respx.get(url__startswith="https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json=OPENALEX_WORK)
    )
    cache = Cache(tmp_path / "cache")
    provider = OpenAlexProvider(cache, mailto="test@example.org")
    first = provider.resolve("W2741809807")
    second = provider.resolve("W2741809807")
    assert first.key == second.key == "doi:10.1145/2601248.2601268"
    assert route.call_count == 1  # second call served from cache
    assert Direction.FORWARD in provider.supports
    provider.close()


@respx.mock
def test_openalex_forward_pagination(tmp_path, monkeypatch):
    monkeypatch.undo()
    pages = [
        httpx.Response(
            200,
            json={"results": [OPENALEX_WORK], "meta": {"next_cursor": "cur2"}},
        ),
        httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}}),
    ]
    respx.get(url__startswith="https://api.openalex.org/works").mock(side_effect=pages)
    provider = OpenAlexProvider(Cache(tmp_path / "cache"))
    from snowballslr.types import Work

    parent = Work(key="oa:W1", title="p", title_norm="p", source_ids={"openalex": "W1"})
    assert len(provider.citations(parent)) == 1
    provider.close()


@respx.mock
def test_404_is_cached_as_a_negative_result(tmp_path, monkeypatch):
    monkeypatch.undo()
    route = respx.get(url__startswith="https://api.crossref.org").mock(
        return_value=httpx.Response(404)
    )
    provider = CrossrefProvider(Cache(tmp_path / "cache"))
    assert provider.resolve("10.1000/missing") is None
    assert provider.resolve("10.1000/missing") is None
    assert route.call_count == 1
    provider.close()


@respx.mock
def test_non_json_body_raises_provider_error_naming_the_provider(tmp_path, monkeypatch):
    """A 2xx status does not guarantee JSON.

    Interception proxies, CDN maintenance pages and WAF challenges answer 200 with
    HTML. An unguarded ``resp.json()`` surfaced that as a bare ``JSONDecodeError``
    from inside the cache callback, naming neither the provider nor the URL, and
    aborted the whole iteration.
    """
    monkeypatch.undo()
    respx.get(url__startswith="https://api.crossref.org").mock(
        return_value=httpx.Response(
            200, text="<html><body>Service unavailable</body></html>",
            headers={"content-type": "text/html"},
        )
    )
    provider = CrossrefProvider(Cache(tmp_path / "cache"))
    with pytest.raises(ProviderError) as exc:
        provider.resolve("10.1000/htmlpage")
    message = str(exc.value)
    assert "crossref" in message
    assert "non-JSON" in message
    assert "text/html" in message
    provider.close()


@respx.mock
def test_redirected_doi_is_followed(tmp_path, monkeypatch):
    """Crossref answers 301 with an empty body for reassigned DOI prefixes.

    httpx does not follow redirects by default, so such a record used to abort the
    iteration: status 301 is neither 404 nor a retry status, so it fell through to
    the JSON decode with nothing to decode.
    """
    monkeypatch.undo()
    final = "https://api.crossref.org/works/10.1000/new"
    respx.get("https://api.crossref.org/works/10.1000/old").mock(
        return_value=httpx.Response(301, headers={"location": final})
    )
    respx.get(final).mock(
        return_value=httpx.Response(200, json={"message": CROSSREF_ITEM})
    )
    provider = CrossrefProvider(Cache(tmp_path / "cache"))
    work = provider.resolve("10.1000/old")
    assert work is not None
    assert work.doi == CROSSREF_ITEM["DOI"].lower()
    provider.close()


@respx.mock
def test_persistent_429_raises_rate_limit_error(tmp_path, monkeypatch):
    monkeypatch.undo()
    respx.get(url__startswith="https://api.crossref.org").mock(
        return_value=httpx.Response(429)
    )
    provider = CrossrefProvider(
        Cache(tmp_path / "cache"), retry=RetryPolicy(max_retries=2, base=0.0, factor=0.0)
    )
    with pytest.raises(RateLimitError):
        provider.resolve("10.1000/limited")
    provider.close()


@respx.mock
def test_client_error_raises_provider_error(tmp_path, monkeypatch):
    monkeypatch.undo()
    respx.get(url__startswith="https://api.crossref.org").mock(
        return_value=httpx.Response(403)
    )
    provider = CrossrefProvider(Cache(tmp_path / "cache"))
    with pytest.raises(ProviderError):
        provider.resolve("10.1000/forbidden")
    provider.close()


@respx.mock
def test_crossref_references_and_matching(tmp_path, monkeypatch):
    monkeypatch.undo()
    respx.get(url__startswith="https://api.crossref.org/works/10.1002").mock(
        return_value=httpx.Response(200, json={"message": CROSSREF_ITEM})
    )
    respx.get(url__startswith="https://api.crossref.org/works").mock(
        return_value=httpx.Response(
            200, json={"message": {"items": [{**CROSSREF_ITEM, "score": 90.0}]}}
        )
    )
    provider = CrossrefProvider(Cache(tmp_path / "cache"))
    parent = parse_item(CROSSREF_ITEM)
    refs = provider.references(parent)
    assert len(refs) == 3
    assert provider.citations(parent) == []  # crossref is backward-only
    matches = provider.match_bibliographic("Citationchaser Haddaway 2022")
    assert matches and matches[0][1] == 90.0
    provider.close()


@respx.mock
def test_semanticscholar_paged_references(tmp_path, monkeypatch):
    monkeypatch.undo()
    respx.get(url__startswith="https://api.semanticscholar.org").mock(
        return_value=httpx.Response(200, json={"data": [{"citedPaper": S2_PAPER}], "next": None})
    )
    provider = SemanticScholarProvider(Cache(tmp_path / "cache"))
    from snowballslr.types import Work

    parent = Work(key="doi:10.1/x", title="p", title_norm="p", doi="10.1000/x")
    refs = provider.references(parent)
    assert len(refs) == 1 and refs[0].year == 2022
    provider.close()


def test_base_provider_default_methods_are_empty():
    class Dummy(BaseProvider):
        name = "dummy"

    d = Dummy(Cache("/tmp/snowball-dummy-cache"))
    from snowballslr.types import Work

    w = Work(key="k", title="t", title_norm="t")
    assert d.references(w) == [] and d.citations(w) == []
    with pytest.raises(NotImplementedError):
        d.resolve("x")


# -- regression: identifier filter and internal parameters ----------------


@respx.mock
def test_batch_fetch_falls_back_to_the_alternate_id_filter(tmp_path, monkeypatch):
    """`openalex_id:` is not a valid filter and returns 400, not an empty page."""
    monkeypatch.undo()
    seen: list[str] = []

    def responder(request):
        f = request.url.params.get("filter", "")
        seen.append(f.split(":", 1)[0])
        if f.startswith("ids.openalex:"):
            return httpx.Response(400, json={"message": "Invalid query parameters"})
        return httpx.Response(200, json={"results": [OPENALEX_WORK]})

    respx.get(url__startswith="https://api.openalex.org/works").mock(side_effect=responder)
    provider = OpenAlexProvider(
        Cache(tmp_path / "cache"), retry=RetryPolicy(max_retries=1, base=0.0, factor=0.0)
    )
    works = provider._fetch_by_ids(["W1", "W2"])
    assert len(works) == 1
    assert seen == ["ids.openalex", "openalex"]
    assert provider._id_filter_key == "openalex"
    provider.close()


@respx.mock
def test_batch_fetch_raises_when_no_filter_alias_works(tmp_path, monkeypatch):
    monkeypatch.undo()
    respx.get(url__startswith="https://api.openalex.org/works").mock(
        return_value=httpx.Response(400, json={"message": "Invalid query parameters."})
    )
    provider = OpenAlexProvider(
        Cache(tmp_path / "cache"), retry=RetryPolicy(max_retries=1, base=0.0, factor=0.0)
    )
    with pytest.raises(ProviderError) as exc:
        provider._fetch_by_ids(["W1"])
    # The provider's own message must survive into the exception.
    assert "Invalid query parameters." in str(exc.value)
    provider.close()


@respx.mock
def test_internal_params_never_reach_the_wire(tmp_path, monkeypatch):
    """Underscore keys disambiguate cache entries; they are not query parameters."""
    monkeypatch.undo()
    captured: list[dict] = []

    def responder(request):
        captured.append(dict(request.url.params))
        return httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}})

    respx.get(url__startswith="https://api.openalex.org/works").mock(side_effect=responder)
    provider = OpenAlexProvider(Cache(tmp_path / "cache"))
    from snowballslr.types import Work as W

    provider.citations(W(key="oa:W1", title="p", title_norm="p", source_ids={"openalex": "W1"}))
    assert captured
    assert not any(k.startswith("_") for params in captured for k in params)
    provider.close()


@respx.mock
def test_cursor_pages_are_cached_separately(tmp_path, monkeypatch):
    """Successive cursor pages share a query string, so they need distinct cache keys."""
    monkeypatch.undo()
    pages = [
        httpx.Response(200, json={"results": [OPENALEX_WORK], "meta": {"next_cursor": "c2"}}),
        httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}}),
    ]
    respx.get(url__startswith="https://api.openalex.org/works").mock(side_effect=pages)
    cache = Cache(tmp_path / "cache")
    provider = OpenAlexProvider(cache)
    from snowballslr.types import Work as W

    parent = W(key="oa:W1", title="p", title_norm="p", source_ids={"openalex": "W1"})
    assert len(provider.citations(parent)) == 1
    assert len(cache) == 2  # two distinct pages retained
    provider.close()
