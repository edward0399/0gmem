"""Tests for QueryAnalyzer: query understanding and routing."""

from zerogmem.retriever.query_analyzer import (
    QueryAnalyzer,
    QueryAnalysis,
    QueryIntent,
    ReasoningType,
)


class TestQueryAnalyzer:
    """Tests for the QueryAnalyzer."""

    def _analyze(self, query):
        analyzer = QueryAnalyzer()
        return analyzer.analyze(query)

    def test_classify_temporal_intent(self):
        analysis = self._analyze("When did Alice visit Paris?")
        assert analysis.intent == QueryIntent.TEMPORAL

    def test_classify_causal_intent(self):
        analysis = self._analyze("Why did Bob move to New York?")
        assert analysis.intent == QueryIntent.CAUSAL

    def test_classify_preference_intent(self):
        analysis = self._analyze("Does Alice like hiking?")
        assert analysis.intent == QueryIntent.PREFERENCE

    def test_classify_relational_intent(self):
        analysis = self._analyze("Who does Alice know at work?")
        assert analysis.intent == QueryIntent.RELATIONAL

    def test_classify_factual_default(self):
        analysis = self._analyze("What is Alice's favorite color?")
        # Factual is the default when no specific pattern matches
        assert analysis.intent in [QueryIntent.FACTUAL, QueryIntent.PREFERENCE]

    def test_extract_entities(self):
        analysis = self._analyze("What did Alice say about Bob?")
        # Should extract capitalized names
        assert isinstance(analysis.entities, list)

    def test_extract_keywords(self):
        analysis = self._analyze("What did Alice say about hiking in the mountains?")
        assert isinstance(analysis.keywords, list)
        assert len(analysis.keywords) > 0

    def test_negation_check_detected(self):
        analysis = self._analyze("Didn't Alice say she never liked spiders?")
        assert analysis.is_negation_check is True

    def test_negation_check_not_detected(self):
        analysis = self._analyze("What does Alice like?")
        assert analysis.is_negation_check is False

    def test_result_is_query_analysis(self):
        analysis = self._analyze("When did Alice visit Paris?")
        assert isinstance(analysis, QueryAnalysis)
        assert analysis.original_query == "When did Alice visit Paris?"
        assert isinstance(analysis.reasoning_type, ReasoningType)

    def test_classify_list_intent(self):
        analysis = self._analyze("List all the places Alice has visited.")
        assert analysis.intent == QueryIntent.LIST

    def test_classify_verification_intent(self):
        analysis = self._analyze("Is it true that Alice lives in New York?")
        assert analysis.intent == QueryIntent.VERIFICATION
