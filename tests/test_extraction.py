from aspect_sentiment_pipeline.extraction import ClauseAspectExtractor


def test_sentiment_stays_with_the_nearest_aspect():
    extractor = ClauseAspectExtractor(
        {
            "food": ["food"],
            "service": ["service"],
            "ambience": ["ambience"],
        }
    )

    mentions = extractor.extract(
        "The food was excellent but the service was painfully slow. The ambience was lovely."
    )
    scores = {mention.aspect: mention.sentiment for mention in mentions}

    assert scores["food"] > 0
    assert scores["service"] < 0
    assert scores["ambience"] > 0


def test_two_aspects_in_one_clause_use_local_context():
    extractor = ClauseAspectExtractor({"food": ["food"], "service": ["service"]})

    mentions = extractor.extract("The food was wonderful and the service was terrible.")
    scores = {mention.aspect: mention.sentiment for mention in mentions}

    assert scores["food"] > 0
    assert scores["service"] < 0
