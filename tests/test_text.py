from aicd.normalize import normalize
from aicd.prompts import clean_output, too_similar


def test_backtick_and_mojibake_become_apostrophes():
    assert normalize("I`d have responded") == "I'd have responded"
    assert normalize("itÃ¯Â¿Â½s pizza-time") == "it's pizza-time"
    assert normalize("it’s fine") == "it's fine"


def test_whitespace_urls_mentions_and_user_fragments():
    assert normalize("  hello   world ") == "hello world"
    assert normalize("see http://bit.ly/AF8JT now") == "see URL now"
    assert normalize("@bob thanks!") == "thanks!"
    assert normalize("_elj Appreciated, uni email is helpful..") == "Appreciated, uni email is helpful.."


def test_profanity_masked_like_the_tweet_dataset():
    assert normalize("this is shit") == "this is ****"
    assert normalize("this is ****") == "this is ****"


def test_emoji_and_dashes():
    assert normalize("so good \U0001F602\U0001F525") == "so good"
    assert normalize("wait—what") == "wait - what"
    assert normalize("hmm…") == "hmm..."


def test_normalize_is_idempotent():
    s = " _x I`m   here http://a.b — ok \U0001F600 "
    assert normalize(normalize(s)) == normalize(s)


def test_clean_output_strips_wrappers():
    assert clean_output('"just got home lol"') == "just got home lol"
    assert clean_output("Here's a tweet:\nmy cat is judging me again") == "my cat is judging me again"
    assert clean_output("Reply: same here honestly") == "same here honestly"
    assert clean_output("<post>ok fine</post>") == "ok fine"
    assert clean_output("   ") is None


def test_too_similar():
    assert too_similar("sleeping in my own bed tonight", "sleeping in my own bed tonight!")
    assert not too_similar("finally home, my own pillow at last", "sleeping in my own bed tonight")
