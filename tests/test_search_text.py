"""Подготовка текста для BM25: порядок нормализации и отсутствие стоп-слов."""
import pytest

from p3.search.text import lemma, normalize_references, reference_token, tokenize


# --- ссылки нормализуются до лемматизации ------------------------------

@pytest.mark.parametrize("text", [
    "п. 3 ст. 88 НК РФ",
    "пункт 3 статьи 88 Налогового кодекса",
    "п.3 ст.88 НК РФ",
    "пунктом 3 статьи 88 Кодекса",
])
def test_citation_variants_collapse_to_one_token(text):
    """Самый информативный токен запроса не должен распадаться на четыре."""
    assert tokenize(text) == ["nk1_st88_p3"]


def test_reference_token_is_word_safe():
    """Точка разорвала бы токен: статьи 93.1 и 129.1 — ядро процедурных норм."""
    assert reference_token("nk1/st93.1/p2") == "nk1_st93_1_p2"
    assert tokenize("статьи 93.1 НК РФ") == ["nk1_st93_1"]


def test_code_designator_is_absorbed():
    """«НК РФ» — часть цитаты, а не отдельное слово.

    В корпусе кодекс называет себя «настоящего Кодекса», поэтому токен «рф»
    встречается в единицах документов, получает высокий IDF и перебивает точную
    ссылку. Ровно это и происходило на запросе «п. 3 ст. 88 НК РФ».
    """
    assert "рф" not in tokenize("п. 3 ст. 88 НК РФ")
    assert "нк" not in tokenize("п. 3 ст. 88 НК РФ")


def test_other_codes_absorbed_too():
    assert tokenize("частью 1 статьи 19.4 КоАП РФ") == ["koap_st19_4_p1"]
    assert tokenize("статьи 15 Гражданского кодекса Российской Федерации") == ["gk_st15"]


def test_reference_inside_sentence():
    tokens = tokenize("Документы представляются в порядке п. 3 ст. 88 НК РФ")
    assert "nk1_st88_p3" in tokens
    assert "порядок" in tokens


# --- лемматизация ------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("камеральной", "камеральный"),
    ("проверкой", "проверка"),
    ("документов", "документ"),
    ("налогоплательщика", "налогоплательщик"),
])
def test_lemmatization(word, expected):
    assert lemma(word) == expected


def test_numbers_not_lemmatized():
    assert lemma("2026") == "2026"
    assert lemma("03-02-07/1") == "03-02-07/1"


def test_reference_tokens_not_lemmatized():
    assert lemma("nk1_st88_p3") == "nk1_st88_p3"


# --- стоп-слова --------------------------------------------------------

@pytest.mark.parametrize("word", ["не", "либо", "иной", "если"])
def test_load_bearing_words_survive(word):
    """Стандартный стоп-лист выбрасывает их, а в праве они несущие:
    «не вправе истребовать» превратилось бы во «вправе истребовать»."""
    assert word in tokenize(f"норма {word} применяется")


def test_negation_preserved():
    assert "не" in tokenize("налоговый орган не вправе истребовать документы")


# --- прочее ------------------------------------------------------------

def test_yo_is_folded():
    assert tokenize("учёт") == tokenize("учет")


def test_text_without_citations_unchanged():
    assert normalize_references("простой текст без ссылок") == "простой текст без ссылок"


def test_empty_input():
    assert tokenize("") == []
