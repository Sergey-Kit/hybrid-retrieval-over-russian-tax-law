"""Guard'ы dense-канала из ТЗ §4: обрезка и префиксы.

Модель здесь не загружается — токенизатор подменяется. Guard'ы должны быть
проверяемы без GPU и без сети, иначе их никто не будет прогонять.
"""
import pytest

from p3.search.dense import ENCODERS, DenseEncoder, DenseIndex, TruncationError
import numpy as np


class FakeTokenizer:
    """Считает токены как слова — этого достаточно, чтобы проверить границу."""

    def encode(self, text, truncation=False):
        return text.split()


def encoder_with_fake(name="user-bge-m3"):
    encoder = DenseEncoder(name=name)
    encoder._tokenizer = FakeTokenizer()
    encoder._model = object()  # чтобы _ensure_loaded не полез в сеть
    return encoder


# --- guard на обрезку --------------------------------------------------

def test_input_within_limit_passes():
    encoder = encoder_with_fake()
    encoder.assert_fits(["короткий текст нормы"])


def test_too_long_input_raises():
    """Не предупреждение в лог, а исключение: обрезка молча отбрасывает хвост
    нормы, оставляет префикс, и метрики выходят правдоподобные."""
    encoder = encoder_with_fake("frida")  # лимит 512
    with pytest.raises(TruncationError, match="512"):
        encoder.assert_fits([" ".join(["слово"] * 600)])


def test_error_names_the_worst_case():
    encoder = encoder_with_fake("frida")
    with pytest.raises(TruncationError) as info:
        encoder.assert_fits(["ок", " ".join(["слово"] * 900)])
    assert "900" in str(info.value)
    assert "1 из 2" in str(info.value)


# --- префиксы ----------------------------------------------------------

def test_prefixes_declared_for_every_encoder():
    """Пустая строка — осознанное значение, а не забытое поле."""
    for name, profile in ENCODERS.items():
        assert "query_prefix" in profile, name
        assert "document_prefix" in profile, name


def test_instruction_model_uses_different_prefixes():
    """Одинаковый префикс с обеих сторон — самый коварный вариант ошибки:
    всё считается, поиск работает, recall просто хуже, чем мог бы."""
    profile = ENCODERS["frida"]
    assert profile["query_prefix"] != profile["document_prefix"]
    assert profile["query_prefix"] and profile["document_prefix"]


def test_bge_family_declares_no_prefix_explicitly():
    for name in ("user-bge-m3", "bge-m3"):
        assert ENCODERS[name]["query_prefix"] == ""
        assert ENCODERS[name]["document_prefix"] == ""


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError, match="неизвестный энкодер"):
        DenseEncoder(name="нет-такого")


def test_limits_match_the_model_cards():
    assert ENCODERS["frida"]["max_tokens"] == 512
    assert ENCODERS["user-bge-m3"]["max_tokens"] == 8192


# --- точный поиск ------------------------------------------------------

def test_brute_force_search_ranks_by_cosine():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    index = DenseIndex(doc_ids=["x", "y", "mid"], vectors=vectors)
    ranked = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [doc for doc, _ in ranked] == ["x", "mid"]


def test_empty_index_is_safe():
    assert DenseIndex(doc_ids=[], vectors=np.zeros((0, 2), dtype=np.float32)).search(
        np.array([1.0, 0.0], dtype=np.float32)
    ) == []


def test_index_roundtrip(tmp_path):
    index = DenseIndex(doc_ids=["a", "b"], vectors=np.eye(2, dtype=np.float32))
    path = tmp_path / "vectors.npz"
    index.save(path)
    loaded = DenseIndex.load(path)
    assert loaded.doc_ids == ["a", "b"]
    assert loaded.vectors.shape == (2, 2)
