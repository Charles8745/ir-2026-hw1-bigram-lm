import jieba
from hw1 import DICT_PATH


def test_dict_big_exists():
    assert DICT_PATH.exists()
    assert DICT_PATH.stat().st_size > 8_000_000


def test_jieba_big_dict_segments_traditional():
    jieba.setLogLevel(40)
    jieba.set_dictionary(str(DICT_PATH))
    jieba.initialize()
    assert jieba.lcut("我喜歡你。") == ["我", "喜歡", "你", "。"]
    assert jieba.lcut("台灣是一個島嶼") == ["台灣", "是", "一個", "島嶼"]
