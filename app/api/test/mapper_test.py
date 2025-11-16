"""
unit tests for API mapper functions.

tests the conversion from API request models to internal DataSource models.
"""

from app.models.api import Request, ContentItem, ContentType
from app.api.mapper import (
    map_content_type_to_source_type,
    request_to_data_sources,
)


# ===== map_content_type_to_source_type tests =====

def test_map_text_content_type():
    """should map TEXT to original_text"""
    result = map_content_type_to_source_type(ContentType.TEXT)
    assert result == "original_text"


def test_map_image_content_type():
    """should map IMAGE to image"""
    result = map_content_type_to_source_type(ContentType.IMAGE)
    assert result == "image"


def test_map_audio_content_type():
    """should map AUDIO to audio_transcript"""
    result = map_content_type_to_source_type(ContentType.AUDIO)
    assert result == "audio_transcript"


# ===== request_to_data_sources tests =====

def test_single_text_content_item():
    """should convert single text content item to data source"""
    request = Request(content=[
        ContentItem(textContent="Neymar voltou ao Santos", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 1
    assert sources[0].source_type == "original_text"
    assert sources[0].original_text == "Neymar voltou ao Santos"
    assert sources[0].locale == "pt-BR"


def test_single_image_content_item():
    """should convert single image content item to data source"""
    request = Request(content=[
        ContentItem(textContent="Image shows a news headline", type=ContentType.IMAGE)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 1
    assert sources[0].source_type == "image"
    assert sources[0].original_text == "Image shows a news headline"


def test_single_audio_content_item():
    """should convert single audio content item to data source"""
    request = Request(content=[
        ContentItem(textContent="Transcribed audio: speaker says...", type=ContentType.AUDIO)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 1
    assert sources[0].source_type == "audio_transcript"
    assert sources[0].original_text == "Transcribed audio: speaker says..."


def test_multiple_content_items():
    """should convert multiple content items to separate data sources"""
    request = Request(content=[
        ContentItem(textContent="Check this out", type=ContentType.TEXT),
        ContentItem(textContent="Image shows data", type=ContentType.IMAGE),
        ContentItem(textContent="Audio says something", type=ContentType.AUDIO)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 3
    assert sources[0].source_type == "original_text"
    assert sources[0].original_text == "Check this out"
    assert sources[1].source_type == "image"
    assert sources[1].original_text == "Image shows data"
    assert sources[2].source_type == "audio_transcript"
    assert sources[2].original_text == "Audio says something"


def test_empty_content_list():
    """should return empty list for empty content"""
    request = Request(content=[])
    
    sources = request_to_data_sources(request)
    
    assert sources == []


def test_custom_locale():
    """should use custom locale when provided"""
    request = Request(content=[
        ContentItem(textContent="Hello world", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request, locale="en-US")
    
    assert len(sources) == 1
    assert sources[0].locale == "en-US"


def test_default_locale():
    """should use pt-BR as default locale"""
    request = Request(content=[
        ContentItem(textContent="Olá mundo", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert sources[0].locale == "pt-BR"


def test_unique_ids_generated():
    """should generate unique IDs for each data source"""
    request = Request(content=[
        ContentItem(textContent="First", type=ContentType.TEXT),
        ContentItem(textContent="Second", type=ContentType.TEXT),
        ContentItem(textContent="Third", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    # extract all IDs
    ids = [source.id for source in sources]
    
    # check all IDs are unique
    assert len(ids) == len(set(ids))
    
    # check IDs are not empty
    for source_id in ids:
        assert source_id != ""
        assert len(source_id) > 0


def test_timestamp_is_set():
    """should set timestamp on all data sources"""
    request = Request(content=[
        ContentItem(textContent="Content", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert sources[0].timestamp is not None
    assert sources[0].timestamp != ""
    # check it ends with Z (UTC timezone marker)
    assert sources[0].timestamp.endswith("Z")


def test_same_timestamp_for_all_items():
    """should use same timestamp for all items in a single request"""
    request = Request(content=[
        ContentItem(textContent="First", type=ContentType.TEXT),
        ContentItem(textContent="Second", type=ContentType.IMAGE),
        ContentItem(textContent="Third", type=ContentType.AUDIO)
    ])
    
    sources = request_to_data_sources(request)
    
    # all timestamps should be the same
    timestamps = [source.timestamp for source in sources]
    assert len(set(timestamps)) == 1


def test_empty_metadata():
    """should have empty metadata dict by default"""
    request = Request(content=[
        ContentItem(textContent="Text", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert sources[0].metadata == {}


def test_long_text_content():
    """should handle long text content"""
    long_text = "A" * 5000  # 5000 characters
    request = Request(content=[
        ContentItem(textContent=long_text, type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 1
    assert sources[0].original_text == long_text
    assert len(sources[0].original_text) == 5000


def test_special_characters_in_text():
    """should preserve special characters in text"""
    special_text = "Text with émojis 🎉 and spëcial çharacters & symbols: @#$%"
    request = Request(content=[
        ContentItem(textContent=special_text, type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert sources[0].original_text == special_text


def test_multiline_text_content():
    """should preserve multiline text content"""
    multiline_text = """First line
Second line
Third line"""
    request = Request(content=[
        ContentItem(textContent=multiline_text, type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert sources[0].original_text == multiline_text
    assert "\n" in sources[0].original_text


def test_mixed_content_types():
    """should correctly handle mixed content types in correct order"""
    request = Request(content=[
        ContentItem(textContent="Text content", type=ContentType.TEXT),
        ContentItem(textContent="Audio content", type=ContentType.AUDIO),
        ContentItem(textContent="Image content", type=ContentType.IMAGE),
        ContentItem(textContent="More text", type=ContentType.TEXT)
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 4
    # verify order is preserved
    assert sources[0].source_type == "original_text"
    assert sources[1].source_type == "audio_transcript"
    assert sources[2].source_type == "image"
    assert sources[3].source_type == "original_text"
    
    # verify content
    assert sources[0].original_text == "Text content"
    assert sources[1].original_text == "Audio content"
    assert sources[2].original_text == "Image content"
    assert sources[3].original_text == "More text"


def test_real_world_whatsapp_example():
    """should handle real-world WhatsApp message example"""
    request = Request(content=[
        ContentItem(
            textContent="Neymar está dando tudo de si para tentar tirar o Santos da zona de rebaixamento",
            type=ContentType.TEXT
        )
    ])
    
    sources = request_to_data_sources(request)
    
    assert len(sources) == 1
    assert sources[0].source_type == "original_text"
    assert "Neymar" in sources[0].original_text
    assert "Santos" in sources[0].original_text
    assert sources[0].locale == "pt-BR"

