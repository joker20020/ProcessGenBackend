import pytest
from unittest.mock import patch, MagicMock
from milvus_service import MilvusService


def test_singleton_returns_same_instance():
    a = MilvusService()
    b = MilvusService()
    assert a is b


def test_init_collection_calls_create_when_absent():
    svc = MilvusService.__new__(MilvusService)
    svc.client = MagicMock()
    svc.client.has_collection.return_value = False
    svc.client.create_schema.return_value = MagicMock()
    svc.client.prepare_index_params.return_value = MagicMock()
    svc.client.get_load_state.return_value = True
    with patch("milvus_service.EmbeddingService") as EmbMock:
        EmbMock.return_value.get_dimension.return_value = 1024
        result = svc.init_collection("capp")
    svc.client.create_collection.assert_called_once()
    assert result is True


def test_init_collection_returns_true_when_exists():
    svc = MilvusService.__new__(MilvusService)
    svc.client = MagicMock()
    svc.client.has_collection.return_value = True
    svc.client.get_load_state.return_value = True
    result = svc.init_collection("capp")
    assert result is True
    svc.client.create_collection.assert_not_called()
