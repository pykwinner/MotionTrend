"""Native import selection tests using mock modules, without loading CUDA."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from model import point_ops


@pytest.fixture
def native_import(monkeypatch):
    native = Path(point_ops.__file__).resolve().parents[1] / 'tools/native'
    package = native / 'pointnet2'
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'pointnet2', raising=False)
    monkeypatch.delitem(sys.modules, 'pointnet2._ext', raising=False)
    monkeypatch.setattr(point_ops, '_ext', None)
    return native, package


def test_default_binding_resolves_relative_to_release_from_any_cwd(native_import, monkeypatch, tmp_path):
    native, package = native_import
    expected = SimpleNamespace(__file__=str(package / '_ext.so'))
    calls = []

    def importing(name):
        calls.append(name)
        assert sys.path[0] == str(native)
        return expected

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(point_ops.importlib, 'import_module', importing)
    assert point_ops.bind() is point_ops
    assert point_ops._ext is expected
    assert calls == ['pointnet2._ext']
    assert point_ops.bind() is point_ops
    assert sys.path.count(str(native)) == 1


@pytest.mark.parametrize('name', ['pointnet2', 'pointnet2._ext'])
def test_bundled_binding_rejects_already_imported_external_package(native_import, monkeypatch, tmp_path, name):
    native, package = native_import
    loaded = SimpleNamespace(__file__=str(tmp_path / 'external/pointnet2/module.py'))
    monkeypatch.setitem(sys.modules, name, loaded)
    monkeypatch.setattr(point_ops.importlib, 'import_module', lambda name: pytest.fail('Must reject before importing'))
    with pytest.raises(RuntimeError, match='already imported from a different source'):
        point_ops.bind()


def test_explicit_research_binding_uses_installed_resolution(native_import, monkeypatch, tmp_path):
    native, package = native_import
    expected = SimpleNamespace(__file__=str(tmp_path / 'installed/pointnet2/_ext.so'))
    sys.path.insert(0, str(native))

    def importing(name):
        assert str(native) not in sys.path
        return expected

    monkeypatch.setattr(point_ops.importlib, 'import_module', importing)
    assert point_ops.bind(prefer_bundled=False) is point_ops
    assert point_ops._ext is expected


def test_binding_cannot_switch_source_after_loading(native_import, monkeypatch):
    native, package = native_import
    monkeypatch.setitem(sys.modules, 'pointnet2', SimpleNamespace(__file__=str(package / '__init__.py')))
    with pytest.raises(RuntimeError, match='fresh Python process'):
        point_ops.bind(prefer_bundled=False)


def test_imported_extension_origin_is_checked(native_import, monkeypatch, tmp_path):
    monkeypatch.setattr(point_ops.importlib, 'import_module',
                        lambda name: SimpleNamespace(__file__=str(tmp_path / '_ext.so')))
    with pytest.raises(RuntimeError, match='resolved to a different source'):
        point_ops.bind()


def test_binding_option_requires_boolean(native_import):
    with pytest.raises(TypeError, match='real bool'):
        point_ops.bind(prefer_bundled=1)
