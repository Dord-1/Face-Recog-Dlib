"""Bảo vệ cấu trúc package: luật phụ thuộc một chiều giữa các module (xem docs/architecture.md).

Nếu test này fail sau khi bạn thêm `import`, hãy tự hỏi phụ thuộc mới có hợp lý không. Nếu có, cập nhật
ALLOWED_DEPENDENCIES bên dưới (và sơ đồ trong docs/architecture.md); nếu không, đổi chỗ đặt code.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / 'face_recog'

# module -> các module nội bộ được phép import. Module mới phải được khai báo ở đây.
ALLOWED_DEPENDENCIES = {
    'config': set(),
    'geometry': set(),
    'camera': set(),
    'smoothing': {'config'},
    'known_faces': {'config'},
    'matching': {'config', 'smoothing'},
    'quality': {'config', 'geometry'},
    'settings': {'config'},
    'activity_log': {'config'},
    'people': {'config', 'known_faces'},
    'recognizer': {'activity_log', 'camera', 'config', 'geometry', 'known_faces', 'matching',
                   'settings', 'smoothing'},
    'capture': {'activity_log', 'config', 'quality'},
    'gui': {'activity_log', 'camera', 'capture', 'people', 'recognizer', 'settings'},
    'tools.check_detect': {'config', 'geometry', 'known_faces'},
    'tools.evaluate': {'config', 'geometry', 'known_faces'},
}

# Chỉ các module giao diện được phép dùng Tkinter; lõi phải chạy được mà không cần GUI.
TKINTER_MODULES = {'capture', 'gui'}


def module_files():
    files = {}
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name in ('__init__.py', '__main__.py'):
            continue
        files['.'.join(path.relative_to(PACKAGE).with_suffix('').parts)] = path
    return files


def imports_of(path):
    """(các module face_recog được import, có import tkinter không)."""
    internal, uses_tk = set(), False
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f'{node.module}.{alias.name}' for alias in node.names]
        else:
            continue
        for name in names:
            if name.split('.')[0] == 'tkinter':
                uses_tk = True
            parts = name.split('.')
            if parts[0] == 'face_recog' and len(parts) > 1:
                internal.add('.'.join(parts[1:]))
    return internal, uses_tk


def internal_dependencies(files):
    """{module: các module nội bộ mà nó import}; chỉ giữ tên khớp một module thật."""
    deps = {}
    for name, path in files.items():
        imported, _ = imports_of(path)
        deps[name] = {m for m in imported if m in files and m != name}
    return deps


def test_every_module_is_declared():
    assert set(module_files()) == set(ALLOWED_DEPENDENCIES), (
        'Module mới/đã xoá chưa được cập nhật trong ALLOWED_DEPENDENCIES')


def test_dependencies_follow_the_allowed_map():
    deps = internal_dependencies(module_files())
    violations = {
        name: sorted(found - ALLOWED_DEPENDENCIES[name])
        for name, found in deps.items() if found - ALLOWED_DEPENDENCIES[name]
    }
    assert not violations, f'Phụ thuộc không được phép: {violations}'


def test_no_import_cycles():
    deps = internal_dependencies(module_files())
    visiting, done = [], set()

    def visit(node):
        if node in done:
            return
        assert node not in visiting, f'Import vòng: {" -> ".join([*visiting, node])}'
        visiting.append(node)
        for dep in sorted(deps[node]):
            visit(dep)
        visiting.pop()
        done.add(node)

    for name in deps:
        visit(name)


def test_only_gui_modules_import_tkinter():
    offenders = [name for name, path in module_files().items()
                 if imports_of(path)[1] and name not in TKINTER_MODULES]
    assert not offenders, f'Lõi không được import tkinter: {offenders}'


def test_core_never_imports_gui_or_tools():
    core = set(ALLOWED_DEPENDENCIES) - TKINTER_MODULES - {name for name in ALLOWED_DEPENDENCIES
                                                          if name.startswith('tools.')}
    forbidden = {'gui', 'capture'} | {n for n in ALLOWED_DEPENDENCIES if n.startswith('tools.')}
    deps = internal_dependencies(module_files())
    offenders = {name: sorted(deps[name] & forbidden) for name in core if deps[name] & forbidden}
    assert not offenders, f'Lõi import GUI/tools: {offenders}'


def test_root_launchers_only_call_the_package():
    root = PACKAGE.parent
    for launcher in ('Main.py', 'Check_Detect.py', 'evaluate.py'):
        imported, _ = imports_of(root / launcher)
        assert imported, f'{launcher} phải import từ face_recog'
        lines = [line for line in (root / launcher).read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(lines) <= 5, f'{launcher} chỉ nên là script mỏng (đang {len(lines)} dòng)'
