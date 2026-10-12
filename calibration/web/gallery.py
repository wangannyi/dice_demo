"""Read-only, allowlisted access to the selected dataset's saved observations."""
import hashlib
import re
from pathlib import Path
from urllib.parse import urlencode

from calibration.web.common import read


def dataset(service):
    path = service.selected('dataset').resolve()
    # Only a selected acquisition dataset is exposed, never arbitrary query paths.
    if not (path/'manifest.json').is_file():
        raise FileNotFoundError('采样数据集不存在')
    return path, hashlib.sha256(str(path).encode()).hexdigest()[:24]


def local_file(directory, name):
    path = directory/name
    if path.resolve().parent != directory.resolve() or not path.is_file():
        raise FileNotFoundError('采样文件不存在')
    return path


def frame_files(service, sample, frame, key):
    path, current = dataset(service)
    if key != current or not re.fullmatch(r'sample_[0-9]{4,}', sample):
        raise ValueError('采样数据已更新，请刷新图片列表')
    source = local_file(path, sample+'.json')
    if frame == 'selected':
        directory, stem = path, sample
    else:
        if not re.fullmatch(r'frame_[0-9]{3}', frame):
            raise ValueError('无效检测帧')
        archive = read(source).get('frame_archive')
        if not archive:
            raise FileNotFoundError('历史样本未保存逐帧记录')
        directory = Path(archive).resolve()
        jobs = (service.data/'jobs').resolve()
        try:
            relative = directory.relative_to(jobs)
        except ValueError as exc:
            raise ValueError('逐帧记录不属于当前服务') from exc
        if (len(relative.parts) != 3 or relative.parts[1] != 'capture_frames'
                or not re.fullmatch(r'[0-9]{8}_[0-9]{6}_[a-f0-9]{8}', relative.parts[0])
                or not re.fullmatch(r'attempt_[0-9]{4,}', relative.parts[2])):
            raise ValueError('无效逐帧记录目录')
        stem = frame
        local_file(directory, stem+'.json')
    # Image names are derived here, not trusted from JSON or URL filenames.
    return directory, stem


def asset(service, sample, frame, key, kind):
    suffix = {'raw': '.png', 'detected': '_detected.png', 'data': '.json'}.get(kind)
    if suffix is None:
        raise ValueError('无效采样资源')
    directory, stem = frame_files(service, sample, frame, key)
    return local_file(directory, stem+suffix)


def catalog(service):
    try:
        path, key = dataset(service)
    except (FileNotFoundError, ValueError):
        return {'samples': [], 'dataset': None, 'key': None}
    samples = []
    for file in sorted(path.glob('sample_*.json')):
        if not re.fullmatch(r'sample_[0-9]{4,}', file.stem):
            continue
        try:
            sample = read(local_file(path, file.name))
            frames = [('selected', '最终样本')]
            for index in range(3):
                frame = f'frame_{index:03d}'
                try:
                    asset(service, file.stem, frame, key, 'data')
                    label = f'检测帧 {index+1}'
                    if sample.get('selected_frame_index') == index:
                        label += ' · 被选中'
                    frames.append((frame, label))
                except (OSError, ValueError):
                    pass
            entries = []
            for frame, label in frames:
                links = {}
                for kind in ('raw', 'detected', 'data'):
                    try:
                        asset(service, file.stem, frame, key, kind)
                        links[kind] = '/api/sample-file?'+urlencode(dict(
                            sample=file.stem, frame=frame, key=key, kind=kind))
                    except (OSError, ValueError):
                        links[kind] = None
                entries.append({'id': frame, 'label': label, **links})
            samples.append({'id': file.stem, 'number': int(file.stem[7:])+1, 'frames': entries})
        except (OSError, ValueError):
            continue  # Ignore partially written sample JSON during acquisition.
    return {'samples': samples, 'dataset': str(path), 'key': key}
