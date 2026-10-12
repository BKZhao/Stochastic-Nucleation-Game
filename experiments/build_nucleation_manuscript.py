"""Convert this paper's Markdown/LaTeX hybrid drafts into matching section files.

Input grammar: plain paragraphs, #/## headings, displayed $$ equations, and
[FIGURE:key] markers. Mathematics, citations and references retain LaTeX syntax.
This is a deliberately narrow manuscript converter, not a general Markdown parser.
"""
import argparse
import json
import re
from pathlib import Path

from research_common import PROJECT_ROOT, file_hash, write_json
from nucleation_paper_inputs import REPORTS

PAPER = PROJECT_ROOT/'paper'


def figure_block(key: str, figures: dict) -> str:
    figure = figures[key]
    environment = figure.get('environment', 'figure*')
    if environment not in ('appendixfigure', 'figure*'):
        raise ValueError('unsupported figure environment')
    placement = '' if environment == 'appendixfigure' else '[tp]'
    width = figure.get('width', 1.0)
    if not isinstance(width, (int, float)) or not 0 < width <= 1:
        raise ValueError('figure width must be a fraction in (0, 1]')
    return (
        f'\n\\begin{{{environment}}}{placement}\n\\centering\n'
        f"\\includegraphics[width={width:g}\\textwidth]{{{figure['asset']}}}\n"
        f"\\caption{{{figure['caption']}}}\n"
        f"\\label{{fig:{key}}}\n\\end{{{environment}}}\n"
    )


def convert(text: str, figures: dict) -> str:
    text = re.sub(r'<!--.*?-->\s*', '', text, flags=re.S)
    text = re.sub(r'^## (.+)$', r'\\subsection{\1}', text, flags=re.M)
    text = re.sub(r'^# (.+)$', r'\\section{\1}', text, flags=re.M)
    text = re.sub(r'\[FIGURE:([a-z]+)\]',
                  lambda match: figure_block(match[1], figures), text)
    pieces = text.split('$$')
    if len(pieces) % 2 == 0:
        raise ValueError('unpaired display-math delimiter')
    for index in range(1, len(pieces), 2):
        pieces[index] = '\\begin{equation}\n'+pieces[index].strip()+'\n\\end{equation}'
    text = ''.join(pieces)
    text = re.sub(r'(?<!\\)%', r'\\%', text)
    return text.replace('–', '--').replace('—', '---')


def verified_figures(figures: dict, reuse_figures: bool = False) -> dict:
    directory = PROJECT_ROOT/'figures/nucleation/paper'
    missing_inputs = {}
    for figure in figures.values():
        manifest = json.loads((directory/Path(figure['asset']).with_suffix('.json')).read_text())
        for group in ('inputs', 'plotting_sources', 'assets'):
            for item in manifest[group]:
                path = PROJECT_ROOT/item['path']
                if reuse_figures and group == 'inputs' and not path.is_file():
                    missing_inputs[item['path']] = item['sha256']
                    continue
                if file_hash(path) != item['sha256']:
                    raise ValueError(f'stale figure input, source or asset: {path}')
    return {'mode': 'reuse_frozen_figures' if reuse_figures else 'full_inputs',
            'checked_manifests': len(figures),
            'missing_raw_inputs': missing_inputs,
            'scope': 'All assets, plotting sources, and available inputs are hash-checked. '
                     'Missing inputs are not revalidated in reuse mode.'}


def phase_values() -> None:
    pointer = json.loads((PAPER/'phase_inputs.json').read_text())
    path = PROJECT_ROOT/pointer['analysis']
    if file_hash(path) != pointer['sha256']:
        raise ValueError('phase analysis changed after figure generation')
    regimes = json.loads((PAPER/'regimes_inputs.json').read_text())
    if (regimes['phase_sha256'] != pointer['sha256']
            or file_hash(PROJECT_ROOT/regimes['analysis']) != regimes['sha256']):
        raise ValueError('regime figures and probability figures use different analyses')
    report = json.loads(path.read_text())
    manifest = json.loads((PROJECT_ROOT/'figures/nucleation/paper/fig01_phase.json').read_text())
    if not any(item['sha256'] == pointer['sha256'] for item in manifest['inputs']):
        raise ValueError('phase figure and manuscript refer to different analyses')
    values = {'PhaseBPoints': len(report['axes']['model.temptation']),
              'PhaseDPoints': len(report['axes']['model.diffusion']),
              'PhaseRuns': report['n_runs']}
    for shape, prefix in [('compact', 'Compact'), ('random', 'Random')]:
        cells = [r for r in report['cells'] if r['shape'] == shape
                 and abs(r['b']-1.16) < 1e-10 and abs(r['diffusion']-.4) < 1e-10]
        if len(cells) != 1 or cells[0]['n'] != 64:
            raise ValueError('reference cell missing or sample size changed')
        values.update({prefix+k: v for k, v in cells[0]['counts'].items()})
    (PAPER/'phase_values.tex').write_text(
        '% Generated from the complete analysis referenced by phase_inputs.json.\n'+
        ''.join(f'\\newcommand{{\\{key}}}{{{value}}}\n' for key, value in values.items()))
    dense = values['PhaseBPoints'] == values['PhaseDPoints'] == 41
    status = ('密集网格已完整分析并进入稿件。' if dense else
              '当前图和 TeX 使用已完成的探索网格；41×41 密集网格尚未纳入稿件。')
    (PAPER/'PHASE_STATUS.md').write_text(
        '# 当前相图输入\n\n'+status+'\n\n'
        f"网格：{values['PhaseBPoints']}×{values['PhaseDPoints']}；"
        f"总运行数：{values['PhaseRuns']}；每格每种几何 64 条轨迹。\n\n"
        f"分析：{pointer['analysis']}。\n"
        '本文件记录图与 TeX 的输入；最近一次 PDF 构建以 pdf_build.json 为准。\n'
        '本文件由 build_nucleation_manuscript.py 生成，phase_inputs.json 是输入指针。\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-figures', action='store_true',
                        help='Reuse frozen figures after checking assets and available inputs; '
                             'record unavailable raw inputs without claiming full revalidation.')
    args = parser.parse_args()
    figures = json.loads((PAPER/'figure_captions.json').read_text())
    figure_validation = verified_figures(figures, args.reuse_figures)
    phase_values()
    target = PAPER/'sections'
    target.mkdir(exist_ok=True)
    drafts = sorted((PAPER/'drafts').glob('*.md'))
    documents = [PAPER/'main.tex', PAPER/'supplement.tex']
    owners = {p.stem: re.findall(r'\\input\{sections/([^}]+)\}', p.read_text())
              for p in documents}
    sections = [section for names in owners.values() for section in names]
    if len(sections) != len(set(sections)) or set(sections) != {p.stem for p in drafts}:
        raise ValueError('each Markdown draft must belong to exactly one document')
    supplemental_text = '\n'.join(
        (PAPER/'drafts'/f'{name}.md').read_text() for name in owners['supplement'])
    supplemental_keys = sorted({key.strip()
        for group in re.findall(r'\\cite\w*\{([^}]+)\}', supplemental_text)
        for key in group.split(',')})
    (PAPER/'supplement_citations.tex').write_text(
        '% Generated: APS requires supplemental sources in the article bibliography.\n'
        + '\\nocite{' + ','.join(supplemental_keys) + '}\n')
    for draft in drafts:
        (target/draft.with_suffix('.tex').name).write_text(convert(draft.read_text(), figures))
    inputs = [*drafts, *documents, PAPER/'preamble.tex', PAPER/'frontmatter.tex',
              PAPER/'math_commands.tex', PAPER/'supplement_citations.tex',
              PAPER/'references.bib',
              PAPER/'figure_captions.json', PAPER/'phase_inputs.json',
              PAPER/'physics_inputs.json', PAPER/'regimes_inputs.json',
              PAPER/'response_inputs.json', PAPER/'structure_inputs.json', PAPER/'microstructure_inputs.json', PAPER/'phase_values.tex',
              Path(__file__), *REPORTS.values()]
    write_json(PAPER/'evidence_inputs.json', {
        name: {'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)}
        for name, path in REPORTS.items()})
    write_json(PAPER/'build_inputs.json', {
        'inputs': [{'path': str(p.relative_to(PROJECT_ROOT)), 'sha256': file_hash(p)}
                   for p in inputs],
        'documents': owners,
        'figure_validation': figure_validation,
        'sections': [str(p.relative_to(PROJECT_ROOT)) for p in sorted(target.glob('*.tex'))],
        'figures': [v['asset'] for v in figures.values()],
        'note': 'Input/source consistency checked; PDF compilation is a separate step.'})
    print(f'Converted {len(drafts)} complete section drafts.')


if __name__ == '__main__':
    main()
