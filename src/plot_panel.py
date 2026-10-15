"""Общий лист интерактивных графиков задач с согласованными шкалами."""

import copy
import json
import re
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from interactive_plot import create_interactive_depth_plot, format_axis_tick_labels


def create_plot_panel(tasks, output_path, progress=None, combine_scales=False, **parameters):
    """Объединяет существующие графики; глубина каждой строки независима."""
    if len(tasks) < 2:
        raise ValueError("Для панели выберите не менее двух задач.")
    normalized = [bool(t.get('cross_section_area_m2')) for t in tasks]
    if any(normalized) and not all(normalized):
        raise ValueError("Для сравнения KDE у всех задач должны совпадать единицы нормировки.")
    figures = []
    include_ctd = bool(parameters.get('ctd_columns')) and any(t.get('ctd_path') for t in tasks)
    for index, task in enumerate(tasks):
        result = create_interactive_depth_plot(
            track_sizes_path=task['track_sizes_path'], output_path=output_path,
            ctd_path=task.get('ctd_path'),
            cross_section_area_m2=task.get('cross_section_area_m2'),
            pleurobrachia_cross_section_area_m2=task.get('pleurobrachia_cross_section_area_m2'),
            return_figure=True, include_ctd_panel=include_ctd,
            **parameters,
        )
        if result is None:
            raise ValueError(f"{task['title']}: нет корректных данных размеров и глубин треков.")
        figures.append(result)
        if progress:
            progress(index + 1, len(tasks))

    # Ключ CTD — индекс выбранного параметра, KDE — колонка соответствующего вида.
    ranges = {}
    keys_by_row = []
    size_ranges = {}
    for fig, metadata in figures:
        for species, (low, high) in metadata['size_ranges'].items():
            previous = size_ranges.get(species, [low, high])
            size_ranges[species] = [min(low, previous[0]), max(high, previous[1])]
        keys = {}
        for trace in fig.data:
            if trace.meta and trace.meta.get('role') == 'ctd':
                keys[trace.meta['axis_layout_key']] = ('ctd', trace.meta['parameter_index'])
        for name in fig.layout:
            if re.fullmatch(r'xaxis\d*', name) and fig.layout[name].title.text:
                keys[name] = ('kde', name)
        for axis_name, key in keys.items():
            low, high = fig.layout[axis_name].range
            if key not in ranges:
                ranges[key] = [low, high]
            else:
                ranges[key] = [min(low, ranges[key][0]), max(high, ranges[key][1])]
        keys_by_row.append(keys)

    panel = go.Figure()
    annotations = []
    ctd_configs = []
    inset_configs = []
    row_configs = []
    legend_seen = set()
    axis_offset = 0
    shared_x_refs = {}
    count = len(figures)
    # Шкалу параметра оставляем у последней строки, где есть его данные CTD.
    scale_rows = {key: row for row, keys in enumerate(keys_by_row) for key in keys.values()}
    ctd_keys = [key for key in ranges if key[0] == 'ctd']
    scale_space = max(80, 45 * len(ctd_keys) + 30)
    compact_height = 500 * count + 35 * (count - 1) + scale_space
    shared_bottom = scale_space / compact_height
    for row, ((fig, metadata), task, keys) in enumerate(zip(figures, tasks, keys_by_row)):
        # Домены исходного графика сохраняются внутри собственной строки.
        bottom = (count - row - 1) / count
        height = 0.94 / count
        source_bottom, source_top = fig.layout.yaxis.domain
        if combine_scales:
            plot_top = 1 - row * 535 / compact_height
            plot_bottom = plot_top - 500 / compact_height

        def map_y(value):
            if combine_scales:
                return plot_bottom + (value - source_bottom) / (source_top - source_bottom) * (
                    plot_top - plot_bottom)
            return bottom + value * height

        axis_names = [name for name in fig.layout if re.fullmatch(r'[xy]axis\d*', name)]
        max_axis = max(int(re.search(r'\d+', name)[0]) if re.search(r'\d+', name) else 1
                       for name in axis_names)

        def remap(ref):
            if not ref or ref in ('paper', 'free'):
                return ref
            match = re.fullmatch(r'([xy])(\d*)( domain)?', ref)
            if not match:
                return ref
            number = int(match[2] or 1) + axis_offset
            return match[1] + (str(number) if number > 1 else '') + (match[3] or '')

        def layout_key(name):
            return remap(name.replace('axis', '')).replace(name[0], name[0] + 'axis', 1)

        for name in axis_names:
            axis = copy.deepcopy(fig.layout[name].to_plotly_json())
            for field in ('anchor', 'overlaying', 'matches', 'scaleanchor'):
                if field in axis:
                    axis[field] = remap(axis[field])
            if name.startswith('y') and 'domain' in axis:
                axis['domain'] = [map_y(value) for value in axis['domain']]
            if name.startswith('x') and 'position' in axis:
                axis['position'] = bottom + axis['position'] * height
            if name in keys:
                key = keys[name]
                axis['range'] = ranges[key]
                if key in shared_x_refs:
                    axis['matches'] = shared_x_refs[key]
                else:
                    shared_x_refs[key] = remap(name.replace('axis', ''))
                if key[0] == 'kde':
                    maximum = axis['range'][1] / 0.62
                    axis['tickvals'] = [-maximum / 2, -maximum / 4, 0, maximum / 4, maximum / 2]
                    axis['ticktext'] = format_axis_tick_labels([maximum, maximum / 2, 0, maximum / 2, maximum])
                else:
                    low, high = axis['range']
                    axis['tickvals'] = np.linspace(low, high, 5).tolist()
                    axis['ticktext'] = format_axis_tick_labels(axis['tickvals'])
                if combine_scales:
                    if key[0] == 'ctd':
                        axis['position'] = shared_bottom - (ctd_keys.index(key) + 1) * 45 / compact_height
                    if scale_rows[key] != row:
                        axis.update(showticklabels=False, ticks='', showline=False, title=dict(text=''))
            panel.update_layout({layout_key(name): axis})

        trace_offset = len(panel.data)
        annotation_offset = len(annotations)
        for trace in fig.data:
            data = copy.deepcopy(trace.to_plotly_json())
            for field in ('xaxis', 'yaxis'):
                data[field] = remap(data.get(field, field[0]))
            role = (data.get('meta') or {}).get('role')
            group = data.get('legendgroup')
            if role == 'ctd':
                group = f"ctd-{data['meta']['parameter_index']}"
                data['meta']['axis_layout_key'] = layout_key(data['meta']['axis_layout_key'])
            elif role == 'density_layer':
                group = f"density-layer-{data['meta']['level']}"
                data['name'] = f"SigmaT {data['meta']['level']:g}"
            elif role == 'thermocline':
                data['name'] = 'Термоклин'
            elif group:
                if group in size_ranges:
                    low, high = size_ranges[group]
                    data['name'] = f'{group}: {low:.1f}-{high:.1f} см'
                else:
                    data['name'] = f'{group}: размер н/д'
            if group:
                data['legendgroup'] = group
            if data.get('showlegend'):
                data['showlegend'] = group not in legend_seen
                legend_seen.add(group)
            if role in ('thermocline', 'density_layer'):
                axis_name = 'xaxis' + data['xaxis'][1:]
                data['x'] = list(panel.layout[axis_name].range)
            panel.add_trace(go.Scatter(data))

        ctd_annotations = {config['annotationIndex']: config for config in metadata['ctd']}
        for annotation_index, annotation in enumerate(fig.layout.annotations):
            data = copy.deepcopy(annotation.to_plotly_json())
            for field in ('xref', 'yref'):
                data[field] = remap(data.get(field, 'paper'))
            if data['yref'] == 'paper':
                data['y'] = map_y(data['y'])
                if combine_scales and annotation.y > source_top:
                    # Названия видов располагаем внутри верхнего края строки.
                    data['y'] = plot_top - 15 / compact_height
                    data['yanchor'] = 'top'
            if combine_scales and annotation_index in ctd_annotations:
                config = ctd_annotations[annotation_index]
                key = keys[config['axisKey']]
                data['y'] = shared_bottom - (ctd_keys.index(key) + 1) * 45 / compact_height
                data['visible'] = scale_rows[key] == row
            # Убираем повторяющуюся подсказку из каждой строки.
            if annotation.yref == 'paper' and annotation.y < 0:
                data['visible'] = False
            annotations.append(data)
        if combine_scales:
            annotations.append(dict(x=0, y=(plot_bottom + plot_top) / 2,
                                    xref='paper', yref='paper', xshift=-100,
                                    text=f"<b>{escape(task['title'])}</b>", textangle=-90,
                                    showarrow=False, font=dict(size=16),
                                    xanchor='center', yanchor='middle'))
        else:
            annotations.append(dict(x=0.5, y=bottom + 0.99 / count, xref='paper', yref='paper',
                                    text=f"<b>{escape(task['title'])}</b>", showarrow=False,
                                    font=dict(size=16), yanchor='top'))
        for config in metadata['ctd']:
            ctd_configs.append(dict(traceIndex=trace_offset + config['traceIndex'],
                                    axisKey=layout_key(config['axisKey']),
                                    annotationIndex=annotation_offset + config['annotationIndex'],
                                    showScale=not combine_scales or scale_rows[keys[config['axisKey']]] == row))
        if metadata['thermocline_annotation'] is not None:
            inset_configs.append(dict(group='thermocline',
                                      index=annotation_offset + metadata['thermocline_annotation'],
                                      traces=[trace_offset + i for i in metadata['thermocline']]))
        for indices, annotation_index in zip(metadata['layers'], metadata['layer_annotations']):
            inset_configs.append(dict(index=annotation_offset + annotation_index,
                                      traces=[trace_offset + i for i in indices]))
        row_configs.append(dict(yaxis=remap('y'), first=trace_offset, last=len(panel.data)))
        axis_offset += max_axis

    annotations.append(dict(x=0, y=-0.02, xref='paper', yref='paper', showarrow=False,
                            text='Клик по легенде — скрыть/показать во всех задачах. Размер точек соответствует размеру экземпляров.',
                            font=dict(size=10, color='gray'), xanchor='left'))

    panel.update_layout(
        width=max(fig.layout.width for fig, _ in figures),
        height=compact_height + 135 if combine_scales else 850 * count,
        margin=dict(t=55, b=80, l=160 if combine_scales else 100, r=240), annotations=annotations,
        plot_bgcolor='white', hovermode='closest',
        legend=dict(x=1.01, y=1, groupclick='togglegroup', tracegroupgap=8,
                    bgcolor='white', bordercolor='black', borderwidth=1),
    )
    output_path = Path(output_path).with_suffix('.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = PANEL_SCRIPT.replace('__CTD__', json.dumps(ctd_configs)).replace(
        '__INSETS__', json.dumps(inset_configs)).replace('__ROWS__', json.dumps(row_configs)).replace(
        '__FILENAME__', json.dumps(output_path.stem))
    panel.write_html(str(output_path), include_plotlyjs=True, post_script=script,
                     config=dict(displaylogo=False, toImageButtonOptions=dict(
                         format='svg', filename=output_path.stem, scale=2)))
    return str(output_path)


PANEL_SCRIPT = """
var plot = document.getElementById('{plot_id}');
var ctd = __CTD__, insets = __INSETS__, rows = __ROWS__;
var labelsVisible = true;
function visible(i) { return plot.data[i].visible !== false && plot.data[i].visible !== 'legendonly'; }
function sync() {
    var update = {};
    ctd.forEach(function(c) {
        update[c.axisKey + '.visible'] = visible(c.traceIndex);
        update['annotations[' + c.annotationIndex + '].visible'] = c.showScale && visible(c.traceIndex);
    });
    insets.forEach(function(c) {
        update['annotations[' + c.index + '].visible'] = labelsVisible && c.traces.some(visible);
    });
    Plotly.relayout(plot, update);
}
plot.on('plotly_restyle', sync);
plot.on('plotly_hover', function(e) {
    var point = e.points && e.points[0];
    if (!point || !point.data.meta || point.data.meta.role !== 'ctd') {
        Plotly.relayout(plot, {shapes:[]}); return;
    }
    var row = rows.find(function(r) { return point.curveNumber >= r.first && point.curveNumber < r.last; });
    var y = point.y;
    var axis = plot._fullLayout['yaxis' + row.yaxis.slice(1)];
    if (e.event && axis && axis.p2d) {
        y = axis.p2d(e.event.clientY - plot.getBoundingClientRect().top - axis._offset);
    }
    Plotly.relayout(plot, {shapes: [{type:'line', xref:'paper', x0:0, x1:1,
        yref:row.yaxis, y0:y, y1:y, line:{color:'rgba(0,0,0,0.45)', dash:'dot', width:1}}]});
});
plot.on('plotly_unhover', function() { Plotly.relayout(plot, {shapes:[]}); });
function button(text, top, callback) {
    var b = document.createElement('button');
    b.className = 'plotly-export-button'; b.textContent = text;
    b.style.cssText = 'position:fixed;right:12px;top:' + top + 'px;z-index:1000;padding:6px 12px;background:white;cursor:pointer;';
    b.onclick = callback; document.body.appendChild(b); return b;
}
var labels = button('Метки оси: вкл', 48, function() {
    labelsVisible = !labelsVisible;
    labels.textContent = labelsVisible ? 'Метки оси: вкл' : 'Метки оси: выкл'; sync();
});
button('PNG', 84, function() { Plotly.downloadImage(plot, {format:'png', filename:__FILENAME__, scale:2}); });
button('PDF', 120, function() { window.print(); });
var style = document.createElement('style');
style.textContent = '@media print { @page { size: ' + plot.layout.width + 'px ' + plot.layout.height + 'px; margin:0; }' +
    '.plotly-export-button,.modebar-container {display:none!important;} body {margin:0;} }';
document.head.appendChild(style);
sync();
"""
