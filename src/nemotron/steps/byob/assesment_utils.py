import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from nemotron.steps.byob.translation.quality_metrics import AVAILABLE_QUALITY_METRICS

def get_statistics(df: pd.DataFrame) -> dict:
    for column in ['distractor_quality_score', 'chance_accuracy_score', 'consistency_score', 'rag_score', 'difficulty_score']:
        df[column] = df[column].astype(float)
    for metric in AVAILABLE_QUALITY_METRICS:
        df[f'{metric}_score'] = df[f'{metric}_score'].astype(float)
    QUESTION_FORMATS = list(df['question_format'].unique())
    def _get_statistics(df: pd.DataFrame) -> dict:
        question_format_df = pd.DataFrame({'question_format_score': QUESTION_FORMATS})
        question_format_df = pd.merge(
            question_format_df,
            df['question_format'].value_counts().reset_index().rename(columns={'question_format': 'question_format_score'}),
            on='question_format_score',
            how='left'
        ).fillna(0)
        question_format_df.sort_values(by='question_format_score', key=lambda x: x.map(lambda y: QUESTION_FORMATS.index(y)), inplace=True)

        parsing_failure_rate = {}
        for column in df.columns:
            if column.startswith('answer_easiness_') or column.startswith('answer_hallucination_'):
                failure_count = int(df.apply(lambda x: x[column] not in [chr(ord('A') + i) for i in range(len(x['choices_generated']))], axis=1).sum())
                parsing_failure_rate[column] = failure_count / len(df)
        parsing_failure_rate_mean = sum(parsing_failure_rate.values()) / len(parsing_failure_rate) if len(parsing_failure_rate) > 0 else np.nan
        stat = {
            'distractor_quality_score': float(df['distractor_quality_score'].mean()),
            'chance_accuracy_score': float(df['chance_accuracy_score'].mean()),
            'consistency_score': float(df['consistency_score'].mean()),
            'rag_score': float(df['rag_score'].mean()),
            'difficulty_score': float(df['difficulty_score'].mean()),
            'cognitive_level': float(df['cognitive_level'].mean()),
            'alignment_score': float(df['alignment_score'].mean()),
            'clarity_score': float(df['clarity_score'].mean()),
            'usefulness_score': float(df['usefulness_score'].mean()),
            'question_format_score': question_format_df,
            'parsing_failure_rate': parsing_failure_rate_mean,
            'count': len(df),
        }
        for translation_metric in AVAILABLE_QUALITY_METRICS:
          stat[f'{translation_metric}_score'] = float(df[f'{translation_metric}_score'].mean())
        return stat

    df_dict = {
        'all': df.copy(),
        'filtered': df[(~df['is_easy']) & (~df['is_hallucination'])] if 'is_easy' in df.columns and 'is_hallucination' in df.columns else df,
    }
    statistics = {
        'all': {},
        'filtered': {},
    }
    for key, _df in df_dict.items():
        subjects = _df['_subject'].unique()
        for subject in subjects:
            df_subject = _df[_df['_subject'] == subject]
            statistics[key][subject] = _get_statistics(df_subject)
        statistics[key]['_overall_'] = _get_statistics(_df)
    return statistics

def make_radar_plot(criteria, stats, output_path, ignore_subjects=None):
    def get_colour():
        colours = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'   # blue-teal
        ]
        def hex_to_rgb(hex_color):
            """Convert hex color string (#RRGGBB or #RGB) to RGBA string for plotly (0-255 values, floats allowed for alpha)"""
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        idx = 0
        while True:
            yield hex_to_rgb(colours[idx % len(colours)])
            idx += 1

    fig = go.Figure()
    color_gen = get_colour()
    for subject, stat in sorted(stats.items(), key=lambda x: x[0] if x[0] != '_overall_' else 'z'):
        if ignore_subjects and subject in ignore_subjects:
            continue
        stat_df = pd.DataFrame({'metric': key, 'value': stat[key]} for key in criteria)
        r = stat_df['value'].tolist()
        theta = stat_df['metric'].tolist()
        r.append(r[0])
        theta.append(theta[0])
        colour = next(color_gen)
        trace = go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself',
            name=subject,
            fillcolor=f'rgba({colour[0]}, {colour[1]}, {colour[2]}, 0.1)',
            line=dict(color=f'rgba({colour[0]}, {colour[1]}, {colour[2]}, 0.8)'),
        )
        fig.add_trace(trace)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 5]
            )),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    fig.write_image(output_path, scale=3)

def make_pie_chart(question_format_df: pd.DataFrame, output_path: str):
    fig = px.pie(question_format_df, values='count', names='question_format_score', title='Question Format Distribution')
    fig.update_traces(sort=False)
    fig.write_image(output_path, scale=3)

def make_data_card(statistics: dict):
    data_card = pd.DataFrame(statistics)
    data_card = data_card.drop(labels=['question_format_score']).T.reset_index()
    data_card.rename(columns={'index': 'subject'}, inplace=True)
    return data_card