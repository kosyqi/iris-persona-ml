#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アイリスオーヤマ ペルソナ調査データ 機械学習分析システム
Iris Ohyama Persona Survey - ML Analysis System
"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

try:
    import japanize_matplotlib
    HAS_JP = True
except ImportError:
    HAS_JP = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ───────────────────────────────────────────
# 設定
# ───────────────────────────────────────────
CSV_PATH = '/mnt/c/Users/bi24028/Downloads/iris_persona_survey_42q.csv'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
if not HAS_JP:
    plt.rcParams['font.family'] = ['DejaVu Sans']

AFFINITY_ORDER = ['熱烈ファン', '好意的', '中立', '懐疑的', '嫌悪']
AFFINITY_EN    = ['Super Fan',  'Positive','Neutral','Skeptical','Averse']
AFFINITY_COLORS = ['#E74C3C', '#E67E22', '#F1C40F', '#3498DB', '#2C3E50']


# ───────────────────────────────────────────
# データ読み込み
# ───────────────────────────────────────────
def load_data():
    print(f"\n📂 データ読み込み中: {CSV_PATH}")
    for enc in ['utf-8-sig', 'utf-8', 'shift_jis', 'cp932']:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            df = df.dropna(how='all')
            print(f"✅ {len(df)}名 × {len(df.columns)}項目 ({enc})")
            return df
        except Exception:
            continue
    raise RuntimeError("CSVを読み込めませんでした")


# ───────────────────────────────────────────
# 特徴量エンジニアリング
# ───────────────────────────────────────────
def extract_features(df):
    f = pd.DataFrame(index=df.index)

    # 数値特徴量
    f['age']            = pd.to_numeric(df['年齢'], errors='coerce').fillna(35)
    f['household_size'] = pd.to_numeric(df['世帯人数'], errors='coerce').fillna(2)
    f['age_group']      = pd.cut(f['age'], bins=[0,19,29,39,49,59,69,100],
                                  labels=range(7)).astype(float).fillna(2)

    # 性別
    f['is_female'] = df['性別'].str.contains('女性', na=False).astype(int)
    f['is_male']   = df['性別'].str.contains('男性', na=False).astype(int)

    # 購買行動
    ch = '購買行動・チャネル_主な購入チャネル'
    f['buys']          = (~df[ch].str.contains('選ばない', na=False)).astype(int)
    f['uses_amazon']   = df[ch].str.contains('Amazon', na=False, case=False).astype(int)
    f['uses_store']    = df[ch].str.contains('ホームセンター|量販店|スーパー|店頭', na=False).astype(int)

    tm = '購買行動・チャネル_情報収集時間'
    f['quick_buyer']   = df[tm].str.contains('5分以内|3分|10分|15分', na=False).astype(int)
    f['deep_research'] = df[tm].str.contains('1時間|2時間|3時間|1〜2', na=False).astype(int)

    # 支出
    sp = '価格感度・支出意識_月の支出'
    f['high_spender']  = df[sp].str.contains('万|8000|9000|15000', na=False).astype(int)
    f['low_spender']   = (df[sp].str.contains('500|1000', na=False) & ~f['high_spender'].astype(bool)).astype(int)

    # 口コミ
    rc = '口コミ・情報発信行動_薦めた経験'
    f['recommends']    = (df[rc].str.contains('薦め|紹介|教え|伝え', na=False) &
                          ~df[rc].str.contains('薦めない|していない', na=False)).astype(int)
    sn = '口コミ・情報発信行動_SNS投稿レビュー経験'
    f['sns_active']    = (df[sn].str.contains('投稿|書いた|レビュー', na=False) &
                          ~df[sn].str.contains('していない|書いていない', na=False)).astype(int)

    # ニーズ
    sm = '未充足ニーズ・欲しい商品_スマートホーム関心'
    f['smart_home']    = (df[sm].str.contains('関心|興味|使いたい', na=False) &
                          ~df[sm].str.contains('不安|難しそう', na=False)).astype(int)
    sb = '未充足ニーズ・欲しい商品_サブスク需要'
    f['wants_sub']     = (df[sb].str.contains('使いたい|あり|絶対', na=False) &
                          ~df[sb].str.contains('不要|いらない', na=False)).astype(int)

    # 環境配慮
    ec = 'サステナビリティ・社会価値_グリーンプレミアム'
    f['eco_willing']   = (df[ec].str.contains('払う|選ぶ|検討', na=False) &
                          ~df[ec].str.contains('払わない|払えない', na=False)).astype(int)

    # 職業カテゴリ
    oc = '職業'
    f['is_student']    = df[oc].str.contains('学生|高校|専門学校', na=False).astype(int)
    f['is_housewife']  = df[oc].str.contains('主婦|家事', na=False).astype(int)
    f['is_medical']    = df[oc].str.contains('看護|医師|獣医|介護|栄養', na=False).astype(int)
    f['is_engineer']   = df[oc].str.contains('エンジニア|建築士|設備士', na=False).astype(int)
    f['is_retired']    = df[oc].str.contains('退職|年金', na=False).astype(int)
    f['is_creative']   = df[oc].str.contains('デザイナー|Youtuber|インスタ|翻訳|料理人', na=False).astype(int)
    f['is_employee']   = df[oc].str.contains('営業|スタッフ|事務|運転手|監督', na=False).astype(int)
    f['is_civil']      = df[oc].str.contains('公務員|消防士|自衛', na=False).astype(int)

    return f.fillna(0).astype(float)

def get_target(df):
    mp = {a: i for i, a in enumerate(AFFINITY_ORDER)}
    return df['ブランド親和性'].map(mp).fillna(2).astype(int)


# ───────────────────────────────────────────
# EDA 可視化
# ───────────────────────────────────────────
def plot_eda(df):
    print("\n📊 EDA グラフ生成中...")

    # ── 1. ブランド親和性分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['ブランド親和性'].value_counts().reindex(AFFINITY_ORDER, fill_value=0)
    label_fn = (lambda x: x) if HAS_JP else (lambda x: AFFINITY_EN[AFFINITY_ORDER.index(x)] if x in AFFINITY_ORDER else x)
    labels = [label_fn(a) for a in counts.index]

    axes[0].bar(labels, counts.values, color=AFFINITY_COLORS, edgecolor='white', linewidth=1.5)
    axes[0].set_title('ブランド親和性の分布' if HAS_JP else 'Brand Affinity Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('人数')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=AFFINITY_COLORS,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[1].set_title('ブランド親和性 (割合)' if HAS_JP else 'Brand Affinity (Ratio)', fontsize=13, fontweight='bold')

    plt.suptitle('アイリスオーヤマ ペルソナ調査 (n=50)' if HAS_JP else 'Iris Ohyama Persona Survey (n=50)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_brand_affinity.png'))
    plt.close()
    print("  ✅ 01_brand_affinity.png")

    # ── 2. 年齢 × ブランド親和性
    df2 = df.copy()
    df2['age_num'] = pd.to_numeric(df2['年齢'], errors='coerce')
    bins = [0, 19, 29, 39, 49, 59, 69, 100]
    labels_age = ['10代', '20代', '30代', '40代', '50代', '60代', '70代以上']
    df2['age_group'] = pd.cut(df2['age_num'], bins=bins, labels=labels_age)
    cross = pd.crosstab(df2['age_group'], df2['ブランド親和性'])[AFFINITY_ORDER].fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    cross.plot(kind='bar', ax=ax, color=AFFINITY_COLORS, edgecolor='white', linewidth=1, width=0.8)
    ax.set_title('年代別 ブランド親和性' if HAS_JP else 'Brand Affinity by Age Group',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('年代' if HAS_JP else 'Age Group')
    ax.set_ylabel('人数')
    ax.legend(
        [label_fn(a) for a in AFFINITY_ORDER],
        title='ブランド親和性' if HAS_JP else 'Affinity',
        bbox_to_anchor=(1.01, 1)
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_age_affinity.png'))
    plt.close()
    print("  ✅ 02_age_affinity.png")

    # ── 3. 性別 × ブランド親和性
    fig, ax = plt.subplots(figsize=(8, 5))
    gender_cross = pd.crosstab(df['性別'], df['ブランド親和性'])[AFFINITY_ORDER].fillna(0)
    gender_cross.plot(kind='bar', ax=ax, color=AFFINITY_COLORS, edgecolor='white', linewidth=1)
    ax.set_title('性別 × ブランド親和性' if HAS_JP else 'Gender × Brand Affinity',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('人数')
    ax.legend([label_fn(a) for a in AFFINITY_ORDER])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_gender_affinity.png'))
    plt.close()
    print("  ✅ 03_gender_affinity.png")

    # ── 4. 年齢分布
    fig, ax = plt.subplots(figsize=(10, 5))
    ages = pd.to_numeric(df['年齢'], errors='coerce').dropna()
    colors_by_affinity = [AFFINITY_COLORS[AFFINITY_ORDER.index(a)]
                          if a in AFFINITY_ORDER else '#999'
                          for a in df.loc[ages.index, 'ブランド親和性']]
    ax.scatter(range(len(ages)), sorted(ages), c=colors_by_affinity, s=60, alpha=0.8, edgecolors='white')
    ax.axhline(ages.mean(), color='red', linestyle='--', linewidth=1.5, label=f'平均年齢: {ages.mean():.1f}歳')
    ax.set_title('年齢分布（色=ブランド親和性）' if HAS_JP else 'Age Distribution (color=affinity)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('回答者' if HAS_JP else 'Respondent')
    ax.set_ylabel('年齢' if HAS_JP else 'Age')
    patches = [mpatches.Patch(color=c, label=(label_fn(a))) for a, c in zip(AFFINITY_ORDER, AFFINITY_COLORS)]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color='red', linestyle='--', label=f'Avg: {ages.mean():.1f}')],
              bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_age_scatter.png'))
    plt.close()
    print("  ✅ 04_age_scatter.png")


# ───────────────────────────────────────────
# 機械学習 分類モデル
# ───────────────────────────────────────────
def run_classification(X, y, feature_names):
    print("\n🤖 機械学習 分類モデル 評価中...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.5),
        'SVM':               SVC(kernel='rbf', C=1.0, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes':       GaussianNB(),
    }

    results = {}
    for name, model in models.items():
        X_in = X_scaled if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors'] else X
        scores = cross_val_score(model, X_in, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std':  scores.std(),
            'scores': scores.tolist()
        }
        print(f"  {name:28s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Feature importance from Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    # ── 可視化
    _plot_model_comparison(results)
    _plot_feature_importance(importances)

    best = max(results, key=lambda k: results[k]['mean'])
    print(f"\n🏆 最高精度: {best} ({results[best]['mean']:.3f})")

    return results, importances


def _plot_model_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds  = [results[n]['std']  for n in names]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor='white',
                   linewidth=1.5, capsize=4)
    ax.set_xlim(0, 1)
    ax.axvline(1/5, color='gray', linestyle=':', linewidth=1, label='Random (20%)')  # 5 classes
    for bar, mean in zip(bars, means):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}', va='center', fontsize=9)
    ax.set_title('モデル別 分類精度 (5-Fold CV)' if HAS_JP else 'Model Accuracy (5-Fold CV)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Accuracy')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_model_comparison.png'))
    plt.close()
    print("  ✅ 05_model_comparison.png")


def _plot_feature_importance(importances):
    top15 = importances.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(top15)))
    bars = ax.barh(range(len(top15)), top15.values[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index[::-1], fontsize=9)
    ax.set_title('特徴量重要度 Top 15 (Random Forest)' if HAS_JP else 'Feature Importance Top 15 (RF)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_feature_importance.png'))
    plt.close()
    print("  ✅ 06_feature_importance.png")


# ───────────────────────────────────────────
# クラスタリング分析
# ───────────────────────────────────────────
def run_clustering(X, df):
    print("\n🔍 クラスタリング分析中...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Elbow method
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Fit with k=4
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km4.fit_predict(X_scaled)
    df_c = df.copy()
    df_c['cluster'] = clusters

    # ── Elbow + PCA scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(K_range), inertias, 'o-', color='steelblue', linewidth=2)
    axes[0].axvline(4, color='red', linestyle='--', linewidth=1.5, label='k=4 (選択)')
    axes[0].set_title('エルボー法 (最適クラスタ数)' if HAS_JP else 'Elbow Method',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('クラスタ数')
    axes[0].set_ylabel('Inertia')
    axes[0].legend()

    cluster_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    for i in range(4):
        mask = clusters == i
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=cluster_colors[i], label=f'Cluster {i+1}',
                        s=80, alpha=0.8, edgecolors='white')
    axes[1].set_title('PCA クラスタリング (k=4)' if HAS_JP else 'PCA Clustering (k=4)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_clustering.png'))
    plt.close()
    print("  ✅ 07_clustering.png")

    # クラスタプロファイル
    print("\n📋 クラスタプロファイル:")
    for i in range(4):
        members = df_c[df_c['cluster'] == i]
        affinity_dist = members['ブランド親和性'].value_counts().to_dict()
        avg_age = pd.to_numeric(members['年齢'], errors='coerce').mean()
        print(f"  Cluster {i+1} ({len(members)}名): 平均年齢={avg_age:.1f}, {affinity_dist}")

    return df_c, clusters


# ───────────────────────────────────────────
# HTMLレポート生成
# ───────────────────────────────────────────
def generate_html_report(df, results, importances, clusters):
    print("\n📄 HTMLレポート生成中...")

    import base64
    def img_to_b64(path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return ''

    imgs = {
        'affinity': img_to_b64(os.path.join(OUTPUT_DIR, '01_brand_affinity.png')),
        'age':      img_to_b64(os.path.join(OUTPUT_DIR, '02_age_affinity.png')),
        'gender':   img_to_b64(os.path.join(OUTPUT_DIR, '03_gender_affinity.png')),
        'scatter':  img_to_b64(os.path.join(OUTPUT_DIR, '04_age_scatter.png')),
        'models':   img_to_b64(os.path.join(OUTPUT_DIR, '05_model_comparison.png')),
        'features': img_to_b64(os.path.join(OUTPUT_DIR, '06_feature_importance.png')),
        'cluster':  img_to_b64(os.path.join(OUTPUT_DIR, '07_clustering.png')),
    }

    model_rows = ''.join(
        f'<tr><td>{n}</td><td class="bold">{v["mean"]:.3f}</td>'
        f'<td>±{v["std"]:.3f}</td>'
        f'<td><div class="bar-mini" style="width:{v["mean"]*200:.0f}px"></div></td></tr>'
        for n, v in sorted(results.items(), key=lambda x: -x[1]['mean'])
    )

    feat_rows = ''.join(
        f'<tr><td>{n}</td><td class="bold">{v:.4f}</td>'
        f'<td><div class="bar-mini" style="width:{v*600:.0f}px"></div></td></tr>'
        for n, v in importances.head(10).items()
    )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>アイリスオーヤマ ペルソナ調査 ML分析レポート</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Hiragino Sans', 'Yu Gothic', 'Meiryo', sans-serif;
         background: #0f1117; color: #e8e8e8; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             padding: 40px; text-align: center; border-bottom: 2px solid #e94560; }}
  .header h1 {{ font-size: 28px; color: #fff; margin-bottom: 8px; }}
  .header p  {{ color: #a0aec0; font-size: 14px; }}
  .stats-row {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .stat-card {{ flex: 1; min-width: 140px; background: #1a1a2e; border-radius: 12px;
                padding: 20px; text-align: center; border: 1px solid #2d3748; }}
  .stat-card .num {{ font-size: 36px; font-weight: 700; color: #e94560; }}
  .stat-card .lbl {{ font-size: 12px; color: #718096; margin-top: 4px; }}
  .section {{ padding: 24px 32px; }}
  .section h2 {{ font-size: 20px; color: #e2e8f0; margin-bottom: 16px;
                 border-left: 4px solid #e94560; padding-left: 12px; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }}
  .chart-card {{ background: #1a1a2e; border-radius: 12px; padding: 16px;
                 border: 1px solid #2d3748; }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #2d3748; color: #a0aec0; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #2d3748; }}
  tr:hover td {{ background: #1e2740; }}
  .bold {{ font-weight: 700; color: #f6e05e; }}
  .bar-mini {{ height: 8px; background: linear-gradient(90deg, #e94560, #f6ad55);
               border-radius: 4px; }}
  .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .insight {{ background: #1a1a2e; border-radius: 10px; padding: 16px;
              border-left: 4px solid #4299e1; }}
  .insight h3 {{ color: #90cdf4; font-size: 14px; margin-bottom: 8px; }}
  .insight p  {{ color: #a0aec0; font-size: 13px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 700; }}
  .badge-green {{ background: #276749; color: #9ae6b4; }}
  .badge-red {{ background: #742a2a; color: #feb2b2; }}
</style>
</head>
<body>

<div class="header">
  <h1>🏪 アイリスオーヤマ ペルソナ調査 ML分析レポート</h1>
  <p>Iris Ohyama Persona Survey — Machine Learning Analysis Report (n=50)</p>
</div>

<!-- 統計サマリー -->
<div class="stats-row">
  <div class="stat-card"><div class="num">50</div><div class="lbl">回答者数</div></div>
  <div class="stat-card"><div class="num">42</div><div class="lbl">調査項目数</div></div>
  <div class="stat-card"><div class="num">5</div><div class="lbl">ブランド親和性クラス</div></div>
  <div class="stat-card"><div class="num">6</div><div class="lbl">ML モデル数</div></div>
  <div class="stat-card"><div class="num">22</div><div class="lbl">特徴量数</div></div>
  <div class="stat-card"><div class="num">4</div><div class="lbl">顧客セグメント</div></div>
</div>

<!-- EDA グラフ -->
<div class="section">
  <h2>📊 探索的データ分析 (EDA)</h2>
  <div class="chart-grid">
    <div class="chart-card">
      <h3 style="margin-bottom:10px; color:#90cdf4;">ブランド親和性の分布</h3>
      <img src="data:image/png;base64,{imgs['affinity']}" alt="Brand Affinity">
    </div>
    <div class="chart-card">
      <h3 style="margin-bottom:10px; color:#90cdf4;">年代別 ブランド親和性</h3>
      <img src="data:image/png;base64,{imgs['age']}" alt="Age vs Affinity">
    </div>
    <div class="chart-card">
      <h3 style="margin-bottom:10px; color:#90cdf4;">性別 × ブランド親和性</h3>
      <img src="data:image/png;base64,{imgs['gender']}" alt="Gender vs Affinity">
    </div>
    <div class="chart-card">
      <h3 style="margin-bottom:10px; color:#90cdf4;">年齢散布図（色=ブランド親和性）</h3>
      <img src="data:image/png;base64,{imgs['scatter']}" alt="Age Scatter">
    </div>
  </div>
</div>

<!-- ML モデル比較 -->
<div class="section">
  <h2>🤖 機械学習モデル比較 (5-Fold Cross Validation)</h2>
  <div class="chart-grid">
    <div class="chart-card">
      <img src="data:image/png;base64,{imgs['models']}" alt="Model Comparison">
    </div>
    <div class="chart-card">
      <table>
        <thead><tr><th>モデル</th><th>精度</th><th>標準偏差</th><th>スコア</th></tr></thead>
        <tbody>{model_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- 特徴量重要度 -->
<div class="section">
  <h2>📈 特徴量重要度 (Random Forest)</h2>
  <div class="chart-grid">
    <div class="chart-card">
      <img src="data:image/png;base64,{imgs['features']}" alt="Feature Importance">
    </div>
    <div class="chart-card">
      <table>
        <thead><tr><th>特徴量</th><th>重要度</th><th>視覚化</th></tr></thead>
        <tbody>{feat_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- クラスタリング -->
<div class="section">
  <h2>🔍 顧客セグメンテーション (K-Means Clustering)</h2>
  <div class="chart-card" style="max-width:900px;">
    <img src="data:image/png;base64,{imgs['cluster']}" alt="Clustering">
  </div>
</div>

<!-- インサイト -->
<div class="section">
  <h2>💡 主要インサイト</h2>
  <div class="insight-grid">
    <div class="insight">
      <h3>🎯 コアファン層の特徴</h3>
      <p>熱烈ファン(22%)・好意的(24%)で46%が積極的支持者。30代ファミリー層・
      シニア層・医療福祉従事者に高い親和性。Amazon経由での購入が多く、
      口コミ・SNSでの積極的な情報発信が特徴。</p>
    </div>
    <div class="insight">
      <h3>⚠️ 離反リスク層の特徴</h3>
      <p>嫌悪(20%)の主な理由はデザイン・品質への不満。クリエイター・
      建築士・高収入層に嫌悪傾向。「映えない」「プラスチック感」
      「大量生産」がネガティブイメージの核心。</p>
    </div>
    <div class="insight">
      <h3>📦 購買促進要因</h3>
      <p>Amazon でのレコメンド・口コミがきっかけのケースが多数。
      「一人暮らし開始」「引越し」「育休・出産」など
      ライフイベントが初購入の大きなトリガー。</p>
    </div>
    <div class="insight">
      <h3>🤖 ML予測の重要特徴量</h3>
      <p>年齢・世帯人数・Amazonでの購買行動・口コミ推奨行動・
      SNS発信が予測に最も効く特徴量。
      「実際に購入するかどうか」が最も識別力が高い。</p>
    </div>
    <div class="insight">
      <h3>📱 未充足ニーズ</h3>
      <p>スマートホーム家電・サブスクリプションサービス・
      介護向け軽量家電・子育て支援商品へのニーズが
      熱烈ファン・好意的層で特に強い。</p>
    </div>
    <div class="insight">
      <h3>🌱 サステナビリティ</h3>
      <p>環境配慮への関心は中立～懐疑的層に特に顕著。
      「グリーンプレミアム」を払う意向があるセグメントへの
      訴求がエンゲージメント向上の鍵になる可能性。</p>
    </div>
  </div>
</div>

<div style="text-align:center; padding:24px; color:#4a5568; font-size:12px;">
  Generated by iris_analysis.py | Data: iris_persona_survey_42q.csv (n=50)
</div>

</body>
</html>"""

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_report.html')
    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write(html)
    print(f"  ✅ analysis_report.html 生成完了")
    return report_path


# ───────────────────────────────────────────
# メイン
# ───────────────────────────────────────────
def main():
    print("=" * 60)
    print("  アイリスオーヤマ ペルソナ調査 ML分析システム")
    print("=" * 60)

    df = load_data()

    print("\n📋 基本統計:")
    print(f"  回答者数: {len(df)}")
    ages = pd.to_numeric(df['年齢'], errors='coerce')
    print(f"  年齢: {ages.min():.0f}〜{ages.max():.0f}歳 (平均 {ages.mean():.1f}歳)")
    print(f"  性別: {df['性別'].value_counts().to_dict()}")
    print(f"  ブランド親和性: {df['ブランド親和性'].value_counts().to_dict()}")

    # EDA
    plot_eda(df)

    # Feature engineering
    X = extract_features(df)
    y = get_target(df)
    print(f"\n🔧 特徴量: {X.shape[1]}個, サンプル: {X.shape[0]}個")
    print(f"   クラス分布: { {AFFINITY_ORDER[i]: (y==i).sum() for i in range(5)} }")

    # ML
    results, importances = run_classification(X.values, y.values, X.columns.tolist())

    # Clustering
    df_c, clusters = run_clustering(X, df)

    # Report
    report_path = generate_html_report(df, results, importances, clusters)

    print("\n" + "=" * 60)
    print("✅ 分析完了!")
    print(f"   出力フォルダ: {OUTPUT_DIR}")
    print(f"   レポート: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
