# -*- coding:utf-8 -*-
# Author : liyanchi
# Time : 2024/1/12 17:08

import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw
from pyecharts.charts import Line
import pyecharts.options as opts
import matplotlib.pyplot as plt

# 数据准备
# dataframe：每行代表一个指标的时间序列数据，每列代表同一时间不同指标的数据
# array：每个最内层的list代表一个指标的时间序列数据
data = np.array([
    [
        0.70820561, 0.66635751, 0.64761909, 0.51277711, 0.69104645,
        0.68914818, 0.66988182, 0.64279839, 0.68313003, 0.66656726,
        0.60275364, 0.71987334, 0.74265594, 0.68890755, 0.53591975,
        -1.98055209, -2.61140459, -1.92792463, -0.31124071, -1.70284578,
        -1.68438506, -2.50063641, -0.12336395, -0.57930140, 0.42230268,
        -0.57170881, -0.79201969, -0.53055864, 0.14321483, 0.00610886,
        0.32659209, 0.44864948, 0.63649178, 0.74415626, 0.69421914,
        0.63400929, 0.70890435, 0.68365134
    ],
    [
        0.67156157, 0.66849770, 0.60351746, 0.70243330, 1.08889305,
        0.95012828, 0.78979176, 0.88700149, 0.86083673, 0.96814147,
        0.90535759, 0.57642294, 0.41448000, 0.26716666, 0.23211724,
        -1.08913216, -1.88470276, -1.65526477, -0.65737941, -2.68663081,
        -2.02689104, -2.11660135, -0.20691334, 0.16939921, -0.02875750,
        -0.73573072, -0.90070130, -0.74430782, 0.03890552, -0.35960314,
        -0.25418710, 0.55645787, 0.47648076, 0.99788169, 1.03787076,
        0.27740429, 0.20721091, 0.99884499
    ],
    [
        0.62578280, 0.62324641, 0.60667527, 0.47714988, 0.71895316,
        0.64894857, 0.67600349, 0.75141906, 0.59517693, 0.64269212,
        0.43436930, 0.52500325, 0.69308190, 0.66399787, 0.28235450,
        -0.60589210, -1.13447746, -0.95202589, -0.75367956, -2.04081704,
        -2.00175651, -2.12654730, -2.14920579, -2.17034244, -1.53133922,
        1.11581491, 0.71590948, -0.02370420, -0.14950954, 0.26459972,
        0.17582578, 0.49203008, 0.43842753, 0.47207708, 0.56744565,
        0.70660936, 0.65554321, 1.07015975
    ]
])

# 对时间序列进行标准化
scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
scaled_data = scaler.fit_transform(data)

# 时间序列 KMeans 聚类
# 使用 DTW 算法来度量时间序列之间的相似性，即使它们在时间、速度或长度上不完全对齐
silhouette_scores = []
clusters = [str(i) for i in range(2, 31)]
for n_clusters in clusters:
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_init=5, verbose=True, random_state=0)
    predicted_labels = model.fit_predict(scaled_data)

    # 计算轮廓系数
    _silhouette_score = silhouette_score(cdist_dtw(scaled_data), predicted_labels, metric="precomputed")
    silhouette_scores.append(_silhouette_score)

# 轮廓系数评估最优聚类结果
line = (
    Line()
    .add_xaxis(xaxis_data=clusters)
    .add_yaxis(
        series_name="轮廓系数",
        y_axis=silhouette_scores,
        color="#2878b5",
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(
            data=[
                opts.MarkPointItem(
                    name="第一优聚类数",
                    # 1. 最大值进行标记
                    # type_='max',
                    # 2. 指定值进行标记
                    coord=["20", silhouette_scores[18]],
                    value=round(silhouette_scores[18], 2),
                    itemstyle_opts=opts.ItemStyleOpts(color="#c82423")
                )
            ]
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="使用 KMeans 聚类方法下不同聚类簇数的轮廓系数关系图", pos_left='center'),
        yaxis_opts=opts.AxisOpts(min_=0.25, splitline_opts=opts.SplitLineOpts(is_show=False)),
        legend_opts=opts.LegendOpts(pos_left='right', pos_top='top'),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
    .set_series_opts(
        #         markline_opts=opts.MarkLineOpts(
        #             data=[opts.MarkLineItem(name="水平线", y=best_silhouette_score)],
        #             label_opts=opts.LabelOpts(),
        #             linestyle_opts=opts.LineStyleOpts(width = 3,color = '#FFFF00')
        #         ),
        markarea_opts=opts.MarkAreaOpts(
            data=[
                opts.MarkAreaItem(name="最优聚类簇数 n=20", x=("19", "21")),
                opts.MarkAreaItem(name="第二优聚类簇数 n=7", x=("6", "8"))
            ],
            itemstyle_opts=opts.ItemStyleOpts(color="#f8ac8c", opacity=0.2),
        )
    )
)

line.render_notebook()

# 最优聚类结果可视化
best_n_cluster = 7
model = TimeSeriesKMeans(n_clusters=best_n_cluster, metric="dtw", n_init=5, verbose=True, random_state=0)
predicted_labels = model.fit_predict(scaled_data)
plt.figure(figsize=(20, 15))
for i in range(best_n_cluster):
    plt.subplot(best_n_cluster, 1, i + 1)
    for j in range(len(predicted_labels)):
        if predicted_labels[j] == i:
            plt.plot([i for i in range(data.shape)],
                pivot.iloc[j].values,
                label=f'Time Series {j}'
            )
    plt.title(f'Cluster {i + 1}')
plt.suptitle('聚类结果', fontproperties="SimHei", fontsize='xx-large')

plt.tight_layout()  # 调整子图之间的间距
plt.show()

# 具体聚类结果
cluster_result = {}
for cluster_num in range(best_n_cluster):
    cluster_result[cluster_num] = [i for i, x in enumerate(predicted_labels) if x == cluster_num]
