import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind

mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .util import files, math
from .util.stopwatch import Stopwatch
from .preprocessing.responses_to_nli import get_entailment_label
from .semantic_similarity import SemanticSimilarity
from .coherence import BertCoherence, ESIMCoherence

logger = logging.getLogger(__name__)

entailment_labels = ("entailment", "neutral", "contradiction")


def compute_metric(metric, generator_types, response_file):
    _result = metric.compute_metric_for_file(response_file, generator_types)
    df = pd.DataFrame.from_dict({i: s for i, s in enumerate(_result)}, "index")

    groundtruth_df = df[df["gen_type"] == "ground_truth"]
    groundtruth_df = groundtruth_df.reset_index(drop=True)

    gen_df = df[df["gen_type"] != "ground_truth"]
    gen_df = gen_df.reset_index(drop=True)

    return df, groundtruth_df, gen_df


def _save_or_show(plots_dir, plot_name, show=True):
    if plots_dir:
        plt.savefig(plots_dir / f"{plot_name}.pdf")
    elif show:
        plt.show()


def plot(response_file, human_judgment_file, generator_types,
         bert_model_dir=None, esim_model=None, embedding_method='elmo',
         plots_dir=None):
    is_judgment_included = human_judgment_file is not None

    if is_judgment_included:
        human_judgment = files.load_obj(human_judgment_file)
        logger.info(f"human judgment file loaded ({len(human_judgment)} items found)")
        human_scores = np.asarray([score for _, _, score in human_judgment])
        noisy_human_scores = human_scores + np.random.normal(0.0, 0.1, len(human_scores))
        human_infer_labels = [get_entailment_label(score) for _, _, score in human_judgment]
    else:
        human_scores, noisy_human_scores, human_infer_labels = None, None, None
        logger.info(f"no human judgment provided")

    plot_prefix = files.get_file_name(response_file)

    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(exist_ok=True)

    sns.set_context("paper")

    sw = Stopwatch()
    ss = SemanticSimilarity(embedding_method)
    logger.info("computing semantic similarity...")
    ss_df, _, ss_gen_df = compute_metric(ss, generator_types, response_file)
    if is_judgment_included:
        ss_gen_df = ss_gen_df.assign(human_score=pd.Series(human_scores)) \
            .assign(noisy_human_score=pd.Series(noisy_human_scores)) \
            .assign(human_infer_label=pd.Series(human_infer_labels))

    plot_ss(ss_df, ss_gen_df, is_judgment_included, plot_prefix, plots_dir)
    logger.info(f"semantic similarity plots generated ({sw.elapsed()}s)")

    if bert_model_dir:
        sw = Stopwatch()
        logger.info("processing coherence fine-tuned via BERT...")
        bertc = BertCoherence(bert_model_dir)
        _, bert_groundtruth_df, bert_gen_df = compute_metric(bertc, generator_types, response_file)

        if is_judgment_included:
            bert_gen_df = bert_gen_df.assign(human_score=pd.Series(human_scores)) \
                .assign(noisy_human_score=pd.Series(noisy_human_scores)) \
                .assign(human_infer_label=pd.Series(human_infer_labels))

        plot_coherence(bert_gen_df, bert_groundtruth_df, is_judgment_included, plot_prefix, plots_dir)
        logger.info(f"BERT coherence plots generated ({sw.elapsed()}s)")

    if esim_model:
        sw = Stopwatch()
        logger.info("processing coherence trained using ESIM...")
        esimc = ESIMCoherence(esim_model)
        _, esim_groundtruth_df, esim_gen_df = compute_metric(esimc, generator_types, response_file)

        if is_judgment_included:
            esim_gen_df = esim_gen_df.assign(human_score=pd.Series(human_scores)) \
                .assign(noisy_human_score=pd.Series(noisy_human_scores)) \
                .assign(human_infer_label=pd.Series(human_infer_labels))

        plot_coherence(esim_gen_df, esim_groundtruth_df, is_judgment_included, plot_prefix, plots_dir)
        logger.info(f"ESIM coherence plots generated ({sw.elapsed()}s)")

    logger.info("visualization done")


def plot_coherence(gen_df, groundtruth_df, is_human_judgment_included, plot_prefix, plots_dir):
    coherence_metrics = ["context_label", "Utt_-1_label", "Utt_-2_label"]
    axis_labels = {
        "context_label": "Predicted Class",
        "Utt_-1_label": "Predicted Class (Utt_-1)",
        "Utt_-2_label": "Predicted Class (Utt_-2)"
    }

    for x in coherence_metrics:
        plt.figure()
        sns.countplot(x=x, data=gen_df)
        _save_or_show(plots_dir, f"{plot_prefix}_{x}_bar")

        plt.figure()
        sns.countplot(x=x, data=groundtruth_df)
        _save_or_show(plots_dir, f"{plot_prefix}_{x}_gt_bar", show=False)

        if is_human_judgment_included:
            plt.figure()
            sns.countplot(x=x, hue="human_infer_label", data=gen_df)
            _save_or_show(plots_dir, f"{plot_prefix}_{x}_human_infer_bar", show=False)

            plt.figure()
            fig = sns.boxplot(x=x, y="human_score", data=gen_df,
                              fliersize=3, linewidth=1.5,
                              order=["entailment", "neutral", "contradiction"])
            fig.set(ylabel='Human Score', xlabel=axis_labels[x])
            _save_or_show(plots_dir, f"{plot_prefix}_{x}_human_score_box", show=False)

            plt.figure()
            sns.lineplot(x=x, y="human_score", data=gen_df, err_style="bars")
            _save_or_show(plots_dir, f"{plot_prefix}_{x}_human_score_line", show=False)

            correct_df = gen_df[gen_df[x] == gen_df["human_infer_label"]]
            accuracy = 100.0 * len(correct_df) / len(gen_df)
            print("*****")
            human_score_entail = gen_df[gen_df[x] == "entailment"]["human_score"]
            human_score_contra = gen_df[gen_df[x] == "contradiction"]["human_score"]
            human_score_neutral = gen_df[gen_df[x] == "neutral"]["human_score"]
            print(f"t-test Human Score entailment vs contradiction {ttest_ind(human_score_entail, human_score_contra)}")
            print(f"t-test Human Score entailment vs neutral {ttest_ind(human_score_entail, human_score_neutral)}")
            print(f"t-test Human Score neutral vs contradiction {ttest_ind(human_score_neutral, human_score_contra)}")
            print("---")
            print(f"Inference {x} Accuracy: {accuracy:.2f}% ({len(correct_df)}/{len(gen_df)})")

            for label in entailment_labels:
                n_corrects_per_label = len(correct_df[correct_df[x] == label])
                n_instances_per_label = len(gen_df[gen_df[x] == label])
                label_accuracy = 100.0 * math.safe_div(n_corrects_per_label, n_instances_per_label)
                print(f"  {x} {label}: {label_accuracy:.2f} ({n_corrects_per_label}/{n_instances_per_label})")

            n_gt_corrects = len(groundtruth_df[groundtruth_df[x] == "entailment"])
            gt_accuracy = 100.0 * n_gt_corrects / len(gen_df)
            print(f"GroundTruth {x} Accuracy: {gt_accuracy:.2f}% ({n_gt_corrects}/{len(gen_df)})")

            for label in entailment_labels[1:]:
                n_incorrects_per_label = len(groundtruth_df[groundtruth_df[x] == label])
                print(f"  {x} {label}: {n_incorrects_per_label}")

    plt.close('all')


def plot_ss(df, gen_df, is_human_judgment_included, plot_prefix, plots_dir):
    ss_metrics = ["SS_context", "SS_Utt_-1", "SS_Utt_-2"]
    for y in ss_metrics:
        plt.figure()
        sns.boxplot(x="gen_type", y=y, data=df, linewidth=1.5)
        _save_or_show(plots_dir, f"{plot_prefix}_{y}_gt_included_box")

        plt.figure()
        sns.boxplot(x="gen_type", y=y, data=gen_df, linewidth=1.5)
        _save_or_show(plots_dir, f"{plot_prefix}_{y}_box", show=False)

        plt.figure()
        sns.lineplot(x="gen_type", y=y, data=df, err_style="bars")
        _save_or_show(plots_dir, f"{plot_prefix}_{y}_gt_included_line", show=False)

        plt.figure()
        sns.lineplot(x="gen_type", y=y, data=gen_df, err_style="bars")
        _save_or_show(plots_dir, f"{plot_prefix}_{y}_line", show=False)

        if is_human_judgment_included:
            plt.figure()
            ax = sns.regplot(x='noisy_human_score', y=y, data=gen_df,
                             line_kws={"color": "red"}, scatter_kws={"color": "black"})
            ax.set(xlabel='Human Score', ylabel=y)
            ax.yaxis.label.set_size(12)
            ax.xaxis.label.set_size(12)
            _save_or_show(plots_dir, f"{plot_prefix}_{y}_correlation")

            plt.figure()
            sns.boxplot(x="human_infer_label", y=y, data=gen_df, linewidth=1.5)
            _save_or_show(plots_dir, f"{plot_prefix}_{y}_entail_box", show=False)

            plt.figure()
            sns.lineplot(x="human_infer_label", y=y, data=gen_df, linewidth=1.5)
            _save_or_show(plots_dir, f"{plot_prefix}_{y}_entail_line", show=False)

            print("*****")
            print(f"Human MER vs. {y} Pearson: {pearsonr(gen_df['human_score'], gen_df[y])}")
            print(f"Human MER vs. {y} Spearman: {spearmanr(gen_df['human_score'], gen_df[y])}")

    plt.close('all')
