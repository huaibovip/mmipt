import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import wilcoxon, ttest_rel, ttest_ind


def measure(exp_names, outstructs, num_sample, num_substruct, save_clean_csv=False):
    data_list = []
    metric_list = []
    for exp_name, csv_path in exp_names.items():
        print(exp_name)
        exp_data = np.zeros((len(outstructs), num_sample))
        for stct_idx, stct in enumerate(outstructs.keys()):
            tar_idx = []
            with open(csv_path, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, line in enumerate(reader):
                    if i == 0:
                        names = line[0].split(",")[1:num_substruct + 1]
                        for idx, item in enumerate(names):
                            if stct in item:
                                tar_idx.append(idx)
                    else:
                        val = 0
                        for lr_i in tar_idx:
                            vals = line[0].split(",")[1:num_substruct + 1]
                            val += float(vals[lr_i])
                        val = val / len(tar_idx)
                        exp_data[stct_idx, i - 1] = val
        data_list.append(exp_data)
        metric_per_patient = exp_data.mean(axis=0)
        metric_list.append(metric_per_patient)
        # mean & std by patient
        print(f"  DSC: {metric_per_patient.mean():.3f} ± {metric_per_patient.std():.3f}")
        # jec_det and time
        jdet_list = []
        time_list = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                time_list.append(row[-1])
                jdet_list.append(row[-2])
        time_list = time_list[1:]
        jdet_list = jdet_list[1:]
        time_list = np.array([float(i) for i in time_list])
        jdet_list = np.array([float(i) for i in jdet_list]) * 100
        print("  avg_time: {:.3f} ± {:.3f}".format(time_list.mean(), time_list.std()))
        print("  jec_det(%): {:.3f} ± {:.3f}\n".format(jdet_list.mean(), jdet_list.std()))
        # save clean csv
        if save_clean_csv:
            csv_name = os.path.basename(csv_path)
            csv_root = os.path.dirname(csv_path)
            save_root = os.path.join(csv_root, "clean")
            if not os.path.exists(save_root):
                os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, "clean_" + csv_name), "w") as csv_file:
                csv_file.write(",".join(outstructs.keys()))
                csv_file.write(",\n")
                for p in exp_data.T:
                    line = ",".join(map(str, p))
                    csv_file.write(line)
                    csv_file.write(",\n")

    return data_list, metric_list


def ttest(metric):
    """ T-test """
    vec1 = metric[-1]
    for idx, exp_name in enumerate(list(exp_names.keys())[:-1]):
        vec2 = metric[idx]
        rank, pval = ttest_rel(list(vec1), list(vec2))
        print("{}, p-vale: {:.20f}".format(exp_name, pval))


def set_box_property(bp, color, line_color, linewidth):
    for patch in bp["boxes"]:
        patch.set(color=line_color, linewidth=linewidth)
        patch.set(facecolor=color)
    # plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=line_color)
    plt.setp(bp["caps"], color=line_color)
    plt.setp(bp["medians"], color=line_color)


def plot(datas, save_path, ylabel, show=False):
    sep = 1.0
    showmeans = False
    spacing_factor = 14
    boxplot_widths = 1.0
    linewidth = 0.8
    line_color="#5f5f5f"
    # font_name = "Cambria"
    font_name = "Times New Roman"

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    flierprops = dict(
        marker="o",
        markerfacecolor="grey",
        markersize=2,
        linestyle="none",
        markeredgecolor="grey",
    )
    meanprops = {
        "markerfacecolor": "sandybrown",
        "markeredgecolor": "chocolate"
    }
    medianprops = {
        "linewidth": linewidth,
    }
    colors = [
        "#AECBDE", "#3372A1", "#B4D49A",
        "#41903C", "#F4A5A1", "#C7B5D3",
        "#E8BF81", "#E07D22", "#A05F36",
        "#6A498F", "#F0F2A0", "#CF3235",
    ]

    # sep_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    sep_factors = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    bps = dict()
    for i, name in enumerate(exp_names.keys()):
        bps[name] = plt.boxplot(
            datas[i].T,
            labels=outstructs.values(),
            positions=np.array(range(len(outstructs))) * spacing_factor + sep * sep_factors[i],
            widths=boxplot_widths,
            showmeans=showmeans,
            flierprops=flierprops,
            meanprops=meanprops,
            medianprops=medianprops,
            patch_artist=True,
        )
        set_box_property(bps[name], colors[i], line_color, linewidth)

    # legend
    legend_font = font_manager.FontProperties(family=font_name, style="normal", size=12)
    leg = ax.legend(
        [bp["boxes"][0] for bp in bps.values()],
        [name for name in bps.keys()],
        ncol=4,
        prop=legend_font,
        loc="lower left",
        edgecolor=line_color,
    )
    for line in leg.get_lines():
        line.set_linewidth(8.0)

    minor_ticks = np.arange(-11, len(outstructs) * spacing_factor, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(np.arange(0, 1.05, 0.2))
    ax.set_yticks(np.arange(-0.05, 1.05, 0.05), minor=True)
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")
    plt.xticks(
        range(0, len(outstructs) * spacing_factor, spacing_factor),
        outstructs.values(),
        fontsize=14,
        rotation=20,
    )
    plt.yticks(fontsize=15)
    plt.ylabel(
        ylabel,
        fontdict={
            "family": font_name,
            "weight": "normal",
            "size"  : 20,
        }
    )
    for tick in ax.get_xticklabels():
        tick.set_fontname(font_name)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font_name)
    plt.xlim(-8, len(outstructs) * spacing_factor - 6.2)
    plt.ylim(-0.05, 0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


if __name__ == '__main__':
    exp_names = {
        "Affine"        : "Affine",
        "SyN"           : "ANTs_SyN",
        "NiftyReg"      : "NiftyReg",
        "LDDMM"         : "LDDMM",
        "deedsBCV"      : "deedsBCV",
        "VoxelMorph"    : "Vxm_1_ncc_1_diffusion_1",
        "ViTVNet"       : "ViTVNet_ncc_1_diffusion_1",
        "XMorpher"      : "XMorpher_ncc_1_diffusion_1",
        "TransMorph"    : "TransMorph_ncc_1_diffusion_1",
        "TransMatch"    : "TransMatch_ncc_1_diffusion_1",
        "Ours"          : "DilateMorph_ncc_1_diffusion_2",
        "Ours (optim)"  : "DilateMorphBi_ncc_1_diffusion_2_optim_d1.5_e20_lr0.2",
    }

    outstructs = {
        "Brain-Stem"                : "Brain-Stem",
        "Thalamus"                  : "Thalamus",
        "Cerebellum-Cortex"         : "CC",
        "Cerebral-White-Matter"     : "CWM",
        "Cerebellum-White-Matter"   : "CeWM",
        "Putamen"                   : "Putamen",
        "VentralDC"                 : "VentralDC",
        "Pallidum"                  : "Pallidum",
        "Caudate"                   : "Caudate",
        "Lateral-Ventricle"         : "LV",
        "Hippocampus"               : "Hippocampus",
        "3rd-Ventricle"             : "3rd-Ventricle",
        "4th-Ventricle"             : "4th-Ventricle",
        "Amygdala"                  : "Amygdala",
        "Cerebral-Cortex"           : "CeCo",
        "CSF"                       : "CSF",
        "choroid-plexus"            : "CP",
    }

    metric = 'dice'
    num_sample = 115
    num_substruct = 30
    save_clean_csv = False
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, 'Results/IXI')
    assert metric in ['dice', 'asd', 'hd95']

    for name, csv_name in exp_names.items():
        exp_names[name] = os.path.join(root, metric + "_" + csv_name + ".csv")
        print("{0:>12}: {1}".format(name, exp_names[name]))

    datas, metrics = measure(exp_names, outstructs, num_sample, num_substruct, save_clean_csv)
    plot(datas, 'results_{}_boxplot.png'.format(metric), ylabel=metric.capitalize())
