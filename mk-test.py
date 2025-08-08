# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import argparse
from matplotlib.font_manager import FontProperties

# import yaml # 取消注释并安装PyYAML库以支持配置文件

# 设置中文字体用于绘图
try:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示问题
except Exception as e:
    print(
        f"警告: 无法设置中文字体'SimHei'。图表中的中文字符可能无法正确显示。错误: {e}"
    )


def calculate_uf_ub(data):
    """
    计算Mann-Kendall突变检验的UF和UB统计序列。
    """
    data = np.array(data)
    n = len(data)
    if n < 2:
        print(f"警告: 数据长度({n})过短，无法进行UF/UB计算。")
        return None, None

    s = np.zeros(n)
    uf = np.zeros(n)
    for i in range(1, n):
        r = np.sum(data[:i] < data[i])
        s[i] = s[i - 1] + r

    for i in range(1, n):
        E_s = i * (i + 1) / 4
        Var_s = i * (i + 1) * (2 * i + 5) / 72
        if Var_s == 0:
            uf[i] = 0
        else:
            uf[i] = (s[i] - E_s) / np.sqrt(Var_s)

    data_rev = data[::-1]
    ub = np.zeros(n)
    s_rev = np.zeros(n)
    for i in range(1, n):
        r_rev = np.sum(data_rev[:i] < data_rev[i])
        s_rev[i] = s_rev[i - 1] + r_rev

    for i in range(1, n):
        E_s_rev = i * (i + 1) / 4
        Var_s_rev = i * (i + 1) * (2 * i + 5) / 72
        if Var_s_rev == 0:
            ub[i] = 0
        else:
            ub[i] = (s_rev[i] - E_s_rev) / np.sqrt(Var_s_rev)

    ub = -ub[::-1]
    return list(uf), list(ub)


def load_and_prepare_data(
    filepath,
    year_col,
    data_cols,
    group_by_col,
    missing_value_strategy="drop",
    min_data_points=8,
):
    """
    从文件(CSV或Excel)加载数据，根据参数进行分组和列选择，
    并返回要处理的数据框列表。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"错误: 文件未找到 -> {filepath}")

    try:
        if filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath, encoding="utf-8")
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(
            f"错误: 无法读取或解析文件 -> {filepath}。请确保其为CSV或Excel格式。\n具体错误: {e}"
        )

    df.columns = [str(col).strip().lower() for col in df.columns]
    year_col = year_col.strip().lower()
    if group_by_col:
        group_by_col = group_by_col.strip().lower()
    if data_cols:
        data_cols = [col.strip().lower() for col in data_cols]

    if year_col not in df.columns:
        raise ValueError(f"错误: 指定的年份列 '{year_col}' 在文件中未找到。")
    df.set_index(year_col, inplace=True)

    available_cols = df.columns.tolist()
    if group_by_col and group_by_col not in available_cols:
        raise ValueError(f"错误: 指定的分组列 '{group_by_col}' 在文件中未找到。")

    # 如果用户指定了数据列，则提前筛选，以减少内存占用
    if data_cols:
        for col in data_cols:
            if col not in available_cols:
                raise ValueError(f"错误: 指定的数据列 '{col}' 在文件中未找到。")
        # 保留用户指定的数据列和用于分组的列
        cols_to_select = data_cols + (
            [group_by_col] if group_by_col and group_by_col in df.columns else []
        )
        # 使用 set 去重，防止 group_by_col 和 data_cols 重复
        df = df[list(set(cols_to_select))]
    else:
        # 如果用户未指定数据列，则不在此处筛选，保留所有列进入下一步分组
        print("未指定数据列，将自动选择所有数值类型的列进行分析。")

    dfs_to_process = []
    if group_by_col:
        # 使用分组列对数据进行分组
        groups = df[group_by_col].unique()
        print(f"检测到分组列'{group_by_col}'。将分别分析 {len(groups)} 个组。")
        for group in groups:
            # 筛选出当前组的数据，并在此之后移除分组列
            group_df = df[df[group_by_col] == group].drop(columns=[group_by_col])
            dfs_to_process.append((str(group), group_df))
    else:
        # 不分组，整个DataFrame作为一个整体处理
        print("未提供分组列。将把整个文件作为一个数据集进行分析。")
        dfs_to_process.append(("default", df))

    # 在此循环中对每个（已分组或未分组的）数据框进行最终的数值清洗
    final_dfs = []
    for name, data_df in dfs_to_process:
        proc_df = data_df.copy()

        for col in proc_df.columns:
            proc_df[col] = pd.to_numeric(proc_df[col], errors="coerce")

        proc_df.dropna(axis=1, how="all", inplace=True)

        if proc_df.empty:
            print(f"警告: 组 '{name}' 在转换为数值类型后没有可分析的数据。跳过该组。")
            continue

        if missing_value_strategy == "mean":
            proc_df.fillna(proc_df.mean(), inplace=True)
        elif missing_value_strategy == "median":
            proc_df.fillna(proc_df.median(), inplace=True)
        elif missing_value_strategy == "interpolate":
            proc_df.sort_index(inplace=True)
            proc_df.interpolate(method="linear", limit_direction="both", inplace=True)

        final_dfs.append((name, proc_df))

    return final_dfs


def plot_uf_ub(
    uf,
    ub,
    index,
    alpha,
    group_name,
    category,
    output_dir,
    plot_title=None,
    xlabel="年份",
    ylabel="统计量",
):
    """
    绘制并保存UF和UB曲线图。
    """
    if uf is None or ub is None:
        print(f"注意: 由于数据不足，无法为 {group_name} - {category} 生成图表。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(index, uf, "r-", label="UF")
    plt.plot(index, ub, "b-", label="UB")
    plt.legend(loc="upper left")

    z_alpha = st.norm.ppf(1 - alpha / 2)
    plt.axhline(0, color="k", linestyle="--")
    plt.axhline(z_alpha, color="k", linestyle="--", label=f"显著性水平 (α={alpha})")
    plt.axhline(-z_alpha, color="k", linestyle="--")

    plt.grid(True)

    if plot_title:
        title = plot_title
    else:
        title = f"Mann-Kendall突变检验 ({category})"
        if group_name != "default":
            title = f"Mann-Kendall突变检验 - 组 {group_name} ({category})"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    os.makedirs(output_dir, exist_ok=True)

    filename = (
        f"MK_Test_{group_name}_{category}.png"
        if group_name != "default"
        else f"MK_Test_{category}.png"
    )
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"图表已保存至: {filepath}")


def detect_mutations_and_export(results, output_file):
    """
    从分析结果中检测突变点并导出到汇总文件。

    突变点定义: UF和UB曲线的交点，且该交点位于置信区间(±z_alpha)内部。
    这通常标志着一个潜在变化时段的开始。
    """
    mutation_points = []
    for result in results:
        group_name = result["group_name"]
        category = result["category"]
        uf = result["uf"]
        ub = result["ub"]
        index = result["index"]
        alpha = result["alpha"]

        if uf is None or ub is None:
            continue

        diff = np.array(uf) - np.array(ub)
        intersections = np.where(np.diff(np.sign(diff)))[0]
        z_alpha = st.norm.ppf(1 - alpha / 2)

        is_mutation_found = False
        for i in intersections:
            # 检查交点是否在显著性区间内
            if abs(uf[i]) < z_alpha and abs(ub[i]) < z_alpha:
                mutation_year = index[i]
                mutation_points.append(
                    {
                        "分组": group_name if group_name != "default" else "N/A",
                        "数据类别": category,
                        "突变年份": mutation_year,
                        "显著性水平": alpha,
                        "UF值": f"{uf[i]:.3f}",
                        "UB值": f"{ub[i]:.3f}",
                        "备注": "在显著性水平内检测到突变",
                    }
                )
                is_mutation_found = True

        if not is_mutation_found:
            mutation_points.append(
                {
                    "分组": group_name if group_name != "default" else "N/A",
                    "数据类别": category,
                    "突变年份": "N/A",
                    "显著性水平": alpha,
                    "UF值": "N/A",
                    "UB值": "N/A",
                    "备注": "未在显著性水平内检测到突变点",
                }
            )

    if not mutation_points:
        print("在任何数据类别中均未检测到突变点。")
        return

    summary_df = pd.DataFrame(mutation_points)
    summary_df.to_csv(output_file, index=False, encoding="utf_8_sig")


def main():
    """
    协调脚本执行流程的主函数。
    """
    parser = argparse.ArgumentParser(
        description="执行Mann-Kendall突变检验并绘制UF/UB曲线。",
        epilog=(
            "示例 1 (不分组): python mk_test.py data.csv --year-col '年份' --data-cols '降雨量' '径流量'\n"
            "示例 2 (按站点分组): python mk_test.py data.csv --year-col 'year' --group-by-col 'site' -o ./results\n"
            "示例 3 (完整参数): python mk_test.py your_data.xlsx --year-col '年份' --group-by-col '站号' --data-cols '蒸发量' -a 0.01 --missing-value-strategy 'interpolate'"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="输入CSV或Excel文件路径。")
    parser.add_argument(
        "--year-col",
        type=str,
        required=True,
        help="[必需] 指定包含年份的列名。",
    )
    parser.add_argument(
        "--data-cols",
        type=str,
        nargs="+",
        help="要分析的数据列名列表。如果省略，则分析除年份和分组列外的所有数值列。",
    )
    parser.add_argument(
        "--group-by-col",
        type=str,
        help="用于分组的列名（如 '站点' 或 '站号'）。如果省略，则不进行分组。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="图表和报告的输出目录 (默认: 当前目录)。",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help="检验的显著性水平 (默认: 0.05)。",
    )
    parser.add_argument(
        "--missing-value-strategy",
        type=str,
        default="drop",
        choices=["drop", "mean", "median", "interpolate"],
        help="缺失值处理策略 (默认: 'drop')。",
    )
    parser.add_argument(
        "--min-data-points",
        type=int,
        default=8,
        help="分析所需的最小数据点数 (默认: 8)。",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        help="自定义图表标题。如未提供，将根据数据自动生成。",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="年份",
        help="X轴的自定义标签 (默认: '年份')。",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="统计量",
        help="Y轴的自定义标签 (默认: '统计量')。",
    )

    args = parser.parse_args()

    try:
        dfs_to_process = load_and_prepare_data(
            args.input_file,
            args.year_col,
            args.data_cols,
            args.group_by_col,
            args.missing_value_strategy,
            args.min_data_points,
        )

        all_results = []
        for group_name, df in dfs_to_process:
            print("\n" + "=" * 60)
            print(f"--- 开始处理组: '{group_name}' ---")
            print("=" * 60)

            if df.empty:
                print("该组没有可分析的数据，跳过。")
                continue

            for category in df.columns:
                print(f"\n> 正在分析类别: '{category}'")

                if args.missing_value_strategy == "drop":
                    series = df[category].dropna()
                else:
                    series = df[category]

                if len(series) < args.min_data_points:
                    print(
                        f"  注意: 数据序列 '{category}' (组: '{group_name}') 在处理后数据点不足(需要{args.min_data_points}个，实际{len(series)}个)。跳过。"
                    )
                    continue

                uf, ub = calculate_uf_ub(series)
                if uf is None or ub is None:
                    print(
                        f"  注意: 无法计算UF/UB序列 (组: '{group_name}', 类别: '{category}')。"
                    )
                    continue

                all_results.append(
                    {
                        "group_name": group_name,
                        "category": category,
                        "uf": uf,
                        "ub": ub,
                        "index": series.index.tolist(),
                        "alpha": args.alpha,
                    }
                )

                plot_uf_ub(
                    uf,
                    ub,
                    series.index,
                    args.alpha,
                    group_name,
                    category,
                    args.output_dir,
                    plot_title=args.plot_title,
                    xlabel=args.xlabel,
                    ylabel=args.ylabel,
                )

        if all_results:
            summary_file_path = os.path.join(args.output_dir, "MKtest_summary.csv")
            detect_mutations_and_export(all_results, summary_file_path)

            print("\n" + "=" * 60)
            print("--- 分析总结 ---")
            print(f"所有图表已保存至目录: {os.path.abspath(args.output_dir)}")
            print(f"汇总报告文件: {os.path.abspath(summary_file_path)}")
            print("=" * 60)
        else:
            print("\n未处理任何有效数据，未生成任何输出文件。")

        print("\n所有任务已完成。")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")


if __name__ == "__main__":
    main()
