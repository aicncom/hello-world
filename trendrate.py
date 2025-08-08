import pandas as pd
from scipy.stats import linregress
import argparse
import sys
from pathlib import Path


def calculate_trend_rate(df: pd.DataFrame, year_col: str, factor_col: str):
    """
    对给定的DataFrame中的单个要素列进行线性回归，计算倾向率。

    Args:
        df (pd.DataFrame): 包含数据的DataFrame。
        year_col (str): 年份列的名称。
        factor_col (str): 要分析的要素列的名称。

    Returns:
        dict or None: 包含回归结果的字典，如果数据点不足则返回None。
    """
    # 移除包含NaN的行，以确保回归计算的准确性
    clean_df = df.dropna(subset=[year_col, factor_col])

    # 确保数据是数值类型
    x = pd.to_numeric(clean_df[year_col], errors="coerce")
    y = pd.to_numeric(clean_df[factor_col], errors="coerce")

    # 移除转换后可能产生的NaN值
    valid_indices = ~(x.isna() | y.isna())
    x = x[valid_indices]
    y = y[valid_indices]

    # 再次检查数据点数量
    if len(x) < 2:
        print(f"警告: 要素 '{factor_col}' 的有效数据点少于2个，无法计算趋势。")
        return None

    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "要素 (Factor)": factor_col,
        "倾向率 (Slope)": slope,
        "截距 (Intercept)": intercept,
        "R平方 (R-squared)": r_value**2,
        "P值 (P-value)": p_value,
        "标准误差 (Std_err)": std_err,
        "数据点数 (N)": len(x),
    }


def _read_data_file(file_path: str) -> pd.DataFrame:
    """
    根据文件扩展名读取数据，处理可能的错误。

    Args:
        file_path (str): 数据文件路径。

    Returns:
        pd.DataFrame: 读取到的数据。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果文件格式不被支持。
        ImportError: 如果缺少读取Excel文件所需的库。
    """
    file_p = Path(file_path)
    if not file_p.is_file():
        raise FileNotFoundError(f"文件未找到 '{file_path}'")

    file_extension = file_p.suffix.lower()
    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(file_path)
        except ImportError:
            # 提醒用户安装 openpyxl
            raise ImportError(
                "读取Excel文件需要 `openpyxl` 库。请运行 `pip install openpyxl` 安装。"
            )
    else:
        raise ValueError(
            f"不支持的文件格式: '{file_extension}'。请提供 .csv 或 .xlsx 文件。"
        )


def process_data(
    file_path: str, year_col: str, factor_cols: list, group_by_cols: list = None
):
    """
    加载数据并对指定的要素列计算倾向率。

    Args:
        file_path (str): 数据文件路径 (csv 或 xlsx)。
        year_col (str): 年份列的名称。
        factor_cols (list): 霁要计算倾向率的要素列名称列表。
        group_by_cols (list, optional): 用于分组的列名列表 (如 ['站点'])。默认为 None。

    Returns:
        pd.DataFrame: 包含所有计算结果的DataFrame。

    Raises:
        KeyError: 如果数据文件中缺少必要的列。
    """
    df = _read_data_file(file_path)

    # 检查所有必需的列是否存在
    required_cols = [year_col] + factor_cols + (group_by_cols if group_by_cols else [])
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"数据文件中缺少以下列: {', '.join(missing_cols)}")

    all_results = []

    if group_by_cols:
        # 按指定列分组计算
        print(f"按 {', '.join(group_by_cols)} 分组进行计算...")
        grouped = df.groupby(group_by_cols)
        for group_name, group_df in grouped:
            # 确保group_name是元组时能正确处理
            group_identifier = dict(
                zip(
                    group_by_cols,
                    [group_name] if not isinstance(group_name, tuple) else group_name,
                )
            )

            for factor in factor_cols:
                result = calculate_trend_rate(group_df, year_col, factor)
                if result:
                    # 将分组信息添加到结果中
                    result.update(group_identifier)
                    all_results.append(result)
    else:
        # 对整个数据集计算
        print("对整个数据集进行计算...")
        for factor in factor_cols:
            # 检查factor列是否为数值型，如果不是则跳过
            if not pd.api.types.is_numeric_dtype(df[factor]):
                print(f"警告: 要素列 '{factor}' 不是数值类型，将跳过计算。")
                continue

            result = calculate_trend_rate(df, year_col, factor)
            if result:
                all_results.append(result)

    if not all_results:
        print("未能计算出任何结果。请检查数据。")
        return pd.DataFrame()

    # 将结果转换为DataFrame并调整列顺序
    results_df = pd.DataFrame(all_results)

    # 将分组列和要素列移动到前面
    id_cols = (group_by_cols if group_by_cols else []) + ["要素 (Factor)"]
    data_cols = [col for col in results_df.columns if col not in id_cols]
    results_df = results_df[id_cols + data_cols]

    return results_df


def main():
    """主函数，解析命令行参数并执行计算。"""
    parser = argparse.ArgumentParser(
        description="计算CSV或Excel文件中数据的倾向率（线性趋势）。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
示例:
1. 对单个文件计算，不分组:
   python calculate_trend.py data.csv --year-col 年份 --factor-cols 温度 降水量

2. 按 '站点' 列分组计算:
   python calculate_trend.py data.xlsx --year-col Year --factor-cols Temp Precipitation --group-by-cols Station

3. 将结果保存到指定文件:
   python calculate_trend.py my_data.xlsx --year-col Year --factor-cols Temp --group-by-cols Station --output-path results.xlsx
""",
    )

    parser.add_argument("file_path", help="输入的数据文件路径 (.csv or .xlsx)")
    parser.add_argument("--year-col", required=True, help="包含年份的列名")
    parser.add_argument(
        "--factor-cols",
        required=True,
        nargs="+",
        help="一个或多个需要分析的要素列名，多个用空格分开",
    )
    parser.add_argument(
        "--group-by-cols",
        nargs="*",
        help="（可选）一个或多个用于分组的列名，例如 '站点'",
    )
    parser.add_argument(
        "--output-path", help="（可选）保存结果的文件路径 (.csv or .xlsx)"
    )

    args = parser.parse_args()

    try:
        results = process_data(
            args.file_path, args.year_col, args.factor_cols, args.group_by_cols
        )
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误: {e}", file=sys.stderr)
        sys.exit(1)

    if not results.empty:
        print("\n--- 计算结果 ---")
        # 使用 to_string() 以便在终端中完整显示所有行和列
        print(results.to_string())

        if args.output_path:
            try:
                output_ext = Path(args.output_path).suffix.lower()
                if output_ext == ".csv":
                    results.to_csv(args.output_path, index=False, encoding="utf-8-sig")
                elif output_ext in [".xlsx", ".xls"]:
                    results.to_excel(args.output_path, index=False)
                else:
                    print(f"警告: 不支持的输出文件格式 '{output_ext}'。将保存为CSV。")
                    results.to_csv(
                        args.output_path + ".csv", index=False, encoding="utf-8-sig"
                    )

                print(f"\n结果已成功保存到: {args.output_path}")
            except Exception as e:
                print(f"\n保存文件时出错: {e}")


if __name__ == "__main__":
    main()
