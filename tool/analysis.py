import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['toolbar'] = 'None'# 关闭工具栏

file_path = r'D:\learning\autolearn\Life Expectancy Data.csv'#调试过程中文件的路径

def get_country():
    # 固定中日韩英美法德 7 个国家的列表,若不存在数据项则剔除
    country_list = ['China', 'Japan', 'Republic of Korea', 'United Kingdom of Great Britain and Northern Ireland', 'United States of America', 'France', 'Germany']
    df = pd.read_csv(file_path)
    country_list_from_file = df['Country'].unique()
    valid_countries = [country for country in country_list if country in country_list_from_file]
    return valid_countries

def read_and_clean_data():
    # 读取数据
    df = pd.read_csv(file_path)
    # 获取有效国家列表
    country_list = get_country()
    # 提取指定国家的数据
    df = df[df['Country'].isin(country_list)]

    # 提取需要的列
    df = df[['Country', ' BMI ', 'Life expectancy ', 'Alcohol']]

    # 列名中英文映射
    column_mapping = {
        'Country': '国家',
        ' BMI ': '身体质量指数',
        'Life expectancy ': '预期寿命',
        'Alcohol': '酒精消费量'
    }
    df = df.rename(columns=column_mapping)

    # 国家名称中英文映射
    country_name_mapping = {
        'China': '中国',
        'Japan': '日本',
        'Republic of Korea': '韩国',
        'United Kingdom of Great Britain and Northern Ireland': '英国',
        'United States of America': '美国',
        'France': '法国',
        'Germany': '德国'
    }
    df['国家'] = df['国家'].map(country_name_mapping)

    # 处理缺失值：分组计算平均值并填充
    grouped = df.groupby('国家')
    for col in df.columns[1:]:  # 跳过 '国家' 列
        if df[col].isna().any():
            df[col] = grouped[col].transform(lambda x: x.fillna(x.mean()))

    # 处理异常值：使用箱型图方法
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), np.nan)
        df[col] = df[col].interpolate()

    # 确认缺失值已处理
    print(df.isna().sum())
    return df

def plot_bmi(cleaned_data):
    """
    绘制各国身体质量指数柱形图
    :param cleaned_data: 清洗后的数据 DataFrame
    """
    avg_bmi = cleaned_data.groupby('国家')['身体质量指数'].mean()
    countries = avg_bmi.index.tolist()  # 将 pandas.Index 转换为列表
    colors = plt.cm.tab10(np.arange(len(countries)))  # 生成不同颜色

    plt.figure(figsize=(10, 6))
    bars = plt.bar(countries, avg_bmi, color=colors)

    plt.xlabel('国家')
    plt.ylabel('身体质量指数')
    plt.title('各国平均身体质量指数')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 添加图例
    plt.legend(handles=bars, labels=countries)
    plt.tight_layout()
    plt.show()

def plot_life_expectancy(cleaned_data):
    """
    绘制各国预期寿命柱形图
    :param cleaned_data: 清洗后的数据 DataFrame
    """
    avg_life_expectancy = cleaned_data.groupby('国家')['预期寿命'].mean()
    countries = avg_life_expectancy.index.tolist()  # 将 pandas.Index 转换为列表
    colors = plt.cm.tab10(np.arange(len(countries)))  # 生成不同颜色

    plt.figure(figsize=(10, 6))
    bars = plt.bar(countries, avg_life_expectancy, color=colors)

    plt.xlabel('国家')
    plt.ylabel('预期寿命')
    plt.title('各国平均预期寿命')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 添加图例
    plt.legend(handles=bars, labels=countries)
    plt.tight_layout()
    plt.show()

def plot_alcohol(cleaned_data):
    """
    绘制各国酒精消费量柱形图
    :param cleaned_data: 清洗后的数据 DataFrame
    """
    avg_alcohol = cleaned_data.groupby('国家')['酒精消费量'].mean()
    countries = avg_alcohol.index.tolist()  # 将 pandas.Index 转换为列表
    colors = plt.cm.tab10(np.arange(len(countries)))  # 生成不同颜色

    plt.figure(figsize=(10, 6))
    bars = plt.bar(countries, avg_alcohol, color=colors)

    plt.xlabel('国家')
    plt.ylabel('酒精消费量')
    plt.title('各国平均酒精消费量')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 添加图例
    plt.legend(handles=bars, labels=countries)
    plt.tight_layout()
    plt.show()

def plot_pairplot(cleaned_data):
    """
    根据清洗后的数据绘制矩阵散点图，不同国家用不同颜色表示，并添加图例
    :param cleaned_data: 清洗后的数据 DataFrame
    """
    # 选择要绘制的列
    columns_to_plot = ['身体质量指数', '预期寿命', '酒精消费量']
    # 使用 hue 参数按国家分组，不同国家用不同颜色表示
    g = sns.pairplot(cleaned_data, vars=columns_to_plot, hue='国家', markers='o')
    # 添加图例
    g.add_legend()
    # 设置图标题，调整 y 参数，让标题位置更高
    g.fig.suptitle('身体质量指数、预期寿命和酒精消费量的矩阵散点图')
    # 调整子图布局，增加顶部空间
    plt.subplots_adjust(top=0.9)
    plt.show()

def summarize_data(cleaned_data):
    """
    根据清洗后的数据给出身体质量指数、预期寿命和酒精消费量关系的统计总结结果
    :param cleaned_data: 清洗后的数据 DataFrame
    :return: 包含统计信息的字符串
    """
    summary = []

    # 提取需要分析的列
    columns = ['身体质量指数', '预期寿命', '酒精消费量']
    data_to_analyze = cleaned_data[columns]

    # 计算相关系数矩阵
    corr_matrix = data_to_analyze.corr()

    # 分析身体质量指数与预期寿命的关系
    bmi_life_corr = corr_matrix.loc['身体质量指数', '预期寿命']
    if bmi_life_corr > 0.7:
        bmi_life_relationship = "高度正相关"
    elif 0.3 < bmi_life_corr <= 0.7:
        bmi_life_relationship = "中度正相关"
    elif -0.3 <= bmi_life_corr <= 0.3:
        bmi_life_relationship = "几乎无相关性"
    elif -0.7 <= bmi_life_corr < -0.3:
        bmi_life_relationship = "中度负相关"
    else:
        bmi_life_relationship = "高度负相关"
    summary.append(f"身体质量指数与预期寿命的相关系数为 {bmi_life_corr:.2f}，二者{ bmi_life_relationship }。")

    # 分析身体质量指数与酒精消费量的关系
    bmi_alcohol_corr = corr_matrix.loc['身体质量指数', '酒精消费量']
    if bmi_alcohol_corr > 0.7:
        bmi_alcohol_relationship = "高度正相关"
    elif 0.3 < bmi_alcohol_corr <= 0.7:
        bmi_alcohol_relationship = "中度正相关"
    elif -0.3 <= bmi_alcohol_corr <= 0.3:
        bmi_alcohol_relationship = "几乎无相关性"
    elif -0.7 <= bmi_alcohol_corr < -0.3:
        bmi_alcohol_relationship = "中度负相关"
    else:
        bmi_alcohol_relationship = "高度负相关"
    summary.append(f"身体质量指数与酒精消费量的相关系数为 {bmi_alcohol_corr:.2f}，二者{ bmi_alcohol_relationship }。")

    # 分析预期寿命与酒精消费量的关系
    life_alcohol_corr = corr_matrix.loc['预期寿命', '酒精消费量']
    if life_alcohol_corr > 0.7:
        life_alcohol_relationship = "高度正相关"
    elif 0.3 < life_alcohol_corr <= 0.7:
        life_alcohol_relationship = "中度正相关"
    elif -0.3 <= life_alcohol_corr <= 0.3:
        life_alcohol_relationship = "几乎无相关性"
    elif -0.7 <= life_alcohol_corr < -0.3:
        life_alcohol_relationship = "中度负相关"
    else:
        life_alcohol_relationship = "高度负相关"
    summary.append(f"预期寿命与酒精消费量的相关系数为 {life_alcohol_corr:.2f}，二者{ life_alcohol_relationship }。")

    return '\n'.join(summary)

if __name__ == "__main__":
    cleaned_data = read_and_clean_data()
    # plot_bmi(cleaned_data)
    # plot_life_expectancy(cleaned_data)
    # plot_alcohol(cleaned_data)
    plot_pairplot(cleaned_data)
    # summary_result = summarize_data(cleaned_data)
    # print(summary_result)