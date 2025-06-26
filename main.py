import tkinter as tk
from tkinter import filedialog
from tool.analysis import read_and_clean_data, plot_bmi, plot_life_expectancy, plot_alcohol,plot_pairplot,summarize_data
'''
make by 2025.6.25 @atulearning 
'''

root = tk.Tk()
root.title("数据分析工具")
root.geometry("600x400")

# 定义选择文件的函数
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        global cleaned_data
        file_label.config(text=f"所选文件路径: {file_path}")
        # 读取并清洗数据
        cleaned_data = read_and_clean_data()

# 创建选择文件的按钮
top_text = tk.Label(root, text="根据身体质量指数、预期寿命 和 酒精消费量\n分析中日韩英美法德")
top_text.pack()
select_button = tk.Button(root, text="选择文件", command=select_file)
select_button.pack(pady=10)

# 创建显示文件路径的标签
file_label = tk.Label(root, text="未选择分析文件")
file_label.pack(pady=10)

# 创建三个按钮的框架，让按钮在同一行
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# 假设 cleaned_data 是全局变量
cleaned_data = None

def call_plot_bmi():
    if cleaned_data is not None:
        plot_bmi(cleaned_data)
    else:
        file_label.config(text="请先选择文件并完成数据清洗")

def call_plot_life_expectancy():
    if cleaned_data is not None:
        plot_life_expectancy(cleaned_data)
    else:
        file_label.config(text="请先选择文件并完成数据清洗")

def call_plot_alcohol():
    if cleaned_data is not None:
        plot_alcohol(cleaned_data)
    else:
        file_label.config(text="请先选择文件并完成数据清洗")

def call_plot_pairplot():
    if cleaned_data is not None:
        plot_pairplot(cleaned_data)
    else:
        file_label.config(text="请先选择文件并完成数据清洗")

def show_correlation_result():
    if cleaned_data is not None:
        result = summarize_data(cleaned_data)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, result)
    else:
        file_label.config(text="请先选择文件并完成数据清洗")

# 创建三个按钮
bmi_button = tk.Button(button_frame, text="绘制 BMI 柱形图", command=call_plot_bmi)
bmi_button.pack(side=tk.LEFT, padx=5)

life_expectancy_button = tk.Button(button_frame, text="绘制预期寿命柱形图", command=call_plot_life_expectancy)
life_expectancy_button.pack(side=tk.LEFT, padx=5)

alcohol_button = tk.Button(button_frame, text="绘制酒精消费量柱形图", command=call_plot_alcohol)
alcohol_button.pack(side=tk.LEFT, padx=5)

# 创建绘制矩阵图按钮
pairplot_button = tk.Button(button_frame, text="绘制矩阵图", command=call_plot_pairplot)
pairplot_button.pack(side=tk.LEFT, padx=5)

# 创建显示相关性结果按钮
correlation_button = tk.Button(button_frame, text="显示相关性结果", command=show_correlation_result)
correlation_button.pack(side=tk.LEFT, padx=5)

# 创建文本框用于显示相关性结果
result_text = tk.Text(root, height=10, width=60)
result_text.pack(pady=10)

root.mainloop()
