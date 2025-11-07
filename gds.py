import numpy as np
import csv
import os

def run_thermal_simulation(pillar_temperatures, grid_size=27, pillar_side=3, subgrid_side=9, tolerance=0.1, max_iterations=10000):
    """
    根据给定的9个恒温柱温度，进行热力学模拟。

    Args:
        pillar_temperatures (list or np.ndarray): 包含9个恒温柱温度的列表或数组。
        grid_size (int): 模拟区域的边长。
        pillar_side (int): 恒温柱的边长。
        subgrid_side (int): 子区域的边长。
        tolerance (float): 收敛容差。
        max_iterations (int): 最大迭代次数。

    Returns:
        numpy.ndarray: 代表稳态温度的27x27网格。
    """
    if len(pillar_temperatures) != 9:
        raise ValueError("pillar_temperatures 必须包含9个温度值。")

    temperature_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    is_pillar = np.zeros_like(temperature_grid, dtype=bool)
    pillar_offset = (subgrid_side - pillar_side) // 2
    
    temp_iterator = iter(pillar_temperatures)

    # 根据输入的温度列表设置恒温柱
    for i in range(0, grid_size, subgrid_side):
        for j in range(0, grid_size, subgrid_side):
            pillar_temp = next(temp_iterator)
            row_start, row_end = i + pillar_offset, i + pillar_offset + pillar_side
            col_start, col_end = j + pillar_offset, j + pillar_offset + pillar_side
            temperature_grid[row_start:row_end, col_start:col_end] = pillar_temp
            is_pillar[row_start:row_end, col_start:col_end] = True

    # 弛豫计算主循环
    for iteration in range(max_iterations):
        previous_grid = temperature_grid.copy()
        padded_grid = np.pad(temperature_grid, pad_width=1, mode='wrap')

        for r in range(grid_size):
            for c in range(grid_size):
                if not is_pillar[r, c]:
                    temperature_grid[r, c] = (padded_grid[r, c+1] +   # Left
                                            padded_grid[r+2, c+1] + # Right
                                            padded_grid[r+1, c] +   # Top
                                            padded_grid[r+1, c+2]) / 4 # Bottom
        
        max_change = np.max(np.abs(temperature_grid - previous_grid))
        if max_change < tolerance:
            # print(f"Converged after {iteration + 1} iterations.") # 在批量生成时可以注释掉此行
            break
    
    return temperature_grid

if __name__ == '__main__':
    # --- 参数设置 ---
    num_datasets_to_generate = 10000  # 您希望生成的数据集的总数
    output_filename = 'thermo_simulation_dataset.csv' # 输出文件名
    min_temp = 1
    max_temp = 100

    print(f"准备生成 {num_datasets_to_generate} 组数据集...")
    
    # --- 创建CSV文件并写入表头 ---
    # 表头包含9个输入和729个输出 (27*27)
    header = [f'input_temp_{i+1}' for i in range(9)]
    header.extend([f'output_temp_{r}_{c}' for r in range(27) for c in range(27)])

    # 使用 'w' 模式打开文件，如果文件已存在则会覆盖
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        

        # --- 循环生成每一行数据 ---
        for i in range(num_datasets_to_generate):
            # 1. 随机生成9个输入温度
            input_temperatures = np.random.uniform(min_temp, max_temp, 9)

            # 2. 运行模拟得到输出
            output_grid = run_thermal_simulation(pillar_temperatures=input_temperatures, tolerance=0.1)

            # 3. 将27x27的输出网格展平为一维数组
            output_flat = output_grid.flatten()
            
            # 4. 将输入和输出拼接成一行
            #    使用 list() 转换确保格式统一
            csv_row = list(input_temperatures) + list(output_flat)

            # 5. 将该行写入CSV文件
            writer.writerow(csv_row)
            
            # 打印进度
            print(f"已生成并保存数据集 {i + 1}/{num_datasets_to_generate}")

    print(f"\n任务完成！数据集已成功保存到文件 '{os.path.abspath(output_filename)}'")